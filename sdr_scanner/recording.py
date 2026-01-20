"""
Channel recording management with Broadcast WAV format support
"""

import asyncio
import collections
import datetime
import json
import logging
import numpy
import numpy.typing
import os
import soundfile
import struct
import threading
import typing
import uuid


logger = logging.getLogger(__name__)


class ChannelRecorder:
	"""
	Manages recording for a single channel with memory buffering and async disk writing
	"""

	def __init__(
		self,
		channel_freq: float,
		channel_index: int,
		band_name: str,
		audio_sample_rate: int,
		buffer_size_seconds: float,
		disk_flush_interval_seconds: float,
		audio_output_dir: str,
		modulation: str = "Unknown"
	) -> None:
		"""
		Initialize a channel recorder

		Args:
			channel_freq: Channel center frequency in Hz
			channel_index: Channel index number
			band_name: Name of the band (e.g., 'pmr')
			audio_sample_rate: Output audio sample rate in Hz
			buffer_size_seconds: Maximum buffer size in seconds
			disk_flush_interval_seconds: How often to flush to disk
			audio_output_dir: Output directory path
			modulation: Modulation type (e.g., 'NFM', 'AM')
		"""
		self.channel_freq = channel_freq
		self.channel_index = channel_index
		self.band_name = band_name
		self.audio_sample_rate = audio_sample_rate
		self.disk_flush_interval = disk_flush_interval_seconds
		self.modulation = modulation

		# Calculate maximum buffer size in samples
		max_buffer_samples = int(buffer_size_seconds * audio_sample_rate)
		self.max_buffer_samples = max(0, max_buffer_samples)

		# Chunked buffer to reduce per-sample Python overhead.
		self.audio_buffer: collections.deque[numpy.typing.NDArray[numpy.float32]] = collections.deque()
		self.audio_buffer_samples = 0

		# Recording start time
		self.start_time = datetime.datetime.now()

		# Calculate TimeReference: samples since midnight (for multi-file sync)
		midnight = self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
		seconds_since_midnight = (self.start_time - midnight).total_seconds()
		self.time_reference = int(seconds_since_midnight * audio_sample_rate)

		date_str = self.start_time.strftime("%Y-%m-%d")
		time_str = self.start_time.strftime("%H-%M-%S")
		filename = f"{date_str}_{time_str}_{band_name}_{channel_index}.wav"
		self.filepath = os.path.join(audio_output_dir, date_str, filename)

		# Create output directory with date subdirectory if needed
		os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

		# Open WAV file for writing using soundfile (supports Broadcast WAV)

		self.wav_file = soundfile.SoundFile(
			self.filepath,
			mode='w',
			samplerate=audio_sample_rate,
			channels=1,
			subtype='PCM_16',
			format='WAV'
		)

		# Prepare BEXT metadata for soundfile
		# Note: soundfile uses 'extra_info' parameter for BEXT chunk

		# Broadcast WAV metadata

		bwf_description_data = {
			"band": band_name,
			"channel_index": channel_index,
			"channel_freq": channel_freq,
		}

		bwf_description = json.dumps(bwf_description_data, separators=(",", ":"), ensure_ascii=True)

		if len(bwf_description) > 256:
			logger.warning("BWF description exceeds 256 characters and will be truncated.")

		bwf_originator = "SDR Scanner"
		bwf_originator_reference = str(uuid.uuid4())[:32]  # Truncate to 32 chars

		bwf_coding_history = (
			f"A=PCM,F={audio_sample_rate},W=16,M=mono,T={modulation};"
			f"Frequency={channel_freq/1e6:.5f}MHz\r\n"
		)

		# Store BEXT metadata to write on close

		self.bext_metadata = {
			'description': bwf_description,
			'originator': bwf_originator,
			'originator_reference': bwf_originator_reference,
			'origination_date': self.start_time.strftime('%Y-%m-%d'),
			'origination_time': self.start_time.strftime('%H:%M:%S'),
			'time_reference': self.time_reference,
			'version': 1,
			'coding_history': bwf_coding_history
		}

		# Track total samples written (for logging)
		self.total_samples_written = 0

		# Async flush task (will be set by caller) - can be Task or Future depending on how it's created
		self.flush_task: typing.Any = None

		# Flag to indicate if recorder is closing
		self.closing = False

		self._write_lock = threading.Lock()
		self._buffer_lock = threading.Lock()

		logger.info(f"Started recording channel {channel_index} (f = {channel_freq/1e6:.5f} MHz) to {self.filepath}")

	def append_audio (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Append audio samples to the buffer (non-blocking)

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		if self.closing:
			return

		with self._buffer_lock:

			if self.max_buffer_samples <= 0:
				return

			incoming_len = len(samples)

			if incoming_len == 0:
				return

			if incoming_len >= self.max_buffer_samples:

				dropped = self.audio_buffer_samples + (incoming_len - self.max_buffer_samples)

				if dropped > 0:
					logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {dropped} oldest samples")

				self.audio_buffer.clear()
				self.audio_buffer_samples = 0
				samples = samples[-self.max_buffer_samples:]
				self.audio_buffer.append(samples)
				self.audio_buffer_samples = len(samples)
				return

			overflow = self.audio_buffer_samples + incoming_len - self.max_buffer_samples

			if overflow > 0:

				self._drop_oldest_samples(overflow)
				logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {overflow} oldest samples")

			self.audio_buffer.append(samples)
			self.audio_buffer_samples += incoming_len

	def _drop_oldest_samples (self, count:int) -> None:

		"""
		Drop a number of samples from the start of the chunked buffer.
		"""

		remaining = count
		while remaining > 0 and self.audio_buffer:
			chunk = self.audio_buffer[0]
			if len(chunk) <= remaining:
				self.audio_buffer.popleft()
				self.audio_buffer_samples -= len(chunk)
				remaining -= len(chunk)
			else:
				self.audio_buffer[0] = chunk[remaining:]
				self.audio_buffer_samples -= remaining
				remaining = 0

	async def _flush_to_disk_periodically (self) -> None:

		"""
		Async task that periodically flushes buffer to disk
		Runs until cancelled
		"""

		try:

			while not self.closing:

				await asyncio.sleep(self.disk_flush_interval)
				await self._flush_buffer_to_disk()

		except asyncio.CancelledError:
			# Let close() handle final flush and file shutdown.
			return

	async def _flush_buffer_to_disk (self) -> None:

		"""
		Flush accumulated buffer samples to disk in executor (non-blocking)
		"""

		with self._buffer_lock:

			if self.audio_buffer_samples == 0:
				return

			if len(self.audio_buffer) == 1:
				samples_to_write = self.audio_buffer[0]
			else:
				samples_to_write = numpy.concatenate(list(self.audio_buffer)).astype(numpy.float32, copy=False)

			self.audio_buffer.clear()
			self.audio_buffer_samples = 0

		await asyncio.get_running_loop().run_in_executor(None, self._write_samples_to_wav, samples_to_write)

	def _write_samples_to_wav (self, samples:numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Write audio samples to WAV file (runs in executor thread)

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		# soundfile expects float32 samples in range [-1.0, 1.0] for PCM_16 output

		# It will automatically convert to int16 internally
		with self._write_lock:

			# Write to WAV file (soundfile handles float32 to int16 conversion)
			self.wav_file.write(samples)
			self.total_samples_written += len(samples)

	async def close (self) -> None:

		"""
		Close the recorder, flush remaining buffer, and finalize WAV file
		"""

		self.closing = True

		# Cancel flush task if running
		# Note: flush_task is a concurrent.futures.Future, not an asyncio.Task
		if self.flush_task and not self.flush_task.done():

			self.flush_task.cancel()

			try:
				await asyncio.wait_for(asyncio.wrap_future(self.flush_task), timeout=5)
			except asyncio.TimeoutError:
				pass
			except asyncio.CancelledError:
				pass
			except Exception:
				pass

		# Final flush of any remaining samples
		await self._flush_buffer_to_disk()

		# Close WAV file (this writes headers)
		self.wav_file.close()

		# Write BEXT chunk for Broadcast Wave Format
		if self.bext_metadata:
			self._write_bext_chunk()

		duration_seconds = self.total_samples_written / self.audio_sample_rate
		logger.info(f"Stopped recording channel {self.channel_index} (f = {self.channel_freq/1e6:.5f} MHz) - Duration: {duration_seconds:.1f}s, File: {self.filepath}")

	def _write_bext_chunk (self) -> None:

		"""
		Write BEXT chunk to WAV file for Broadcast Wave Format
		This modifies the WAV file in-place after closing
		"""

		# Safety check (should never happen due to check in close())
		if not self.bext_metadata:
			return

		# Read the existing WAV file
		with open(self.filepath, 'rb') as f:

			riff_header = f.read(12)  # 'RIFF' + size + 'WAVE'

			if riff_header[:4] != b'RIFF' or riff_header[8:12] != b'WAVE':
				logger.warning(f"File {self.filepath} is not a valid WAV file, cannot add BEXT chunk")
				return

			remaining_data = f.read()

		# Build BEXT chunk (EBU Tech 3285)
		description = self.bext_metadata['description'].encode('ascii', errors='replace')[:256].ljust(256, b'\x00')
		originator = self.bext_metadata['originator'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		originator_ref = self.bext_metadata['originator_reference'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		origination_date = self.bext_metadata['origination_date'].encode('ascii')[:10].ljust(10, b'\x00')
		origination_time = self.bext_metadata['origination_time'].encode('ascii')[:8].ljust(8, b'\x00')
		time_reference = self.bext_metadata['time_reference']
		version = self.bext_metadata['version']
		umid = b'\x00' * 64  # UMID (64 bytes, all zeros)
		loudness_value = 0
		loudness_range = 0
		max_true_peak = 0
		max_momentary = 0
		max_short_term = 0
		reserved = b'\x00' * 180

		coding_history = self.bext_metadata['coding_history'].encode('ascii', errors='replace')

		# Calculate BEXT chunk size (602 fixed bytes + coding history length)
		bext_data_size = 602 + len(coding_history)

		# Build BEXT chunk
		bext_chunk = b'bext'
		bext_chunk += struct.pack('<I', bext_data_size)  # Chunk size (little-endian)
		bext_chunk += description
		bext_chunk += originator
		bext_chunk += originator_ref
		bext_chunk += origination_date
		bext_chunk += origination_time
		bext_chunk += struct.pack('<Q', time_reference)  # 64-bit time reference
		bext_chunk += struct.pack('<H', version)  # Version (16-bit)
		bext_chunk += umid
		bext_chunk += struct.pack('<H', loudness_value)
		bext_chunk += struct.pack('<H', loudness_range)
		bext_chunk += struct.pack('<H', max_true_peak)
		bext_chunk += struct.pack('<H', max_momentary)
		bext_chunk += struct.pack('<H', max_short_term)
		bext_chunk += reserved
		bext_chunk += coding_history

		# Pad to even boundary if needed
		if len(bext_chunk) % 2 != 0:
			bext_chunk += b'\x00'

		# Calculate new RIFF size
		original_riff_size = struct.unpack('<I', riff_header[4:8])[0]
		new_riff_size = original_riff_size + len(bext_chunk)

		# Write the modified WAV file
		with open(self.filepath, 'wb') as f:

			# Write RIFF header with updated size
			f.write(b'RIFF')
			f.write(struct.pack('<I', new_riff_size))
			f.write(b'WAVE')

			# Write BEXT chunk
			f.write(bext_chunk)

			# Write remaining original data
			f.write(remaining_data)

		logger.debug(f"Added BEXT chunk to {self.filepath}")
