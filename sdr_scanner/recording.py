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
import sdr_scanner.dsp.noise_reduction
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
		modulation: str = "Unknown",
		filename_suffix: str = None,
		soft_limit_drive: float = 2.0
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
			filename_suffix: An optional string to be added to the auto-generated filename
			soft_limit_drive: Soft limiter drive amount (higher = stronger limiting)
		"""
		self.channel_freq = channel_freq
		self.channel_index = channel_index
		self.band_name = band_name
		self.audio_sample_rate = audio_sample_rate
		self.disk_flush_interval = disk_flush_interval_seconds
		self.modulation = modulation
		# Precompute soft limiter parameters for efficiency
		self.soft_limit_drive = max(0.1, float(soft_limit_drive))
		self.soft_limit_scale = 1.0 / numpy.tanh(self.soft_limit_drive)

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

		filename = f"{date_str}_{time_str}_{band_name}_{channel_index}"

		if filename_suffix:
			filename += "_" + filename_suffix

		filename += ".wav"

		self.filepath = os.path.join(audio_output_dir, date_str, band_name, filename)

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

		logger.debug(f"Started recording channel {channel_index} (f = {channel_freq/1e6:.5f} MHz) to {self.filepath}")

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

		# Apply noise reduction using faster spectral subtraction method
		# This is 5-10x faster than the noisereduce library
		try:
			samples = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(
				samples, self.audio_sample_rate, oversub=0.7, floor=0.06
			)
		except Exception as exc:
			logger.warning(f"Noise reduction failed for {self.filepath}: {exc}")

		# Apply soft limiter using precomputed parameters
		if samples.size > 0 and self.soft_limit_drive > 0:
			samples = numpy.tanh(samples * self.soft_limit_drive) * self.soft_limit_scale

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
			self._append_bext_chunk()

		duration_seconds = self.total_samples_written / self.audio_sample_rate
		logger.debug(f"Stopped recording channel {self.channel_index} (f = {self.channel_freq/1e6:.5f} MHz) - Duration: {duration_seconds:.1f}s, File: {self.filepath}")

	def _append_bext_chunk (self) -> None:
		"""
		Append BEXT chunk to the end of the WAV file and patch the RIFF header.
		This is O(1) compared to the previous O(N) rewrite.
		"""
		if not self.bext_metadata:
			return

		# Build BEXT chunk (EBU Tech 3285)
		description = self.bext_metadata['description'].encode('ascii', errors='replace')[:256].ljust(256, b'\x00')
		originator = self.bext_metadata['originator'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		originator_ref = self.bext_metadata['originator_reference'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		origination_date = self.bext_metadata['origination_date'].encode('ascii')[:10].ljust(10, b'\x00')
		origination_time = self.bext_metadata['origination_time'].encode('ascii')[:8].ljust(8, b'\x00')
		time_reference = self.bext_metadata['time_reference']
		version = self.bext_metadata['version']
		umid = b'\x00' * 64
		reserved = b'\x00' * 190 # Version 1 has 190 reserved bytes before coding history if not using loudness

		coding_history = self.bext_metadata['coding_history'].encode('ascii', errors='replace')
		bext_data_size = 602 + len(coding_history)

		bext_chunk = b'bext'
		bext_chunk += struct.pack('<I', bext_data_size)
		bext_chunk += description
		bext_chunk += originator
		bext_chunk += originator_ref
		bext_chunk += origination_date
		bext_chunk += origination_time
		bext_chunk += struct.pack('<Q', time_reference)
		bext_chunk += struct.pack('<H', version)
		bext_chunk += umid
		bext_chunk += b'\x00' * 10 # Loudness fields (10 bytes)
		bext_chunk += reserved
		bext_chunk += coding_history

		if len(bext_chunk) % 2 != 0:
			bext_chunk += b'\x00'

		try:
			# 1. Append the chunk to the end of the file
			with open(self.filepath, 'ab') as f:
				f.write(bext_chunk)

			# 2. Patch the RIFF header size
			with open(self.filepath, 'r+b') as f:
				# Get current file size - 8 bytes (RIFF and size field itself)
				f.seek(0, os.SEEK_END)
				new_riff_size = f.tell() - 8
				f.seek(4)
				f.write(struct.pack('<I', new_riff_size))
			
			logger.debug(f"Appended BEXT chunk to {self.filepath} (New RIFF size: {new_riff_size})")
		except Exception as e:
			logger.error(f"Failed to append BEXT chunk to {self.filepath}: {e}")
