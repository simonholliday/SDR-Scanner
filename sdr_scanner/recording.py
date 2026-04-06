"""
Channel recording management with Broadcast WAV format support.

Handles buffered audio recording to WAV files with industry-standard Broadcast
WAV (BWF) metadata. The recorder uses a memory buffer to avoid blocking the main
processing thread, and flushes to disk periodically in the background.

Broadcast WAV format includes additional metadata chunks (BEXT) that store
information like channel frequency, timestamp, and encoding history - useful
for archival and post-processing.
"""

import asyncio
import datetime
import json
import logging
import os
import struct
import threading
import typing
import uuid

import numpy
import numpy.typing
import soundfile

import sdr_scanner.dsp.noise_reduction


logger = logging.getLogger(__name__)


class ChannelRecorder:
	
	"""
	Manages buffered audio recording for a single channel.

	This class handles the complete lifecycle of recording a channel's audio:
	1. Buffers incoming audio samples in memory (circular buffer, drops oldest on overflow)
	2. Periodically flushes buffer to disk in the background (non-blocking async I/O)
	3. Applies noise reduction and soft limiting before writing
	4. Finalizes the WAV file with Broadcast WAV metadata when closed

	The buffering approach prevents blocking the main signal processing thread,
	which is critical for real-time operation. If processing falls behind, old
	samples are dropped rather than causing the entire system to stall.

	Broadcast WAV metadata (BEXT chunk) includes channel frequency, timestamp,
	and modulation type, making the recordings self-documenting and archival-friendly.
	"""

	def __init__ (
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
		soft_limit_drive: float = 2.0,
		noise_reduction_enabled: bool = True
	) -> None:

		"""
		Initialize a channel recorder and open the output WAV file.

		Creates the output directory structure, opens a WAV file for writing,
		prepares Broadcast WAV metadata, and sets up the memory buffer for
		accumulating audio samples.

		Files are organized as: output_dir/YYYY-MM-DD/band_name/filename.wav
		This hierarchy makes it easy to find recordings by date and band.

		Args:
			channel_freq: Channel center frequency in Hz (e.g., 446.00625e6)
			channel_index: Channel number for display (e.g., 0 for PMR channel 1)
			band_name: Name of the band (e.g., 'pmr', 'airband')
			audio_sample_rate: Output audio sample rate in Hz (e.g., 16000)
			buffer_size_seconds: Maximum buffer size in seconds (prevents unbounded memory growth)
			disk_flush_interval_seconds: How often to flush buffer to disk (trade-off: latency vs overhead)
			audio_output_dir: Base output directory path
			modulation: Modulation type (e.g., 'NFM', 'AM') - stored in metadata
			filename_suffix: Optional suffix for filename (e.g., SNR and device info)
			soft_limit_drive: Soft limiter drive amount (1.0-4.0, higher = more aggressive)
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
		self.noise_reduction_enabled = noise_reduction_enabled
		self.initial_noise_floor_db: float | None = None

		# Pre-allocated circular buffer: a fixed NumPy array with modulo wrap-around.
		# Eliminates per-flush concatenation and Python-level iteration for drops.
		max_buffer_samples = max(0, int(buffer_size_seconds * audio_sample_rate))
		self.max_buffer_samples = max_buffer_samples
		self._ring = numpy.zeros(max_buffer_samples, dtype=numpy.float32) if max_buffer_samples > 0 else numpy.array([], dtype=numpy.float32)
		self._ring_write_head: int = 0      # Next write position (wraps via modulo)
		self._ring_frames_written: int = 0  # Monotonic total frames written
		self._ring_frames_flushed: int = 0  # Monotonic total frames already flushed

		# Recording start time (used for filename and BWF metadata)
		self.start_time = datetime.datetime.now()

		# Calculate TimeReference: sample count since midnight
		# This is part of the Broadcast WAV spec and allows precise synchronization
		# between multiple recordings (they all share the same midnight reference)
		midnight = self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
		seconds_since_midnight = (self.start_time - midnight).total_seconds()
		self.time_reference = int(seconds_since_midnight * audio_sample_rate)

		# Build filename with timestamp, band, and channel information
		# Format: YYYY-MM-DD_HH-MM-SS_band_channel_[suffix].wav
		# Example: 2026-01-25_14-30-45_pmr_0_12.5dB_rtlsdr_0.wav
		date_str = self.start_time.strftime("%Y-%m-%d")
		time_str = self.start_time.strftime("%H-%M-%S")

		filename = f"{date_str}_{time_str}_{band_name}_{channel_index}"

		if filename_suffix:
			filename += "_" + filename_suffix

		filename += ".wav"

		# Organize files: base_dir/YYYY-MM-DD/band_name/filename.wav
		# This hierarchical structure makes it easy to manage recordings by date and band
		self.filepath = os.path.abspath(os.path.join(audio_output_dir, date_str, band_name, filename))

		# Create the directory structure if it doesn't exist
		# Note: We use a robust check here because network filesystems (CIFS/SMB)
		# can have metadata lag that causes os.makedirs(exist_ok=True) to fail
		# if another process/thread created the directory at the exact same time.
		dir_path = os.path.dirname(self.filepath)
		if not os.path.exists(dir_path):
			try:
				os.makedirs(dir_path, exist_ok=True)
			except FileExistsError:
				# On some network filesystems, mkdir can fail with EEXIST even
				# if stat() hasn't updated its cache yet. We ignore this safely.
				pass

		# Open WAV file for writing using soundfile library
		# soundfile is chosen because it supports Broadcast WAV extensions
		# PCM_16 = 16-bit signed integer audio (standard CD quality)

		self.wav_file = soundfile.SoundFile(
			self.filepath,
			mode='w',
			samplerate=audio_sample_rate,
			channels=1,  # Mono (single channel)
			subtype='PCM_16',  # 16-bit PCM encoding
			format='WAV'
		)

		# Prepare Broadcast WAV (BWF) metadata according to EBU Tech 3285 standard
		# BEXT chunk contains machine-readable metadata about the recording
		# This makes the files self-documenting and suitable for archival

		# Description field: store channel info as JSON (max 256 chars in spec)
		# This allows easy parsing of metadata without manual filename parsing

		bwf_description_data = {
			"band": band_name,
			"channel_index": channel_index,
			"channel_freq": channel_freq,
		}

		bwf_description = json.dumps(bwf_description_data, separators=(",", ":"), ensure_ascii=True)

		if len(bwf_description) > 256:
			logger.warning("BWF description exceeds 256 characters and will be truncated.")

		# Originator: software that created this file
		bwf_originator = "SDR Scanner"
		# Originator reference: unique ID for this recording (UUID ensures uniqueness)
		bwf_originator_reference = str(uuid.uuid4())[:32]  # BWF spec limits to 32 chars

		# Coding history: human-readable text describing the signal chain
		# Format: A=algorithm, F=sample rate, W=bit depth, M=channels, T=transformation
		# This follows EBU recommendations for documenting audio processing
		bwf_coding_history = (
			f"A=PCM,F={audio_sample_rate},W=16,M=mono,T={modulation};"
			f"Frequency={channel_freq/1e6:.5f}MHz\r\n"
		)

		# Store BEXT metadata to write when file is closed
		# We can't write it now because we don't know the final file size yet
		# (BEXT chunk is appended after all audio data is written)

		self.bext_metadata = {
			'description': bwf_description,
			'originator': bwf_originator,
			'originator_reference': bwf_originator_reference,
			'origination_date': self.start_time.strftime('%Y-%m-%d'),
			'origination_time': self.start_time.strftime('%H:%M:%S'),
			'time_reference': self.time_reference,
			'version': 1,  # BEXT version 1
			'coding_history': bwf_coding_history
		}

		# Track total samples written (for logging)
		self.total_samples_written = 0

		# Async flush task (will be set by caller) - can be Task or Future depending on how it's created
		self.flush_task: typing.Any = None

		# Flag to indicate if recorder is closing
		self.closing = False
		self.noise_mag: numpy.ndarray | None = None

		self._write_lock = threading.Lock()
		self._buffer_lock = threading.Lock()

		logger.debug(f"Started recording channel {channel_index} (f = {channel_freq/1e6:.5f} MHz) to {self.filepath}")

	def append_audio (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Append audio samples to the pre-allocated circular buffer (non-blocking).

		The ring buffer silently overwrites the oldest samples when full,
		keeping recordings aligned with real-time activity.  No Python-level
		iteration or per-chunk allocation is required.

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		if self.closing:
			return

		with self._buffer_lock:

			cap = self.max_buffer_samples
			if cap <= 0:
				return

			n = len(samples)
			if n == 0:
				return

			if n >= cap:
				# Incoming chunk is larger than the whole buffer: keep only the tail.
				overflow = self._ring_frames_written - self._ring_frames_flushed + n - cap
				if overflow > 0:
					logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {overflow} oldest samples")
				samples = samples[-cap:]
				n = cap
				self._ring[:n] = samples
				self._ring_write_head = n % cap
				self._ring_frames_written += n
				return

			unflushed = self._ring_frames_written - self._ring_frames_flushed
			if unflushed + n > cap:
				overflow = unflushed + n - cap
				logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {overflow} oldest samples")
				# Advance the flushed pointer to discard oldest unflushed data
				self._ring_frames_flushed += overflow

			head = self._ring_write_head
			space_to_end = cap - head
			if n <= space_to_end:
				self._ring[head:head + n] = samples
			else:
				self._ring[head:] = samples[:space_to_end]
				remainder = n - space_to_end
				self._ring[:remainder] = samples[space_to_end:]

			self._ring_write_head = (head + n) % cap
			self._ring_frames_written += n

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
		Flush accumulated buffer samples to disk in executor (non-blocking).

		Reads all unflushed samples from the ring buffer into a contiguous
		copy (so the writer thread owns its data), then advances the flushed
		pointer.  No concatenation of separate chunks is needed.
		"""

		with self._buffer_lock:

			n_unflushed = self._ring_frames_written - self._ring_frames_flushed
			if n_unflushed <= 0:
				return

			cap = self.max_buffer_samples
			# Oldest unflushed position in the ring
			start = self._ring_frames_flushed % cap

			if n_unflushed <= cap - start:
				# Contiguous region — single slice, copy to detach from ring
				samples_to_write = self._ring[start:start + n_unflushed].copy()
			else:
				# Wrapped — two slices
				tail_len = cap - start
				samples_to_write = numpy.empty(n_unflushed, dtype=numpy.float32)
				samples_to_write[:tail_len] = self._ring[start:]
				samples_to_write[tail_len:] = self._ring[:n_unflushed - tail_len]

			self._ring_frames_flushed = self._ring_frames_written

		await asyncio.get_running_loop().run_in_executor(None, self._write_samples_to_wav, samples_to_write)

	def _write_samples_to_wav (self, samples:numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Write audio samples to WAV file (runs in executor thread)

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		# Apply noise reduction using faster spectral subtraction method
		# This is 5-10x faster than the noisereduce library
		if self.noise_reduction_enabled:
			try:
				samples, self.noise_mag = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(
					samples, self.audio_sample_rate, oversub=0.7, floor=0.06,
					noise_mag=self.noise_mag, noise_floor_db=self.initial_noise_floor_db,
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
		Close the recorder, flush remaining buffer, and finalize the WAV file.

		Ensures any background flush task is stopped, writes remaining audio,
		closes the WAV file handle, and appends the BEXT metadata chunk so the
		recording is self-describing.
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

		The BEXT chunk stores machine-readable metadata (frequency, timestamps,
		encoding history). Appending it avoids rewriting the entire file; we
		only patch the RIFF size field after the append (O(1)).
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
		reserved = b'\x00' * 180 # Version 1 has 180 reserved bytes after the 10 loudness bytes

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
