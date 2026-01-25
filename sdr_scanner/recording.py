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
		soft_limit_drive: float = 2.0
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
		self.soft_limit_drive = float(soft_limit_drive)

		# Calculate maximum buffer size in samples (e.g., 5 seconds * 16000 Hz = 80000 samples)
		# This prevents memory from growing unbounded if disk writes fall behind
		max_buffer_samples = int(buffer_size_seconds * audio_sample_rate)
		self.max_buffer_samples = max(0, max_buffer_samples)

		# Chunked buffer: stores audio in variable-sized chunks rather than individual samples
		# This reduces Python overhead - appending one chunk is faster than appending many samples
		# Using deque allows efficient pop from front (when dropping old samples)
		self.audio_buffer: collections.deque[numpy.typing.NDArray[numpy.float32]] = collections.deque()
		self.audio_buffer_samples = 0  # Total samples currently in buffer

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
		self.filepath = os.path.join(audio_output_dir, date_str, band_name, filename)

		# Create the directory structure if it doesn't exist
		# exist_ok=True prevents errors if directory already exists (thread-safe)
		os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

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

		# Track total samples written (used to calculate recording duration on close)
		self.total_samples_written = 0

		# Background flush task reference (will be set by the scanner)
		# This task runs _flush_to_disk_periodically() in the async event loop
		# Type is Any because it could be Task or Future depending on how it's created
		self.flush_task: typing.Any = None

		# Flag to indicate if recorder is shutting down
		# Prevents new audio from being buffered during cleanup
		self.closing = False

		# Thread locks for thread-safe access to file and buffer
		# _write_lock: protects WAV file writes (called from executor thread)
		# _buffer_lock: protects audio_buffer (accessed from multiple threads)
		self._write_lock = threading.Lock()
		self._buffer_lock = threading.Lock()

		logger.debug(f"Started recording channel {channel_index} (f = {channel_freq/1e6:.5f} MHz) to {self.filepath}")

	def append_audio (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Add audio samples to the memory buffer (non-blocking, thread-safe).

		This is called from the signal processing thread every time a channel
		has new audio. The samples are added to an in-memory buffer and will
		be written to disk later by the background flush task.

		If the buffer is full (disk writes falling behind), the oldest samples
		are dropped to prevent unbounded memory growth. This is preferable to
		blocking, which would stall the entire scanning process.

		Args:
			samples: Audio samples as float32 normalized to range [-1.0, 1.0]
		"""

		# Don't accept new samples if we're shutting down
		if self.closing:
			return

		with self._buffer_lock:
			# Sanity check: if buffer size is disabled, don't buffer anything
			if self.max_buffer_samples <= 0:
				return

			incoming_len = len(samples)

			# Nothing to do if empty
			if incoming_len == 0:
				return

			# Edge case: incoming block is larger than entire buffer capacity
			# Keep only the most recent samples that fit in the buffer
			if incoming_len >= self.max_buffer_samples:
				dropped = self.audio_buffer_samples + (incoming_len - self.max_buffer_samples)

				if dropped > 0:
					logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {dropped} oldest samples")

				# Clear everything and keep only the tail of the new samples
				self.audio_buffer.clear()
				self.audio_buffer_samples = 0
				samples = samples[-self.max_buffer_samples:]
				self.audio_buffer.append(samples)
				self.audio_buffer_samples = len(samples)
				return

			# Normal case: check if adding these samples would overflow the buffer
			overflow = self.audio_buffer_samples + incoming_len - self.max_buffer_samples

			if overflow > 0:
				# Drop oldest samples to make room for new ones
				self._drop_oldest_samples(overflow)
				logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {overflow} oldest samples")

			# Add the new chunk to the buffer
			self.audio_buffer.append(samples)
			self.audio_buffer_samples += incoming_len

	def _drop_oldest_samples (self, count: int) -> None:

		"""
		Drop the oldest samples from the buffer to make room for new ones.

		Since the buffer stores audio in chunks (not individual samples), we
		may need to remove entire chunks or trim the first chunk. This is
		more efficient than maintaining a flat array and using array slicing.

		Args:
			count: Number of samples to drop from the front of the buffer
		"""

		remaining = count
		while remaining > 0 and self.audio_buffer:
			chunk = self.audio_buffer[0]
			if len(chunk) <= remaining:
				# Entire first chunk needs to be dropped
				self.audio_buffer.popleft()
				self.audio_buffer_samples -= len(chunk)
				remaining -= len(chunk)
			else:
				# Only drop part of the first chunk (trim it)
				self.audio_buffer[0] = chunk[remaining:]
				self.audio_buffer_samples -= remaining
				remaining = 0

	async def _flush_to_disk_periodically (self) -> None:

		"""
		Background task that flushes the buffer to disk at regular intervals.

		This runs continuously in the async event loop, sleeping most of the time
		and periodically waking up to write buffered audio to disk. By batching
		writes, we reduce file I/O overhead (opening/closing, system calls).

		The task runs until cancelled (when the channel turns off or the scanner
		shuts down). Cancellation is the normal way to stop this task.
		"""

		try:
			while not self.closing:
				# Sleep until next flush interval
				await asyncio.sleep(self.disk_flush_interval)
				# Write accumulated samples to disk
				await self._flush_buffer_to_disk()

		except asyncio.CancelledError:
			# Task was cancelled - this is normal during shutdown
			return

	async def _flush_buffer_to_disk (self) -> None:

		"""
		Write all buffered samples to disk without blocking the event loop.

		This method:
		1. Atomically extracts all samples from the buffer (with lock held)
		2. Concatenates chunks into a single array if needed
		3. Offloads the actual write to a thread pool (disk I/O can be slow)

		The buffer lock is held only briefly while extracting samples, not during
		the actual disk write. This keeps the main processing thread responsive.
		"""

		# Atomically grab all buffered samples and clear the buffer
		with self._buffer_lock:
			if self.audio_buffer_samples == 0:
				return  # Nothing to write

			# Optimization: if there's only one chunk, avoid concatenation overhead
			if len(self.audio_buffer) == 1:
				samples_to_write = self.audio_buffer[0]
			else:
				# Concatenate all chunks into a single contiguous array
				# copy=False avoids an extra copy if the data is already contiguous
				samples_to_write = numpy.concatenate(list(self.audio_buffer)).astype(numpy.float32, copy=False)

			# Clear the buffer now that we've extracted the data
			self.audio_buffer.clear()
			self.audio_buffer_samples = 0

		# Run the disk write in a thread pool to avoid blocking the event loop
		# File I/O can take unpredictable amounts of time (disk cache, OS buffering, etc.)
		await asyncio.get_running_loop().run_in_executor(None, self._write_samples_to_wav, samples_to_write)

	def _write_samples_to_wav (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Process and write audio samples to the WAV file.

		This runs in a thread pool executor (not the event loop) so blocking
		I/O doesn't stall async operations. The processing pipeline:

		1. Noise reduction: spectral subtraction to reduce background noise
		2. Soft limiting: prevents clipping while maintaining dynamic range
		3. WAV write: soundfile converts float32 to PCM_16 automatically

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		# Apply noise reduction to clean up the audio
		# Spectral subtraction estimates the noise spectrum and subtracts it from the signal
		# This is CPU-intensive but improves audio quality for weak signals
		try:
			# Alternative noisereduce library is available but slower:
			# samples = sdr_scanner.dsp.noise_reduction.apply_noisereduce(samples, self.audio_sample_rate)
			samples = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(samples, self.audio_sample_rate)
		except Exception as exc:
			logger.warning(f"Noise reduction failed for {self.filepath}: {exc}")

		# Apply soft limiter to prevent clipping (hard limiting causes distortion)
		# tanh() is a soft limiter: it asymptotically approaches ±1 but never exceeds it
		# Drive controls how aggressively we compress loud signals
		if samples.size > 0:
			drive = max(0.1, self.soft_limit_drive)
			# Normalize by tanh(drive) so that input of 1.0 maps to output of 1.0
			den = numpy.tanh(drive)
			if den != 0.0:
				samples = numpy.tanh(samples * drive) / den

		# Write to WAV file (thread-safe with lock)
		with self._write_lock:
			# soundfile automatically converts float32 [-1,1] to PCM_16 [-32768,32767]
			self.wav_file.write(samples)
			self.total_samples_written += len(samples)

	async def close (self) -> None:

		"""
		Finalize the recording and close the WAV file.

		This method performs a graceful shutdown:
		1. Sets the closing flag to prevent new samples from being buffered
		2. Cancels the periodic flush task (no more automatic flushes)
		3. Performs a final flush to write any remaining buffered samples
		4. Closes the WAV file (writes the WAV header with final size)
		5. Appends the Broadcast WAV metadata chunk

		Called when a channel turns OFF or the scanner shuts down.
		"""

		# Signal that we're closing (stops append_audio from accepting new samples)
		self.closing = True

		# Stop the periodic flush task if it's still running
		if self.flush_task and not self.flush_task.done():
			self.flush_task.cancel()

			# Wait for cancellation to complete (with timeout to avoid hanging)
			try:
				await asyncio.wait_for(asyncio.wrap_future(self.flush_task), timeout=5)
			except asyncio.TimeoutError:
				pass  # Task didn't stop in time, proceed anyway
			except asyncio.CancelledError:
				pass  # Task was cancelled (expected)
			except Exception:
				pass  # Other errors during cancellation, log and continue

		# Final flush: write any remaining samples in the buffer
		await self._flush_buffer_to_disk()

		# Close the WAV file (this writes the final WAV header with correct size)
		self.wav_file.close()

		# Append Broadcast WAV metadata chunk to the end of the file
		if self.bext_metadata:
			self._append_bext_chunk()

		# Log recording summary
		duration_seconds = self.total_samples_written / self.audio_sample_rate
		logger.debug(f"Stopped recording channel {self.channel_index} (f = {self.channel_freq/1e6:.5f} MHz) - Duration: {duration_seconds:.1f}s, File: {self.filepath}")

	def _append_bext_chunk (self) -> None:
		"""
		Append Broadcast WAV Extension (BEXT) chunk to the completed WAV file.

		The BEXT chunk is defined by EBU Tech 3285 and adds professional metadata
		to WAV files. It's widely supported by broadcast and archival systems.

		We append it after closing the WAV file (rather than embedding it during
		creation) because some libraries don't support custom chunks. This approach
		is efficient: O(1) append + small header patch, vs O(N) full file rewrite.

		The BEXT chunk contains:
		- Description (256 bytes): JSON metadata about the channel
		- Originator (32 bytes): software name
		- OriginatorReference (32 bytes): unique recording ID
		- OriginationDate/Time: when recording started
		- TimeReference: sample count since midnight (for sync)
		- CodingHistory: text describing the signal processing chain

		After appending the chunk, we update the RIFF header to reflect the new
		file size (RIFF format requires the size to be correct).
		"""
		if not self.bext_metadata:
			return

		# Build BEXT chunk according to EBU Tech 3285 specification
		# All text fields are ASCII, null-padded to fixed lengths

		# Description: 256 bytes, JSON-encoded channel info
		description = self.bext_metadata['description'].encode('ascii', errors='replace')[:256].ljust(256, b'\x00')
		# Originator: 32 bytes, software name
		originator = self.bext_metadata['originator'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		# OriginatorReference: 32 bytes, unique ID for this recording
		originator_ref = self.bext_metadata['originator_reference'].encode('ascii', errors='replace')[:32].ljust(32, b'\x00')
		# OriginationDate: 10 bytes, YYYY-MM-DD format
		origination_date = self.bext_metadata['origination_date'].encode('ascii')[:10].ljust(10, b'\x00')
		# OriginationTime: 8 bytes, HH:MM:SS format
		origination_time = self.bext_metadata['origination_time'].encode('ascii')[:8].ljust(8, b'\x00')
		# TimeReference: 64-bit unsigned int, samples since midnight
		time_reference = self.bext_metadata['time_reference']
		# Version: 16-bit unsigned int, BEXT version (we use version 1)
		version = self.bext_metadata['version']
		# UMID: 64 bytes, Unique Material Identifier (not used, set to zeros)
		umid = b'\x00' * 64
		# Reserved: 190 bytes for future use (version 1 spec)
		reserved = b'\x00' * 190  # Version 1 has 190 reserved bytes before coding history

		# CodingHistory: variable length, describes the signal processing chain
		coding_history = self.bext_metadata['coding_history'].encode('ascii', errors='replace')
		# Calculate total BEXT data size (602 fixed bytes + variable coding history)
		bext_data_size = 602 + len(coding_history)

		# Build the chunk: chunk ID + size + data
		bext_chunk = b'bext'  # Chunk identifier (4 bytes)
		bext_chunk += struct.pack('<I', bext_data_size)  # Chunk size (little-endian 32-bit)
		bext_chunk += description
		bext_chunk += originator
		bext_chunk += originator_ref
		bext_chunk += origination_date
		bext_chunk += origination_time
		bext_chunk += struct.pack('<Q', time_reference)  # 64-bit little-endian
		bext_chunk += struct.pack('<H', version)  # 16-bit little-endian
		bext_chunk += umid
		bext_chunk += b'\x00' * 10  # Loudness fields (10 bytes, not used)
		bext_chunk += reserved
		bext_chunk += coding_history

		# RIFF chunks must be word-aligned (even number of bytes)
		# Add a padding byte if needed
		if len(bext_chunk) % 2 != 0:
			bext_chunk += b'\x00'

		try:
			# Step 1: Append the BEXT chunk to the end of the WAV file
			# This is fast (O(1) seek to end + write) compared to rewriting the entire file
			with open(self.filepath, 'ab') as f:
				f.write(bext_chunk)

			# Step 2: Update the RIFF header to reflect the new file size
			# RIFF format: "RIFF" + size (32-bit) + "WAVE" + chunks...
			# The size field at offset 4 must equal (file_size - 8)
			with open(self.filepath, 'r+b') as f:
				# Get current file size
				f.seek(0, os.SEEK_END)
				# RIFF size = file size - 8 (excludes "RIFF" and size field itself)
				new_riff_size = f.tell() - 8
				# Seek to the size field (offset 4) and update it
				f.seek(4)
				f.write(struct.pack('<I', new_riff_size))

			logger.debug(f"Appended BEXT chunk to {self.filepath} (New RIFF size: {new_riff_size})")
		except Exception as e:
			logger.error(f"Failed to append BEXT chunk to {self.filepath}: {e}")
