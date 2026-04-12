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
import math
import os
import struct
import threading
import typing
import uuid

import numpy
import numpy.typing
import soundfile

import substation.constants
import substation.dsp.filters
import substation.dsp.noise_reduction


logger = logging.getLogger(__name__)


def _trim_carrier_transient_start (audio: numpy.typing.NDArray[numpy.float32], sample_rate: int) -> numpy.typing.NDArray[numpy.float32]:

	"""
	Remove a carrier key-ON transient from the start of the audio.

	Scans the first 50ms for a sharp spike preceded by silence.  A
	carrier transient is identified by: (1) short duration (<15ms),
	(2) peak amplitude much higher than the pre-spike region (ratio >
	CARRIER_TRANSIENT_RATIO).  Voice transients fail criterion 2
	because they are preceded by other voice content, not silence.
	"""

	scan_len = min(len(audio), int(sample_rate * 0.05))
	if scan_len < 10:
		return audio

	env = numpy.abs(audio[:scan_len])
	win = max(1, int(sample_rate * 0.0005))
	env_s = numpy.convolve(env, numpy.ones(win) / win, mode='same')

	spike_idx = int(numpy.argmax(env_s))
	spike_peak = float(env_s[spike_idx])

	# The pre-spike silence region ends before the smoothing window
	# reaches the peak — ensures the click's rising edge doesn't
	# contaminate the noise estimate.
	pre_end = max(0, spike_idx - win)
	min_pre = int(sample_rate * 0.003)

	if pre_end < min_pre:
		# Spike is at the very start — no pre-silence available.
		# Compare against signal body (20-50ms in) instead.
		body_start = int(sample_rate * 0.02)
		body_end = min(len(audio), int(sample_rate * 0.05))
		if body_end <= body_start:
			return audio
		body_rms = float(numpy.sqrt(numpy.mean(audio[body_start:body_end] ** 2)))
		if body_rms < 1e-6:
			return audio
		if spike_peak / body_rms < substation.constants.CARRIER_TRANSIENT_RATIO:
			return audio
		pre_rms = body_rms
	else:
		pre_rms = float(numpy.sqrt(numpy.mean(audio[:pre_end] ** 2)))
		if pre_rms < 1e-6:
			pre_rms = 1e-6
		if spike_peak / pre_rms < substation.constants.CARRIER_TRANSIENT_RATIO:
			return audio

	# Find where the spike decays back toward noise.
	threshold = pre_rms * substation.constants.CARRIER_TRANSIENT_RATIO
	decay_win = max(1, int(sample_rate * 0.002))
	spike_end = spike_idx
	for j in range(spike_idx, len(env_s)):
		if env_s[j] >= threshold:
			spike_end = j
		elif j - spike_end > decay_win:
			break

	if (spike_end - spike_idx) / sample_rate * 1000 > 15:
		return audio

	trim_point = min(spike_end + decay_win, len(audio))
	return audio[trim_point:]


def _trim_carrier_transient_end (audio: numpy.typing.NDArray[numpy.float32], sample_rate: int) -> numpy.typing.NDArray[numpy.float32]:

	"""
	Remove a carrier key-OFF transient from the end of the audio.

	Scans the last 500ms for a sharp spike followed by silence.  Same
	detection criteria as _trim_carrier_transient_start but applied to
	the recording tail.  Uses a wider scan window than the start
	because the key-OFF click may be followed by hold-timer noise.
	"""

	scan_len = min(len(audio), int(sample_rate * 0.5))
	if scan_len < 10:
		return audio

	tail = audio[-scan_len:]
	env = numpy.abs(tail)
	win = max(1, int(sample_rate * 0.0005))
	env_s = numpy.convolve(env, numpy.ones(win) / win, mode='same')

	spike_idx = int(numpy.argmax(env_s))
	spike_peak = float(env_s[spike_idx])

	post_start = min(scan_len, spike_idx + win)
	min_post = int(sample_rate * 0.003)
	if post_start > scan_len - min_post:
		return audio

	post_rms = float(numpy.sqrt(numpy.mean(tail[post_start:] ** 2)))
	if post_rms < 1e-6:
		post_rms = 1e-6

	if spike_peak / post_rms < substation.constants.CARRIER_TRANSIENT_RATIO:
		return audio

	threshold = post_rms * substation.constants.CARRIER_TRANSIENT_RATIO
	decay_win = max(1, int(sample_rate * 0.002))
	spike_start = spike_idx
	for j in range(spike_idx, -1, -1):
		if env_s[j] >= threshold:
			spike_start = j
		elif spike_start - j > decay_win:
			break

	if (spike_idx - spike_start) / sample_rate * 1000 > 15:
		return audio

	trim_point = len(audio) - scan_len + max(0, spike_start - decay_win)
	return audio[:trim_point]


class _BextMetadata (typing.TypedDict):

	"""
	Typed view of the Broadcast WAV (BEXT) metadata dict built at recorder
	construction and consumed by _append_bext_chunk().  Exists so mypy can
	resolve self.bext_metadata['description'].encode(...) to str.encode()
	rather than object.encode().  No runtime overhead — TypedDict is a
	compile-time annotation only.
	"""

	description: str
	originator: str
	originator_reference: str
	origination_date: str
	origination_time: str
	time_reference: int
	version: int
	coding_history: str


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
		filename_suffix: str | None = None,
		soft_limit_drive: float = 1.25,
		noise_reduction_enabled: bool = True,
		trim_carrier_transients: bool = False,
		fade_in_ms: float | None = None,
		fade_out_ms: float | None = None,
		dynamics_curve_enabled: bool = False,
		dynamics_curve_config: typing.Any = None,
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
			channel_index: Channel number for display (1-based, e.g., 1 for PMR channel 1)
			band_name: Name of the band (e.g., 'pmr', 'airband')
			audio_sample_rate: Output audio sample rate in Hz (e.g., 16000)
			buffer_size_seconds: Maximum buffer size in seconds (prevents unbounded memory growth)
			disk_flush_interval_seconds: How often to flush buffer to disk (trade-off: latency vs overhead)
			audio_output_dir: Base output directory path
			modulation: Modulation type (e.g., 'NFM', 'AM', 'USB') - stored in BWF metadata
			filename_suffix: Optional suffix for filename (e.g., SNR and device info)
			soft_limit_drive: Tanh soft-limiter drive amount.  Higher values
				compress louder signals more aggressively.  Typical range
				1.0 - 3.0; the config default (1.25) is a gentle setting
				that leaves most voice content untouched.
			noise_reduction_enabled: When True (default), spectral-subtraction
				noise reduction is applied to each flushed block before it is
				written to disk.  Disable to record the raw demodulated audio.
			dynamics_curve_enabled: When True, an experimental dual-region
				expander (cut-below-threshold + boost-above-threshold) is
				applied after noise reduction and before the soft limiter.
				Off by default; driven from RecordingConfig in config.yaml.
			dynamics_curve_config: DynamicsCurveConfig instance (or any
				object with the expected fields) supplying the expander's
				tuning parameters.  Only read when dynamics_curve_enabled
				is True.
		"""

		self.channel_freq = channel_freq
		self.channel_index = channel_index
		self.band_name = band_name
		self.audio_sample_rate = audio_sample_rate
		self.disk_flush_interval = disk_flush_interval_seconds
		self.modulation = modulation
		# Precompute soft limiter parameters for efficiency
		self.soft_limit_drive = max(0.1, float(soft_limit_drive))
		self.soft_limit_scale = 1.0 / math.tanh(self.soft_limit_drive)
		self.noise_reduction_enabled = noise_reduction_enabled
		self.trim_carrier_transients = trim_carrier_transients
		self.fade_in_ms = fade_in_ms
		self.fade_out_ms = fade_out_ms
		self._first_flush_done = False
		self.dynamics_curve_enabled = dynamics_curve_enabled
		self.dynamics_curve_config = dynamics_curve_config
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
		bwf_originator = "Substation"
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

		self.bext_metadata: _BextMetadata = {
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
		self._closing = threading.Event()
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

		if self._closing.is_set():
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

			while not self._closing.is_set():

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

		# NOTE: This intentionally uses a threading.Lock (not asyncio.Lock) because
		# append_audio() is called from the executor thread.  The critical section
		# is a fast memcpy so event-loop blocking is negligible.
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

	def _write_samples_to_wav (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Write audio samples to WAV file (runs in executor thread)

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		# Carrier transient trimming: remove the sharp key-ON click on the
		# first flush and the key-OFF click on the final flush.  Runs
		# before noise reduction so the transient doesn't contaminate the
		# noise estimate.
		if self.trim_carrier_transients and samples.size > 0:
			if not self._first_flush_done:
				samples = _trim_carrier_transient_start(samples, self.audio_sample_rate)
			if self._closing.is_set():
				samples = _trim_carrier_transient_end(samples, self.audio_sample_rate)

		# Fades: apply fade-in on first flush, fade-out on final flush.
		# Runs after transient trimming so the fade is always on the final
		# audio boundary (not on samples that get trimmed away).
		if samples.size > 0:
			if not self._first_flush_done and self.fade_in_ms:
				samples = substation.dsp.filters.apply_fade(samples, self.audio_sample_rate, self.fade_in_ms, None)
			if self._closing.is_set() and self.fade_out_ms:
				samples = substation.dsp.filters.apply_fade(samples, self.audio_sample_rate, None, self.fade_out_ms)

		self._first_flush_done = True

		# Apply noise reduction using faster spectral subtraction method
		# This is 5-10x faster than the noisereduce library.  We catch the
		# family of errors numpy/scipy can realistically raise on bad data
		# (shape mismatches, invalid parameters, filter stability issues)
		# so a single bad block becomes a warning rather than killing the
		# recorder thread, but we deliberately let truly unexpected errors
		# (AttributeError, ImportError, KeyboardInterrupt, etc.) propagate
		# so bugs introduced by refactoring surface immediately instead of
		# being buried in the log.
		if self.noise_reduction_enabled:
			try:
				samples, self.noise_mag = substation.dsp.noise_reduction.apply_spectral_subtraction(
					samples, self.audio_sample_rate, oversub=0.7, floor=0.06,
					noise_mag=self.noise_mag, adaptive_noise_estimation=self.initial_noise_floor_db is not None,
				)
			except (ValueError, TypeError, RuntimeError) as exc:
				logger.warning(f"Noise reduction failed for {self.filepath}: {exc}")

		# Optional dynamics-curve stage: per-sample dual-region expander.
		# Sits between spectral subtraction (which cleans the signal) and the
		# soft limiter (which is the final clip protection).  Disabled by
		# default; configured globally via RecordingConfig.dynamics_curve.
		if self.dynamics_curve_enabled and self.dynamics_curve_config is not None and samples.size > 0:
			try:
				cfg = self.dynamics_curve_config
				samples = substation.dsp.noise_reduction.apply_dynamics_curve(
					samples,
					threshold_dbfs=cfg.threshold_dbfs,
					cut_db=cfg.cut_db,
					boost_db=cfg.boost_db,
					floor_dbfs=cfg.floor_dbfs,
					cut_curve=cfg.cut_curve,
					boost_curve=cfg.boost_curve,
				)
			except (ValueError, TypeError, RuntimeError) as exc:
				logger.warning(f"Dynamics curve failed for {self.filepath}: {exc}")

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

		self._closing.set()

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

	@staticmethod
	def check_empty (filepath: str, flatness_threshold: float | None = None) -> bool:

		"""
		Return True if the finished recording is "empty" (noise-only).

		Uses spectral flatness (Wiener entropy): noise has a flat power
		spectrum (flatness ~0.3-0.5), while any real signal — voice, data,
		tones — has a peaked spectrum (flatness < 0.04 typically).  A
		threshold of 0.15 sits in the large gap between the two, giving
		robust separation without tuning per modulation type.
		"""

		try:
			data, sr = soundfile.read(filepath, dtype='float32')
		except Exception:
			return False

		if len(data) < 512:
			return True

		if flatness_threshold is None:
			import substation.constants
			flatness_threshold = substation.constants.SPECTRAL_FLATNESS_THRESHOLD

		import scipy.signal as _sig
		freqs, psd = _sig.welch(data, sr, nperseg=min(2048, len(data)))
		psd = psd + 1e-12
		log_mean = numpy.mean(numpy.log(psd))
		flatness = float(numpy.exp(log_mean - numpy.log(numpy.mean(psd))))
		return flatness > flatness_threshold

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
