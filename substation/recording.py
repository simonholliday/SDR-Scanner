"""
Channel recording management with WAV and FLAC output support.

Handles buffered audio recording to WAV or FLAC files. WAV files include
industry-standard Broadcast WAV (BWF/BEXT) metadata with sample-accurate
timestamps for timeline placement in audio editors. FLAC files are lossless
compressed (~39% smaller) with Vorbis comment metadata (date and frequency
as text tags, but no timeline positioning support).

The recorder uses a memory buffer to avoid blocking the main processing
thread, and flushes to disk periodically in the background.
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

import mutagen.flac
import numpy
import numpy.typing
import soundfile

import substation.constants
import substation.dsp.filters
import substation.dsp.noise_reduction


logger = logging.getLogger(__name__)


def format_freq (hz: float) -> str:

	"""Format a frequency in Hz as a compact, filename-safe string.

	Automatically selects GHz, MHz, or kHz to keep the string short
	while preserving full precision (no rounding).  Trailing zeros
	are stripped for readability.

	Examples: 1420405000 → '1.420405GHz', 446006250 → '446.00625MHz',
	          125850000 → '125.85MHz', 14200 → '14.2kHz'.
	"""

	if hz >= 1e9:
		s = f"{hz / 1e9:.6f}".rstrip('0').rstrip('.')
		return f"{s}GHz"

	if hz >= 1e6:
		s = f"{hz / 1e6:.6f}".rstrip('0').rstrip('.')
		return f"{s}MHz"

	s = f"{hz / 1e3:.3f}".rstrip('0').rstrip('.')
	return f"{s}kHz"


def _trim_carrier_transient_start (audio: numpy.typing.NDArray[numpy.float32], sample_rate: int) -> numpy.typing.NDArray[numpy.float32]:

	"""Remove a carrier key-ON transient from the start of the audio.

	Carrier transient shape (what we're looking for):
	  - Very fast attack: the transmitter key-on produces a near-
	    instantaneous amplitude spike (sub-millisecond rise time).
	  - Brief duration: the entire transient (attack + decay) is
	    typically 1-10ms, always under 15ms.
	  - Exponential decay: amplitude drops back toward the noise floor
	    over a few milliseconds after the initial spike.
	  - Bordered by silence: the spike is preceded by noise-floor-level
	    audio (or nothing, if at sample zero) and followed by a quiet
	    gap before voice begins.

	This shape distinguishes carrier transients from voice plosives
	(e.g. 'p', 't'), which have a slower attack, are followed immediately
	by voiced content, and are preceded by other speech — not silence.

	Detection strategy:
	  1. Scan the first 500ms (wide window because the scanner may
	     detect the RF carrier hundreds of ms before the key-ON click).
	  2. Find the *earliest* spike that exceeds the detection threshold
	     (not the loudest — voice later in the window is often louder).
	  3. Verify the spike meets all three criteria:
	     - Pre-silence: region before spike is at noise floor (ratio > 8x)
	     - Duration: spike decays within 15ms
	     - Post-silence: 20ms after spike is quiet (< 25% of spike peak)
	"""

	# --- Scan window: first 500ms ---
	# Wide because the scanner may detect the RF carrier 100-300ms before
	# the key-ON transient arrives (carrier ramps up before full keying).
	scan_len = min(len(audio), int(sample_rate * 0.5))
	if scan_len < 10:
		return audio

	# --- Smoothed envelope ---
	# 0.5ms moving-average window smooths sample-level noise while
	# preserving the fast attack of a carrier transient (sub-ms rise).
	env = numpy.abs(audio[:scan_len])
	win = max(1, int(sample_rate * 0.0005))  # 0.5ms = 8 samples at 16kHz
	env_s = numpy.convolve(env, numpy.ones(win) / win, mode='same')

	# Minimum pre-silence region: 3ms.  If the spike is closer to
	# sample zero than this, we use the body-RMS fallback instead.
	min_pre = int(sample_rate * 0.003)  # 48 samples at 16kHz
	ratio_threshold = substation.constants.CARRIER_TRANSIENT_RATIO  # 8.0

	# --- Noise estimate ---
	# 25th percentile of the scan region.  The median would be dominated
	# by voice content in recordings where speech occupies >50% of the
	# scan window; the 25th percentile better represents the quiet
	# carrier/noise floor that precedes the transient.
	noise_est = float(numpy.percentile(env[:scan_len], 25))
	if noise_est < 1e-6:
		noise_est = 1e-6
	detect_threshold = noise_est * ratio_threshold

	# --- Find earliest candidate ---
	# Walk forward to find the first sample exceeding the detection
	# threshold.  By searching chronologically rather than by amplitude
	# (argmax), we find a small carrier transient even when louder voice
	# content follows later in the scan window.
	above = numpy.where(env_s > detect_threshold)[0]
	if len(above) == 0:
		return audio

	# Refine: find the local peak within 15ms of the first crossing.
	# The crossing may be on the rising edge; the true peak is nearby.
	first_above = int(above[0])
	peak_search_end = min(scan_len, first_above + int(sample_rate * 0.015))
	spike_idx = first_above + int(numpy.argmax(env_s[first_above:peak_search_end]))
	spike_peak = float(env_s[spike_idx])

	# --- Check 1: Pre-silence (fast attack from quiet) ---
	# The region before the spike must be at noise-floor level.  This
	# rejects voice plosives, which are preceded by other speech content.
	pre_end = max(0, spike_idx - win)

	if pre_end < min_pre:
		# Spike is within the first 3ms — no pre-silence available.
		# This happens when the recording starts at the exact moment of
		# key-on.  Fall back to comparing against the signal body
		# (20-50ms in) which should be at noise floor if this is a
		# genuine transient followed by a quiet gap before voice.
		body_start = int(sample_rate * 0.02)
		body_end = min(len(audio), int(sample_rate * 0.05))
		if body_end <= body_start:
			return audio
		body_rms = float(numpy.sqrt(numpy.mean(audio[body_start:body_end] ** 2)))
		if body_rms < 1e-6:
			return audio
		if spike_peak / body_rms < ratio_threshold:
			return audio
		pre_rms = body_rms
	else:
		# Normal case: measure silence before the spike.
		pre_rms = float(numpy.sqrt(numpy.mean(audio[:pre_end] ** 2)))
		if pre_rms < 1e-6:
			pre_rms = 1e-6
		if spike_peak / pre_rms < ratio_threshold:
			return audio

	# --- Check 2: Duration (brief spike, fast decay) ---
	# Walk forward from the peak to find where the envelope drops back
	# below the decay threshold and stays there for 2ms.  A carrier
	# transient decays exponentially within a few ms; voice energy
	# persists much longer.  Reject if the spike lasts more than 15ms.
	decay_threshold = pre_rms * ratio_threshold
	decay_win = max(1, int(sample_rate * 0.002))  # 2ms settling window
	spike_end = spike_idx
	for j in range(spike_idx, len(env_s)):
		if env_s[j] >= decay_threshold:
			spike_end = j
		elif j - spike_end > decay_win:
			break

	spike_duration_ms = (spike_end - spike_idx) / sample_rate * 1000
	if spike_duration_ms > 15:
		return audio

	# --- Check 3: Post-silence (decay back to quiet) ---
	# The 20ms region after the spike must be quiet relative to the
	# spike peak (< 25%).  A carrier transient decays to silence before
	# voice begins; a voice plosive ('p', 't', 'k') is followed
	# immediately by vowel energy at a comparable level.
	post_start = min(len(audio), spike_end + decay_win)
	post_end = min(len(audio), post_start + int(sample_rate * 0.02))  # 20ms
	if post_end > post_start:
		post_rms = float(numpy.sqrt(numpy.mean(audio[post_start:post_end] ** 2)))
		if post_rms > spike_peak * 0.25:
			return audio

	trim_point = min(spike_end + decay_win, len(audio))
	return audio[trim_point:]


def _trim_carrier_transient_end (audio: numpy.typing.NDArray[numpy.float32], sample_rate: int) -> numpy.typing.NDArray[numpy.float32]:

	"""Remove a carrier key-OFF transient from the end of the audio.

	Mirror of _trim_carrier_transient_start, applied to the recording
	tail.  The key-OFF transient has the same shape (fast attack, brief
	duration, exponential decay) but is followed by silence rather than
	preceded by it.

	Scans the last 500ms backwards from the end, finding the *last*
	spike that exceeds the detection threshold and passes all checks.
	This mirrors the start-trim's "earliest first" strategy — voice
	content earlier in the window won't mask a later transient.

	Checks:
	  1. Post-silence: region after spike is at noise floor (ratio > 8x)
	  2. Duration: spike is under 15ms
	  3. Pre-silence: 20ms before spike is quiet (< 25% of spike peak)
	"""

	# --- Scan window: last 4 seconds ---
	# Wide because the audio silence timeout (default 3s) keeps the
	# recording running long after the key-OFF transient.  The transient
	# can easily be 500ms+ from the end of the file.
	scan_len = min(len(audio), int(sample_rate * 4.0))
	if scan_len < 10:
		return audio

	tail = audio[-scan_len:]

	# --- Smoothed envelope (0.5ms window, same as start-trim) ---
	env = numpy.abs(tail)
	win = max(1, int(sample_rate * 0.0005))  # 0.5ms = 8 samples at 16kHz
	env_s = numpy.convolve(env, numpy.ones(win) / win, mode='same')

	ratio_threshold = substation.constants.CARRIER_TRANSIENT_RATIO

	# --- Noise estimate ---
	# Use the 10th percentile of the scan region.  For the end-trim,
	# the scan window contains voice + transient + noise floor; the
	# 10th percentile captures the quiet noise floor regardless of
	# how much voice content is in the window.
	noise_est = float(numpy.percentile(env, 10))
	if noise_est < 1e-6:
		noise_est = 1e-6
	detect_threshold = noise_est * ratio_threshold

	# --- Find latest candidate ---
	# Walk backward from the end to find the last sample exceeding the
	# detection threshold.  By searching from the end, we find the
	# key-OFF transient even when louder voice content precedes it.
	above = numpy.where(env_s > detect_threshold)[0]
	if len(above) == 0:
		return audio

	# The last threshold crossing is our candidate.  Find the local
	# peak within 15ms before it (the crossing may be on the falling
	# edge; the true peak is nearby).
	last_above = int(above[-1])
	peak_search_start = max(0, last_above - int(sample_rate * 0.015))
	spike_idx = peak_search_start + int(numpy.argmax(env_s[peak_search_start:last_above + 1]))
	spike_peak = float(env_s[spike_idx])

	# --- Check 1: Post-silence ---
	# The region after the spike must be at noise floor.
	post_start = min(scan_len, spike_idx + win)
	min_post = int(sample_rate * 0.003)  # 3ms minimum post-silence
	if post_start > scan_len - min_post:
		return audio

	post_rms = float(numpy.sqrt(numpy.mean(tail[post_start:] ** 2)))
	if post_rms < 1e-6:
		post_rms = 1e-6

	if spike_peak / post_rms < ratio_threshold:
		return audio

	# --- Check 2: Duration (brief spike) ---
	# Walk backward from the peak to find the onset.
	threshold = post_rms * ratio_threshold
	decay_win = max(1, int(sample_rate * 0.002))  # 2ms settling window
	spike_start = spike_idx
	for j in range(spike_idx, -1, -1):
		if env_s[j] >= threshold:
			spike_start = j
		elif spike_start - j > decay_win:
			break

	spike_duration_ms = (spike_idx - spike_start) / sample_rate * 1000
	if spike_duration_ms > 15:
		return audio

	# --- Check 3: Pre-silence (quiet before the spike) ---
	# The 20ms region before the spike must be quiet relative to the
	# spike peak (< 25%).
	pre_check_end = max(0, spike_start - decay_win)
	pre_check_start = max(0, pre_check_end - int(sample_rate * 0.02))  # 20ms
	if pre_check_end > pre_check_start:
		pre_rms = float(numpy.sqrt(numpy.mean(tail[pre_check_start:pre_check_end] ** 2)))
		if pre_rms > spike_peak * 0.25:
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

	Metadata: WAV files get a Broadcast WAV (BEXT) chunk with sample-accurate
	timestamps for timeline placement in audio editors. FLAC files get Vorbis
	comments with the same fields as text (no timeline positioning support).
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
		start_time: datetime.datetime | None = None,
		audio_format: str = 'wav',
	) -> None:

		"""
		Initialize a channel recorder and open the output audio file.

		Creates the output directory structure, opens a WAV or FLAC file for
		writing, prepares metadata, and sets up the memory buffer for
		accumulating audio samples.

		Files are organized as: output_dir/YYYY-MM-DD/band_name/filename.{wav,flac}
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
		self.audio_format = audio_format
		# Precompute soft limiter parameters for efficiency.
		# Ceiling is 0.98 (-0.18 dB) rather than 1.0 to prevent inter-sample
		# (true-peak) overshoot: tanh limits individual samples, but the
		# reconstructed waveform between samples can exceed the sample peaks
		# by ~0.2 dB.  The 2% headroom absorbs this.
		self.soft_limit_drive = max(0.1, float(soft_limit_drive))
		self.soft_limit_ceiling = 0.98
		self.soft_limit_scale = self.soft_limit_ceiling / math.tanh(self.soft_limit_drive)
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
		self.start_time = start_time if start_time is not None else datetime.datetime.now()

		# Calculate TimeReference: sample count since midnight
		# This is part of the Broadcast WAV spec and allows precise synchronization
		# between multiple recordings (they all share the same midnight reference)
		midnight = self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
		seconds_since_midnight = (self.start_time - midnight).total_seconds()
		self.time_reference = int(seconds_since_midnight * audio_sample_rate)

		# Build filename with timestamp, band, channel, and frequency
		# Format: YYYY-MM-DD_HH-MM-SS_band_channel_freq_[suffix].{wav,flac}
		# Example: 2026-01-25_14-30-45_pmr_0_446.00625MHz_12.5dB_rtlsdr_0.wav
		date_str = self.start_time.strftime("%Y-%m-%d")
		time_str = self.start_time.strftime("%H-%M-%S")
		freq_str = format_freq(channel_freq)

		filename = f"{date_str}_{time_str}_{band_name}_{channel_index}_{freq_str}"

		if filename_suffix:
			filename += "_" + filename_suffix

		ext = '.flac' if audio_format == 'flac' else '.wav'
		filename += ext

		# Organize files: base_dir/YYYY-MM-DD/band_name/filename.{wav,flac}
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

		# Open audio file for writing using soundfile library.
		# PCM_16 = 16-bit signed integer audio, used for both WAV and FLAC.
		# FLAC compression level 8 gives best compression (~39% saving)
		# with no measurable speed penalty on short radio recordings.

		sf_format = 'FLAC' if audio_format == 'flac' else 'WAV'
		self.audio_file = soundfile.SoundFile(
			self.filepath,
			mode='w',
			samplerate=audio_sample_rate,
			channels=1,  # Mono (single channel)
			subtype='PCM_16',  # 16-bit PCM encoding
			format=sf_format,
		)

		# Prepare recording metadata (used for BEXT in WAV, Vorbis comments in FLAC).
		# Stored as a dict and written after the file is closed.

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

		# Store metadata to write when file is closed.
		# For WAV, the BEXT chunk is appended after all audio data is written.
		# For FLAC, Vorbis comments are written via mutagen after close.

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
			self.audio_file.write(samples)
			self.total_samples_written += len(samples)

	async def close (self) -> None:

		"""
		Close the recorder, flush remaining buffer, and finalize the audio file.

		Ensures any background flush task is stopped, writes remaining audio,
		closes the file handle, and writes metadata (BEXT for WAV, Vorbis
		comments for FLAC).
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

		# Close audio file (this writes headers)
		self.audio_file.close()

		# Write metadata after the file is closed
		if self.bext_metadata:
			if self.audio_format == 'flac':
				self._write_flac_metadata()
			else:
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
		origination_date = self.bext_metadata['origination_date'].encode('ascii', errors='replace')[:10].ljust(10, b'\x00')
		origination_time = self.bext_metadata['origination_time'].encode('ascii', errors='replace')[:8].ljust(8, b'\x00')
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

	def _write_flac_metadata (self) -> None:

		"""
		Write Vorbis comments to a FLAC file after it has been closed.

		Stores the same metadata fields as the BEXT chunk (band, frequency,
		date, time, modulation) as Vorbis comment tags.  Note: FLAC cannot
		carry the sample-accurate time_reference that enables broadcast
		timeline placement in audio editors — date and time are stored as
		text strings only.
		"""

		if not self.bext_metadata:
			return

		try:
			flac = mutagen.flac.FLAC(self.filepath)

			flac['COMMENT'] = self.bext_metadata['description']
			flac['ENCODED_BY'] = self.bext_metadata['originator']
			flac['ORIGINATOR_REFERENCE'] = self.bext_metadata['originator_reference']
			flac['DATE'] = self.bext_metadata['origination_date']
			flac['CREATION_TIME'] = self.bext_metadata['origination_time']
			flac['TIME_REFERENCE'] = str(self.bext_metadata['time_reference'])
			flac['CODING_HISTORY'] = self.bext_metadata['coding_history'].rstrip('\r\n')

			flac.save()
			logger.debug(f"Wrote Vorbis comments to {self.filepath}")

		except Exception as e:
			logger.error(f"Failed to write FLAC metadata to {self.filepath}: {e}")
