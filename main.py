import asyncio
import collections
import datetime
import logging
import numpy
import numpy.typing
import os
import rtlsdr
import scipy.signal
import threading
import time
import typing
import wave
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Audio conversion constants
INT16_MAX = 32767
INT16_MIN = -32768

# NFM demodulation constants
NFM_DEEMPHASIS_TAU = 300e-6  # 300µs time constant (standard for NFM)
NFM_DEVIATION_HZ = 5000      # Max frequency deviation for normalization

# AM AGC constants
AM_AGC_ALPHA = 0.05          # AGC smoothing factor (lower = slower response)

class ChannelRecorder:

	"""
	Manages recording for a single channel with memory buffering and async disk writing
	"""

	def __init__ (
		self,
		channel_freq: float,
		channel_index: int,
		band_name: str,
		audio_sample_rate: int,
		buffer_size_seconds: float,
		disk_flush_interval_seconds: float,
		audio_output_dir: str
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
		"""

		self.channel_freq = channel_freq
		self.channel_index = channel_index
		self.band_name = band_name
		self.audio_sample_rate = audio_sample_rate
		self.disk_flush_interval = disk_flush_interval_seconds

		# Calculate maximum buffer size in samples
		max_buffer_samples = int(buffer_size_seconds * audio_sample_rate)

		# Create circular buffer (deque with maxlen drops oldest when full)
		self.audio_buffer: collections.deque = collections.deque(maxlen=max_buffer_samples)

		# Recording start time
		self.start_time = datetime.datetime.now()

		# Generate filename: {band}_{channel_id}_{datetime}.wav
		datetime_str = self.start_time.strftime("%Y%m%d_%H%M%S")
		filename = f"{datetime_str}_{band_name}_{channel_index}.wav"
		self.filepath = os.path.join(audio_output_dir, filename)

		# Create output directory if needed
		os.makedirs(audio_output_dir, exist_ok=True)

		# Open WAV file for writing (mono, 16kHz, 16-bit PCM)
		self.wav_file = wave.open(self.filepath, 'wb')
		self.wav_file.setnchannels(1)  # Mono
		self.wav_file.setsampwidth(2)  # 16-bit = 2 bytes
		self.wav_file.setframerate(audio_sample_rate)

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
			if self.audio_buffer.maxlen is not None:
				space_available = self.audio_buffer.maxlen - len(self.audio_buffer)
				if len(samples) > space_available:
					dropped = len(samples) - space_available
					logger.warning(f"Channel {self.channel_index}: Buffer overflow, dropping {dropped} oldest samples")
			self.audio_buffer.extend(samples)

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
			# Task cancelled - do final flush before exiting
			await self._flush_buffer_to_disk()
			raise

	async def _flush_buffer_to_disk (self) -> None:

		"""
		Flush accumulated buffer samples to disk in executor (non-blocking)
		"""

		with self._buffer_lock:
			if len(self.audio_buffer) == 0:
				return
			samples_to_write = numpy.array(self.audio_buffer, dtype=numpy.float32)
			self.audio_buffer.clear()

		await asyncio.get_running_loop().run_in_executor(None, self._write_samples_to_wav, samples_to_write)

	def _write_samples_to_wav (self, samples: numpy.typing.NDArray[numpy.float32]) -> None:

		"""
		Write audio samples to WAV file (runs in executor thread)

		Args:
			samples: Audio samples as float32 in range [-1.0, 1.0]
		"""

		# Convert float32 [-1, 1] to int16 [-32768, 32767]
		samples_int16 = numpy.clip(samples * INT16_MAX, INT16_MIN, INT16_MAX).astype(numpy.int16)

		with self._write_lock:

			# Write to WAV file
			self.wav_file.writeframes(samples_int16.tobytes())

			self.total_samples_written += len(samples)

	async def close(self) -> None:

		"""
		Close the recorder, flush remaining buffer, and finalize WAV file
		"""

		self.closing = True

		# Cancel flush task if running
		# Note: flush_task is a concurrent.futures.Future, not an asyncio.Task
		if self.flush_task and not self.flush_task.done():

			self.flush_task.cancel()

			try:
				self.flush_task.result(timeout=5)
			except Exception:
				pass

		# Final flush of any remaining samples
		await self._flush_buffer_to_disk()

		# Close WAV file
		self.wav_file.close()

		duration_seconds = self.total_samples_written / self.audio_sample_rate
		logger.info(f"Stopped recording channel {self.channel_index} (f = {self.channel_freq/1e6:.5f} MHz) - Duration: {duration_seconds:.1f}s, File: {self.filepath}")

def _decimate_audio (
	signal: numpy.typing.NDArray,
	sample_rate: float,
	audio_sample_rate: int,
	state: dict
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Decimate signal from sample_rate to audio_sample_rate with state preservation.

	Args:
		signal: Input signal to decimate
		sample_rate: Input sample rate in Hz
		audio_sample_rate: Target sample rate in Hz
		state: State dict for filter continuity

	Returns:
		Tuple of (decimated_samples, updated_state)
	"""

	decimation_factor = int(sample_rate / audio_sample_rate)

	if decimation_factor <= 1:
		return signal.astype(numpy.float32), state

	if 'decimate_sos' not in state:
		nyq_freq = audio_sample_rate / 2
		cutoff = nyq_freq * 0.8  # 80% of Nyquist
		state['decimate_sos'] = scipy.signal.butter(8, cutoff, fs=sample_rate, output='sos')

	if 'decimate_zi' not in state:
		state['decimate_zi'] = scipy.signal.sosfilt_zi(state['decimate_sos']) * 0.0

	if 'decimate_phase' not in state:
		state['decimate_phase'] = 0

	filtered, state['decimate_zi'] = scipy.signal.sosfilt(
		state['decimate_sos'], signal, zi=state['decimate_zi']
	)

	# Downsample with phase continuity
	start_idx = state['decimate_phase']
	audio_samples = filtered[start_idx::decimation_factor].astype(numpy.float32)

	remaining = (len(filtered) - start_idx) % decimation_factor
	state['decimate_phase'] = (decimation_factor - remaining) % decimation_factor

	return audio_samples, state

def demodulate_nfm (
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Demodulate Narrow FM (NFM) from IQ samples with state preservation

	Args:
		iq_samples: Complex IQ samples (already filtered to channel bandwidth)
		sample_rate: Sample rate of IQ samples in Hz
		audio_sample_rate: Desired output audio sample rate in Hz
		state: Optional state dict with 'last_iq' and 'deemph_zi' for continuous demodulation

	Returns:
		Tuple of (audio_samples, new_state) where new_state contains updated filter state
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	# Initialize state if needed
	if state is None:
		state = {}

	# FM demodulation: instantaneous frequency = d(phase)/dt
	if 'last_iq' not in state:
		state['last_iq'] = iq_samples[0]

	iq_with_prev = numpy.concatenate(([state['last_iq']], iq_samples))
	demod = numpy.angle(iq_with_prev[1:] * numpy.conj(iq_with_prev[:-1]))
	state['last_iq'] = iq_samples[-1]

	# De-emphasis filter
	tau = NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + sample_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	# DC removal - subtract block mean
	demod_dc_blocked = demod_deemph - numpy.mean(demod_deemph)

	# Normalize to approximate [-1, 1] range
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * NFM_DEVIATION_HZ / sample_rate)

	# Clip to [-1, 1] range
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# Decimate to audio sample rate
	return _decimate_audio(demod_normalized, sample_rate, audio_sample_rate, state)

def demodulate_am (
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Demodulate Amplitude Modulation (AM) from IQ samples with state preservation

	Args:
		iq_samples: Complex IQ samples (already filtered to channel bandwidth)
		sample_rate: Sample rate of IQ samples in Hz
		audio_sample_rate: Desired output audio sample rate in Hz
		state: Optional state dict for AGC and decimation continuity

	Returns:
		Tuple of (audio_samples, new_state) where new_state contains updated filter state
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# AM demodulation - extract magnitude (envelope detection)
	demod = numpy.abs(iq_samples)

	# DC removal - subtract block mean
	demod_dc_blocked = demod - numpy.mean(demod)

	# Smoothed AGC using leaky integrator
	block_level = numpy.percentile(numpy.abs(demod_dc_blocked), 99)

	if 'agc_level' not in state:
		state['agc_level'] = block_level if block_level > 0.01 else 1.0

	state['agc_level'] = AM_AGC_ALPHA * block_level + (1 - AM_AGC_ALPHA) * state['agc_level']

	if state['agc_level'] > 0.01:
		demod_normalized = demod_dc_blocked / state['agc_level']
	else:
		demod_normalized = demod_dc_blocked

	# Clip to [-1.0, 1.0] range
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# Decimate to audio sample rate
	return _decimate_audio(demod_normalized, sample_rate, audio_sample_rate, state)


# Dictionary of available demodulators
DEMODULATORS = {
	'NFM': demodulate_nfm,
	'AM': demodulate_am,
	# Future demodulators can be added here:
	# 'WFM': demodulate_wfm,
}

class RadioScanner:

	"""
	Self-contained radio scanner for RTL-SDR Blog V4
	Scans bands asynchronously and detects active channels based on SNR
	"""

	# Hysteresis margin in dB - channel turns ON at threshold, OFF at threshold minus this
	HYSTERESIS_DB = 3.0

	# Number of DC bins to exclude around center frequency (RTL-SDR DC spike)
	DC_SPIKE_BINS = 3

	# Number of FFT segments for Welch averaging
	WELCH_SEGMENTS = 8

	def __init__ (self, config_path: str = 'config.yaml', band_name: str = 'pmr') -> None:

		"""
		Initialize the scanner with configuration

		Args:
			config_path: Path to the YAML configuration file
			band_name: Name of the band to scan (default: 'pmr')
		"""
		self.config = self._load_config(config_path)
		self.band_name = band_name
		self.band_config = self.config['bands'][band_name]
		self.scanner_config = self.config['scanner']
		self.recording_config = self.config.get('recording', {})

		# Extract band parameters
		self.freq_start = self.band_config['freq_start']
		self.freq_end = self.band_config['freq_end']

		self.channel_spacing = self.band_config['channel_spacing']
		self.channel_width = self.band_config.get('channel_width', self.channel_spacing * 0.84) # Allow some gaps for inter-channel noise measurement

		self.sample_rate = self.band_config['sample_rate']
		self.snr_threshold_db = self.band_config.get('snr_threshold_db', 12)
		self.snr_threshold_off_db = self.snr_threshold_db - self.HYSTERESIS_DB

		# Validate SNR threshold
		if self.snr_threshold_db <= self.HYSTERESIS_DB:
			logger.error(f"CONFIG ERROR: Band '{band_name}' has snr_threshold_db ({self.snr_threshold_db} dB) <= HYSTERESIS_DB ({self.HYSTERESIS_DB} dB)")
			logger.error(f"This would result in snr_threshold_off_db = {self.snr_threshold_off_db} dB")
			logger.error(f"Channels would never turn OFF because SNR rarely drops to 0 or below")
			logger.error(f"Please set snr_threshold_db to at least {self.HYSTERESIS_DB + 0.1} dB")
			raise ValueError(f"Invalid snr_threshold_db for band '{band_name}': must be > {self.HYSTERESIS_DB} dB")

		self.sdr_gain_db = self.band_config.get('sdr_gain_db', 'auto')

		# Recording parameters
		self.modulation = self.band_config.get('modulation', None)
		self.recording_enabled = self.recording_config.get('enabled', False)
		self.audio_sample_rate = self.recording_config.get('audio_sample_rate', 16000)
		self.buffer_size_seconds = self.recording_config.get('buffer_size_seconds', 30)
		self.disk_flush_interval = self.recording_config.get('disk_flush_interval_seconds', 5)
		self.audio_output_dir = self.recording_config.get('audio_output_dir', './audio')

		# Check if recording is possible (enabled and demodulator available)
		self.can_record = self.recording_enabled and self.modulation in DEMODULATORS

		# Scanner parameters
		self.sdr_device_sample_size = self.scanner_config['sdr_device_sample_size']
		self.band_time_slice_ms = self.scanner_config['band_time_slice_ms']

		# Calculate channels in the band
		self.channels = self._calculate_channels()
		self.num_channels = len(self.channels)

		# Calculate edge margin - half channel spacing on each side to avoid filter rolloff
		self.band_edge_margin_hz = self.channel_spacing / 2

		# Calculate center frequency and bandwidth for SDR (with edge margin)
		self.center_freq = (self.freq_start + self.freq_end) / 2
		self.required_bandwidth = self.freq_end - self.freq_start + self.channel_width + (2 * self.band_edge_margin_hz)

		# Channel state tracking: True = on, False = off
		self.channel_states: dict[float, bool] = {ch_freq: False for ch_freq in self.channels}

		# Channel recorders: one per active channel
		self.channel_recorders: dict[float, ChannelRecorder] = {}

		# SDR device
		self.sdr: rtlsdr.RtlSdr | None = None

		# Pre-computed values (initialized in _precompute_fft_params)
		self.samples_per_slice: int = 0
		self.fft_size: int = 0
		self.window: numpy.typing.NDArray[numpy.float64] | None = None
		self.freqs: numpy.typing.NDArray[numpy.float64] | None = None
		self.channel_indices: dict[float, tuple[int, int]] = {}
		self.noise_indices: list[tuple[int, int]] = []
		self.dc_mask: numpy.typing.NDArray[numpy.bool_] | None = None
		self.channel_filter_sos: numpy.typing.NDArray[numpy.float64] | None = None

		# Per-channel filter state for continuous processing (prevents clicks at block boundaries)
		self.channel_filter_zi: dict[float, numpy.typing.NDArray[numpy.complex128]] = {}

		# Per-channel demodulator state (last IQ sample and de-emphasis filter state)
		self.channel_demod_state: dict[float, dict] = {}

		# Cumulative sample counter for continuous phase in frequency shifting
		self.sample_counter: int = 0

		# Sample queue for async streaming
		self.sample_queue: asyncio.Queue | None = None
		self.loop: asyncio.AbstractEventLoop | None = None

		logger.info(f"Initialized scanner for band '{band_name}'")
		logger.info(f"Frequency range: {self.freq_start/1e6:.5f} - {self.freq_end/1e6:.5f} MHz")
		logger.info(f"Number of channels: {self.num_channels}")
		logger.info(f"Center frequency: {self.center_freq/1e6:.5f} MHz")
		logger.info(f"Required bandwidth: {self.required_bandwidth/1e6:.5f} MHz (inc. {self.band_edge_margin_hz/1e3:.1f}kHz edge margin)")
		logger.info(f"Sample rate: {self.sample_rate/1e6:.3f} MHz")
		logger.info(f"SNR threshold: {self.snr_threshold_db} dB ON / {self.snr_threshold_off_db} dB OFF (hysteresis)")
		logger.info(f"SDR Gain: {self.sdr_gain_db}")
		logger.info(f"Modulation: {self.modulation}")

		if self.can_record:
			status = f"ENABLED ({self.audio_sample_rate} Hz mono WAV to {self.audio_output_dir})"
		elif self.recording_enabled:
			status = f"DISABLED (no demodulator for {self.modulation})"
		else:
			status = "DISABLED"
		logger.info(f"Recording: {status}")

	def _load_config(self, config_path: str) -> dict:

		"""Load configuration from YAML file"""

		with open(config_path, 'r') as f:
			return yaml.safe_load(f)

	def _calculate_channels(self) -> list[float]:

		"""
		Calculate all channel frequencies in the band

		Returns:
			List of channel center frequencies in Hz
		"""

		channels = []
		freq = self.freq_start
		while freq <= self.freq_end:
			channels.append(freq)
			freq += self.channel_spacing
		return channels

	def _precompute_fft_params (self) -> None:

		"""
		Pre-compute FFT window, frequency bins, and channel index ranges.
		Called once after sample size is determined.
		"""

		# Calculate samples per time slice
		time_slice_seconds = self.band_time_slice_ms / 1000.0
		self.samples_per_slice = int(self.sample_rate * time_slice_seconds)

		# Adjust to be multiple of device sample size
		self.samples_per_slice = -(-self.samples_per_slice // self.sdr_device_sample_size) * self.sdr_device_sample_size

		# FFT size per segment for Welch averaging
		self.fft_size = self.samples_per_slice // self.WELCH_SEGMENTS

		# Window function
		self.window = scipy.signal.get_window('hann', self.fft_size).astype(numpy.float64)

		# Frequency bins (shifted)
		freqs_unshifted = numpy.fft.fftfreq(self.fft_size, d=1.0/self.sample_rate)
		self.freqs = self.center_freq + numpy.fft.fftshift(freqs_unshifted)

		# Calculate observable frequency range based on sample rate
		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		# Calculate actual band span required (with margins)
		band_span = self.required_bandwidth

		# Check if band is too wide for sample rate
		if band_span > observable_span:
			logger.error(f"CONFIG ERROR: Band '{self.band_name}' spans {band_span/1e6:.2f} MHz but sample rate is only {self.sample_rate/1e6:.2f} MHz")
			logger.error(f"Band frequency range: {self.freq_start/1e6:.3f} - {self.freq_end/1e6:.3f} MHz (inc. margins)")
			logger.error(f"Observable frequency range: {observable_min_freq/1e6:.3f} - {observable_max_freq/1e6:.3f} MHz")
			logger.error(f"")
			logger.error(f"To fix this, you can either:")
			logger.error(f"  1. Split this band into multiple smaller bands of ~{observable_span*0.8/1e6:.1f} MHz each in config.yaml")
			logger.error(f"  2. Increase the sample_rate for this band (if your SDR hardware supports it)")
			logger.error(f"  3. Use a different SDR with higher bandwidth capability")
			logger.error(f"")
			raise ValueError(f"Band '{self.band_name}' is too wide ({band_span/1e6:.2f} MHz) for sample rate ({self.sample_rate/1e6:.2f} MHz)")

		# Pre-compute channel index ranges
		freq_resolution = self.sample_rate / self.fft_size
		self.channel_indices = {}
		channels_outside_range = []

		for channel_freq in self.channels:
			channel_half_width = self.channel_width / 2
			low_freq = channel_freq - channel_half_width
			high_freq = channel_freq + channel_half_width

			# Find indices
			indices = numpy.where((self.freqs >= low_freq) & (self.freqs <= high_freq))[0]
			if len(indices) > 0:
				self.channel_indices[channel_freq] = (indices[0], indices[-1] + 1)
			else:
				# Channel is outside observable range
				channels_outside_range.append(channel_freq)
				self.channel_indices[channel_freq] = (0, 0)

		# Warn about channels outside range (shouldn't happen if band span check passed)
		if channels_outside_range:
			logger.warning(f"CONFIG WARNING: {len(channels_outside_range)} channels fall outside observable frequency range:")
			for ch_freq in channels_outside_range:
				logger.warning(f"  - Channel {ch_freq/1e6:.5f} MHz is outside {observable_min_freq/1e6:.3f} - {observable_max_freq/1e6:.3f} MHz")
			logger.warning(f"These channels will not be scanned. Check your band configuration in config.yaml")

		# Pre-compute noise estimation regions (gaps between channels)
		self._compute_noise_regions()

		# Pre-compute DC spike mask
		center_bin = self.fft_size // 2
		self.dc_mask = numpy.ones(self.fft_size, dtype=bool)
		dc_start = max(0, center_bin - self.DC_SPIKE_BINS)
		dc_end = min(self.fft_size, center_bin + self.DC_SPIKE_BINS + 1)
		self.dc_mask[dc_start:dc_end] = False

		# Pre-compute channel extraction filter (for recording)
		if self.can_record:
			cutoff_freq = self.channel_width / 2
			normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
			self.channel_filter_sos = scipy.signal.butter(5, normalized_cutoff, btype='low', output='sos')

		logger.info(f"FFT size: {self.fft_size} bins, frequency resolution: {freq_resolution:.1f} Hz")
		logger.info(f"Welch segments: {self.WELCH_SEGMENTS}, samples per slice: {self.samples_per_slice}")
		logger.info(f"DC spike exclusion: {self.DC_SPIKE_BINS * 2 + 1} bins around center")

	def _compute_noise_regions (self) -> None:

		"""
		Compute the index ranges for noise estimation.
		Uses the gaps between channels and areas outside the channel band.
		"""

		self.noise_indices = []

		# Sort channels by frequency
		sorted_channels = sorted(self.channels)

		# Calculate observable frequency range
		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		# Region before first channel (use edge of observable range, not band definition)
		first_channel = sorted_channels[0]
		first_channel_low = first_channel - self.channel_width / 2

		# Use the lower of: (observable minimum) or (band start - margin)
		noise_start_freq = max(observable_min_freq, self.freq_start - self.band_edge_margin_hz)

		band_start_idx = numpy.searchsorted(self.freqs, noise_start_freq)
		first_channel_idx = numpy.searchsorted(self.freqs, first_channel_low)

		gap_hz = (first_channel_low - noise_start_freq) / 1e3
		if first_channel_idx > band_start_idx:
			self.noise_indices.append((band_start_idx, first_channel_idx))
			logger.debug(f"Edge margin BEFORE first channel: {gap_hz:.1f} kHz ({first_channel_idx - band_start_idx} bins)")
		else:
			logger.debug(f"No gap before first channel (gap would be {gap_hz:.1f} kHz)")

		# Gaps between channels
		inter_channel_gaps = 0
		for i in range(len(sorted_channels) - 1):
			ch1 = sorted_channels[i]
			ch2 = sorted_channels[i + 1]

			ch1_high = ch1 + self.channel_width / 2
			ch2_low = ch2 - self.channel_width / 2

			# Only use gap if there's actually space between channels
			if ch2_low > ch1_high:
				idx_start = numpy.searchsorted(self.freqs, ch1_high)
				idx_end = numpy.searchsorted(self.freqs, ch2_low)
				if idx_end > idx_start:
					gap_hz = (ch2_low - ch1_high) / 1e3
					self.noise_indices.append((idx_start, idx_end))
					inter_channel_gaps += 1
					logger.debug(f"Inter-channel gap {inter_channel_gaps}: {gap_hz:.1f} kHz ({idx_end - idx_start} bins)")

		# Region after last channel (use edge of observable range, not band definition)
		last_channel = sorted_channels[-1]
		last_channel_high = last_channel + self.channel_width / 2

		# Use the higher of: (observable maximum) or (band end + margin)
		noise_end_freq = min(observable_max_freq, self.freq_end + self.band_edge_margin_hz)

		last_channel_idx = numpy.searchsorted(self.freqs, last_channel_high)
		band_end_idx = numpy.searchsorted(self.freqs, noise_end_freq)

		gap_hz = (noise_end_freq - last_channel_high) / 1e3
		if band_end_idx > last_channel_idx:
			self.noise_indices.append((last_channel_idx, band_end_idx))
			logger.debug(f"Edge margin AFTER last channel: {gap_hz:.1f} kHz ({band_end_idx - last_channel_idx} bins)")
		else:
			logger.debug(f"No gap after last channel (gap would be {gap_hz:.1f} kHz)")

		total_noise_bins = sum(end - start for start, end in self.noise_indices)
		logger.info(f"Noise estimation regions: {len(self.noise_indices)} gaps, {total_noise_bins} bins total")

	def _calibrate_sdr (self, known_freq: float, bandwidth: float = 300e3, iterations: int = 10) -> None:

		"""
		Calibrate SDR frequency offset using a known strong signal

		Args:
			known_freq: Known signal frequency in Hz (e.g., 93.7 MHz for WFM broadcast)
			bandwidth: Bandwidth to sample in Hz (default: 300 kHz)
			iterations: Number of measurements to average (default: 10)
		"""

		if self.sdr is None:
			raise RuntimeError("SDR device not initialized")

		logger.info(f"Calibrating SDR using known signal at {known_freq/1e6:.3f} MHz within {bandwidth/1e3:.0f} kHz bandwidth...")

		# Store current settings
		initial_center_freq = self.sdr.center_freq
		initial_sample_rate = self.sdr.sample_rate

		# Configure for calibration
		self.sdr.center_freq = known_freq
		self.sdr.sample_rate = bandwidth

		# Warm-up: discard first few reads to flush stale buffer data
		logger.info("Warming up SDR...")
		sample_size = 256 * 1024
		for _ in range(3):
			self.sdr.read_samples(sample_size)
			time.sleep(0.1)

		freq_correction_ppm_list = []
		peak_magnitudes = []
		magnitude_db = None  # Will be set in loop, used for noise floor calculation

		for iteration in range(iterations, 0, -1):
			logger.info(f"Calibration measurement {iterations - iteration + 1}/{iterations}...")

			# Read samples
			samples = self.sdr.read_samples(sample_size)

			# Apply window and compute FFT
			window = numpy.hanning(sample_size)
			fft_result = numpy.fft.fftshift(numpy.fft.fft(samples * window))
			freqs = numpy.fft.fftshift(numpy.fft.fftfreq(sample_size, 1 / self.sdr.sample_rate))
			magnitude_db = 20 * numpy.log10(numpy.abs(fft_result) + 1e-10)

			# Find peak frequency within expected range (±50 kHz)
			# This prevents locking onto wrong signals or noise spikes
			search_range_hz = 50e3
			freq_mask = numpy.abs(freqs) < search_range_hz
			if numpy.sum(freq_mask) == 0:
				logger.warning(f"No frequency bins within ±{search_range_hz/1e3:.0f} kHz search range")
				continue

			peak_index_local = numpy.argmax(magnitude_db[freq_mask])
			freqs_filtered = freqs[freq_mask]
			measured_freq = self.sdr.center_freq + freqs_filtered[peak_index_local]
			peak_mag = magnitude_db[freq_mask][peak_index_local]

			# Store peak magnitude for signal validation
			peak_magnitudes.append(peak_mag)

			# Calculate PPM error (divide by known_freq, not center_freq)
			freq_error_ppm = (measured_freq - known_freq) / known_freq * 1e6
			freq_correction_ppm_list.append(freq_error_ppm)

			# Settling delay between measurements
			time.sleep(0.2)

		# Validate signal strength (peak should be significantly above noise floor)
		# Use the last magnitude_db array from the loop
		avg_peak_mag = numpy.mean(peak_magnitudes)
		if magnitude_db is not None:
			noise_floor = numpy.percentile(magnitude_db, 25)
			signal_strength_db = avg_peak_mag - noise_floor
		else:
			signal_strength_db = 0  # Fallback if no iterations ran

		if signal_strength_db < 10:
			logger.warning(f"Calibration signal weak ({signal_strength_db:.1f} dB SNR)")
			logger.warning("Calibration may be inaccurate - ensure strong signal at calibration frequency")

		# Remove outliers using IQR method
		q1 = numpy.percentile(freq_correction_ppm_list, 25)
		q3 = numpy.percentile(freq_correction_ppm_list, 75)
		iqr = q3 - q1
		filtered_ppm = [x for x in freq_correction_ppm_list if q1 - 1.5*iqr <= x <= q3 + 1.5*iqr]

		# Use median of filtered values (more robust than mean)
		if len(filtered_ppm) == 0:
			logger.warning("All calibration measurements were outliers - using unfiltered data")
			filtered_ppm = freq_correction_ppm_list

		freq_correction_ppm = int(round(numpy.median(filtered_ppm)))
		ppm_std = numpy.std(filtered_ppm)

		# Log measurement statistics
		logger.info(f"Calibration measurements: {len(freq_correction_ppm_list)} total, {len(filtered_ppm)} after outlier removal")

		if ppm_std > 5:
			logger.warning(f"Calibration measurements inconsistent (std dev: {ppm_std:.2f} PPM)")
			logger.warning("Results may be unreliable")

		# Restore original settings
		self.sdr.center_freq = initial_center_freq
		self.sdr.sample_rate = initial_sample_rate

		# Apply correction if needed
		if freq_correction_ppm != 0:
			# Sanity check - typical RTL-SDR drift is within ±100 PPM
			if abs(freq_correction_ppm) > 200:
				logger.warning(f"Calibration calculated unusually large correction: {freq_correction_ppm} PPM")
				logger.warning("This may indicate incorrect calibration frequency or hardware issue")

			self.sdr.freq_correction = freq_correction_ppm
			logger.info(f"SDR calibrated with frequency correction: {freq_correction_ppm} PPM (signal: {signal_strength_db:.1f} dB SNR)")
		else:
			logger.info("SDR calibration complete - no correction needed")

	def _setup_sdr (self) -> None:

		"""Configure the SDR device with calculated parameters"""

		logger.info("Setting up SDR device...")
		self.sdr = rtlsdr.RtlSdr()
		self.sdr.sample_rate = self.sample_rate
		self.sdr.center_freq = self.center_freq
		self.sdr.gain = self.sdr_gain_db
		logger.info("SDR device configured successfully")

		# Calibrate frequency offset if calibration frequency is provided
		calibration_freq = self.scanner_config.get('calibration_frequency_hz', None)
		if calibration_freq is not None:
			self._calibrate_sdr(calibration_freq)

		# Now that we know sample rate is set, precompute FFT parameters
		self._precompute_fft_params()

	async def _cleanup_sdr (self) -> None:

		"""Clean up SDR resources and close any active recordings"""

		# Close all active recordings first
		for channel_freq in list(self.channel_recorders.keys()):
			await self._stop_channel_recording(channel_freq)

		# Close SDR device
		if self.sdr:
			try:
				self.sdr.close()
				logger.info("SDR device closed")
			except Exception as e:
				logger.warning(f"Error closing SDR device (this is normal on interrupt): {e}")

	def _sdr_callback(self, samples: numpy.typing.NDArray[numpy.complex64], _context: typing.Any) -> None:
		"""
		Callback for async SDR streaming (runs in librtlsdr background thread)

		Args:
			samples: IQ samples from SDR
			_context: Context object (unused)
		"""
		if self.loop and self.sample_queue:
			# Thread-safe: schedule queue put on the event loop
			# Make a copy to avoid buffer reuse issues
			self.loop.call_soon_threadsafe(self.sample_queue.put_nowait, samples.copy())

	async def _sample_band_async (self) -> typing.AsyncGenerator[numpy.typing.NDArray[numpy.complex64], None]:

		"""
		Asynchronously sample the band using background streaming

		Yields:
			numpy.ndarray: Complex IQ samples for the time slice
		"""

		logger.info(f"Time slice: {self.band_time_slice_ms} ms ({self.samples_per_slice} samples)")

		while True:
			# Get samples from queue (filled by background thread)
			samples = await self.sample_queue.get()
			yield samples

	def _calculate_psd_welch (self, samples: numpy.typing.NDArray[numpy.complex64]) -> numpy.typing.NDArray[numpy.float64]:

		"""
		Calculate Power Spectral Density using Welch's method (averaged segments).
		More stable than single FFT, reduces noise variance.

		Args:
			samples: Complex IQ samples

		Returns:
			psd_db: Averaged power spectral density (dB), already shifted
		"""

		n_samples = len(samples)
		segment_size = self.fft_size

		# Accumulator for averaged PSD
		psd_accumulator = numpy.zeros(segment_size, dtype=numpy.float64)

		# Process segments with 50% overlap for better averaging
		hop_size = segment_size // 2
		n_segments = (n_samples - segment_size) // hop_size + 1

		if n_segments <= 0:
			raise ValueError("Not enough samples for Welch PSD")

		for i in range(n_segments):

			start = i * hop_size
			end = start + segment_size
			segment = samples[start:end]

			# Apply pre-computed window
			windowed = segment * self.window

			# FFT and power
			fft_result = numpy.fft.fft(windowed)
			psd_accumulator += numpy.abs(fft_result) ** 2

		# Average and convert to dB
		psd_avg = psd_accumulator / n_segments
		psd_db = 10 * numpy.log10(psd_avg + 1e-12)

		# Shift to center frequency
		psd_db_shifted = numpy.fft.fftshift(psd_db)

		return psd_db_shifted

	def _get_channel_power (self, psd_db: numpy.typing.NDArray[numpy.float64], channel_freq: float) -> float:

		"""
		Extract power for a specific channel from PSD using pre-computed indices.

		Args:
			psd_db: Power spectral density (shifted)
			channel_freq: Center frequency of channel

		Returns:
			Mean power of the channel in dB
		"""

		idx_start, idx_end = self.channel_indices[channel_freq]

		if idx_end <= idx_start:
			return -numpy.inf

		# Get channel bins, excluding any that fall in DC spike region
		channel_bins = psd_db[idx_start:idx_end]

		# Check if this channel overlaps with DC spike
		center_bin = self.fft_size // 2
		if idx_start <= center_bin < idx_end:
			# This channel contains the DC spike - mask it out
			local_dc_start = max(0, center_bin - self.DC_SPIKE_BINS - idx_start)
			local_dc_end = min(idx_end - idx_start, center_bin + self.DC_SPIKE_BINS + 1 - idx_start)
			mask = numpy.ones(len(channel_bins), dtype=bool)
			mask[local_dc_start:local_dc_end] = False
			channel_bins = channel_bins[mask]

			if len(channel_bins) == 0:
				return -numpy.inf

		return numpy.mean(channel_bins)

	def _estimate_noise_floor (self, psd_db: numpy.typing.NDArray[numpy.float64]) -> float:

		"""
		Estimate noise floor from inter-channel gaps.
		More accurate than percentile of entire spectrum.

		Args:
			psd_db: Power spectral density (shifted)

		Returns:
			Estimated noise floor in dB
		"""

		if not self.noise_indices:
			# Fallback to percentile method if no gaps defined
			return numpy.percentile(psd_db, 25)

		# Collect all noise samples from gaps
		noise_samples = []
		for idx_start, idx_end in self.noise_indices:
			# Apply DC mask to noise regions too
			region = psd_db[idx_start:idx_end]
			dc_region_mask = self.dc_mask[idx_start:idx_end]
			noise_samples.extend(region[dc_region_mask])

		if len(noise_samples) == 0:
			return numpy.percentile(psd_db, 25)

		# Use median of noise samples - robust to outliers
		return numpy.median(noise_samples)

	def _extract_channel_iq (self, samples: numpy.typing.NDArray[numpy.complex64], channel_freq: float) -> numpy.typing.NDArray[numpy.complex64]:

		"""
		Extract IQ samples for a specific channel by frequency shifting and filtering

		Args:
			samples: Full bandwidth IQ samples from SDR
			channel_freq: Center frequency of channel to extract

		Returns:
			Filtered IQ samples centered at baseband for the channel
		"""

		# Frequency shift to baseband using continuous phase (prevents clicks at block boundaries)
		freq_offset = channel_freq - self.center_freq
		n_samples = len(samples)
		t = (self.sample_counter + numpy.arange(n_samples)) / self.sample_rate
		samples_shifted = samples * numpy.exp(-2j * numpy.pi * freq_offset * t)

		# Initialize filter state if needed (for continuous filtering across blocks)
		if channel_freq not in self.channel_filter_zi:
			# Initialize to zero state to avoid transients from arbitrary first sample
			zi = scipy.signal.sosfilt_zi(self.channel_filter_sos)
			self.channel_filter_zi[channel_freq] = (zi * 0.0).astype(numpy.complex128)

		# Low-pass filter with state preservation (prevents transients at block boundaries)
		filtered, self.channel_filter_zi[channel_freq] = scipy.signal.sosfilt(
			self.channel_filter_sos,
			samples_shifted,
			zi=self.channel_filter_zi[channel_freq]
		)

		return filtered

	def _start_channel_recording (self, channel_freq: float, channel_index: int, loop: asyncio.AbstractEventLoop) -> None:

		"""
		Start recording a channel

		Args:
			channel_freq: Channel center frequency in Hz
			channel_index: Channel index number
			loop: Event loop to use for creating async tasks
		"""

		# Create recorder instance
		recorder = ChannelRecorder(
			channel_freq=channel_freq,
			channel_index=channel_index,
			band_name=self.band_name,
			audio_sample_rate=self.audio_sample_rate,
			buffer_size_seconds=self.buffer_size_seconds,
			disk_flush_interval_seconds=self.disk_flush_interval,
			audio_output_dir=self.audio_output_dir
		)

		# Start the async flush task using the provided event loop
		recorder.flush_task = asyncio.run_coroutine_threadsafe(
			recorder._flush_to_disk_periodically(),
			loop
		)

		# Store recorder
		self.channel_recorders[channel_freq] = recorder

	async def _stop_channel_recording (self, channel_freq: float) -> None:

		"""
		Stop recording a channel and close the file

		Args:
			channel_freq: Channel center frequency in Hz
		"""

		if channel_freq not in self.channel_recorders:
			return

		recorder = self.channel_recorders[channel_freq]

		# Close recorder (flushes buffer and closes WAV file)
		await recorder.close()

		# Remove from dictionary
		del self.channel_recorders[channel_freq]

	def _process_samples (self, samples: numpy.typing.NDArray[numpy.complex64], loop: asyncio.AbstractEventLoop) -> None:

		"""
		Process samples to detect active channels

		Args:
			samples: Complex IQ samples from SDR
			loop: Event loop for creating async tasks
		"""

		# Check for ADC saturation/clipping
		# RTL-SDR outputs samples normalized to roughly [-1, 1]
		# Values consistently near 1.0 indicate clipping
		clipping_threshold = 0.95
		sample_magnitude = numpy.abs(samples)
		clipping_percentage = numpy.sum(sample_magnitude > clipping_threshold) / len(samples) * 100

		if clipping_percentage > 0.1:  # More than 0.1% of samples clipping
			logger.warning(f"ADC SATURATION DETECTED: {clipping_percentage:.1f}% of samples clipping")
			logger.warning("Consider reducing gain or using an RF attenuator")

		# Calculate PSD using Welch averaging
		psd_db = self._calculate_psd_welch(samples)

		# Estimate noise floor from inter-channel gaps
		noise_floor_db = self._estimate_noise_floor(psd_db)

		# Check each channel
		for channel_index, channel_freq in enumerate(self.channels):
			# Get channel power using pre-computed indices
			channel_power_db = self._get_channel_power(psd_db, channel_freq)

			# Calculate SNR
			snr_db = channel_power_db - noise_floor_db

			# Get current state
			current_state = self.channel_states[channel_freq]

			# Apply hysteresis: different thresholds for ON vs OFF
			if current_state:
				# Currently ON - use lower threshold to turn OFF
				is_active = snr_db > self.snr_threshold_off_db
			else:
				# Currently OFF - use higher threshold to turn ON
				is_active = snr_db > self.snr_threshold_db

			# Check for state change
			if is_active != current_state:
				# State changed - update and notify
				self.channel_states[channel_freq] = is_active
				state_str = "ON" if is_active else "OFF"
				channel_mhz = channel_freq / 1e6
				logger.info(f"Channel {channel_index} {state_str} (f = {channel_mhz:.5f} MHz, SNR = {snr_db:.1f}dB)")

				# Handle recording state changes
				if is_active:
					if self.can_record:
						self._start_channel_recording(channel_freq, channel_index, loop)
				else:
					# Channel turned OFF - clean up all state and stop recording
					if channel_freq in self.channel_filter_zi:
						del self.channel_filter_zi[channel_freq]
					if channel_freq in self.channel_demod_state:
						del self.channel_demod_state[channel_freq]
					if channel_freq in self.channel_recorders:
						asyncio.run_coroutine_threadsafe(self._stop_channel_recording(channel_freq), loop)

			# Feed audio samples to active recordings
			if is_active and channel_freq in self.channel_recorders:
				# Extract channel IQ samples
				channel_iq = self._extract_channel_iq(samples, channel_freq)

				# Demodulate to audio with state preservation
				demodulator = DEMODULATORS[self.modulation]
				demod_state = self.channel_demod_state.get(channel_freq, None)
				audio_samples, new_state = demodulator(
					channel_iq,
					self.sample_rate,
					self.audio_sample_rate,
					state=demod_state
				)
				self.channel_demod_state[channel_freq] = new_state

				# Append to recorder's buffer
				recorder = self.channel_recorders[channel_freq]
				recorder.append_audio(audio_samples)

		# Update sample counter for continuous phase tracking
		self.sample_counter += len(samples)

	async def scan (self) -> None:

		"""
		Main scanning loop
		Continuously scans the band and detects active channels
		"""

		logger.info("Starting scan...")

		try:
			self._setup_sdr()

			# Initialize async components
			self.loop = asyncio.get_running_loop()
			self.sample_queue = asyncio.Queue(maxsize=10)

			# Start async SDR streaming in background thread (non-blocking)
			# This must run in an executor because read_samples_async blocks
			async def start_streaming():
				await self.loop.run_in_executor(
					None,
					self.sdr.read_samples_async,
					self._sdr_callback,
					self.samples_per_slice
				)

			# Start streaming task in background
			asyncio.create_task(start_streaming())

			logger.info("Started async SDR streaming")

			async for samples in self._sample_band_async():
				# Process samples in executor to avoid blocking
				await self.loop.run_in_executor(
					None,
					self._process_samples,
					samples,
					self.loop
				)

		except KeyboardInterrupt:
			logger.info("Scan interrupted by user")
		except Exception as e:
			logger.error(f"Error during scan: {e}", exc_info=True)
		finally:
			# Cancel async streaming
			if self.sdr:
				try:
					self.sdr.cancel_read_async()
					logger.info("Cancelled async SDR streaming")
				except Exception as e:
					logger.warning(f"Error cancelling async read: {e}")

			await self._cleanup_sdr()


async def main (band_name:str=None) -> None:

	config_path = 'config.yaml'

	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)

	active_bands = config['scanner'].get('active_bands', [])

	# If no band is specified, choose the first 'active band'.

	if band_name is None:

		if not active_bands:

			logger.error("No active bands configured in scanner.active_bands")
			return

		band_name = active_bands[0]

	scanner = RadioScanner(config_path=config_path, band_name=band_name)

	await scanner.scan()

if __name__ == '__main__':

	asyncio.run(main(band_name='pmr'))
