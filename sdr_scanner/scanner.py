import asyncio
import logging
import time
import typing

import numpy
import numpy.lib.stride_tricks
import numpy.typing
import scipy.fft
import scipy.signal

import sdr_scanner.config
import sdr_scanner.constants
import sdr_scanner.devices
import sdr_scanner.dsp.demodulation
import sdr_scanner.dsp.filters
import sdr_scanner.recording

logger = logging.getLogger(__name__)

class RadioScanner:

	"""
	Self-contained radio scanner for SDR devices.

	Continuously monitors a frequency band and detects active transmissions
	by analyzing signal-to-noise ratio (SNR). When activity is detected above
	a threshold, the scanner can optionally demodulate and record the audio.

	Uses FFT-based power spectral density analysis with Welch averaging to
	reduce noise variance. Implements hysteresis (separate on/off thresholds)
	to prevent rapid state toggling when signals hover near the threshold.
	"""

	def __init__ (self, config_path:str='config.yaml', band_name:str='pmr', device_type:str='rtlsdr', device_index:int=0, config:typing.Any|None=None) -> None:

		"""
		Initialize the scanner with configuration

		Args:
			config_path: Path to the YAML configuration file
			band_name: Name of the band to scan (default: 'pmr')
			device_type: SDR type ('rtlsdr' or 'hackrf')
			device_index: Device index for the selected SDR type
		"""

		if config is None:
			self.config = sdr_scanner.config.load_config(config_path)
		else:
			self.config = sdr_scanner.config.validate_config(config)

		self.band_name = band_name
		self.device_type = device_type
		self.device_index = device_index

		if band_name not in self.config.bands:

			available = ', '.join(self.config.bands.keys())
			raise KeyError(f"Band '{band_name}' not found in configuration. Available bands: {available}")

		self.band_config = self.config.bands[band_name]
		self.scanner_config = self.config.scanner
		self.recording_config = self.config.recording

		self.freq_start = self.band_config.freq_start
		self.freq_end = self.band_config.freq_end
		self.channel_spacing = self.band_config.channel_spacing
		self.channel_width = self.band_config.channel_width
		self.sample_rate = self.band_config.sample_rate
		self.snr_threshold_db = self.band_config.snr_threshold_db
		# Hysteresis: use a lower threshold to turn OFF than to turn ON
		# This prevents rapid toggling when signal strength hovers near the threshold
		self.snr_threshold_off_db = self.snr_threshold_db - sdr_scanner.constants.HYSTERESIS_DB

		# Validate that the off threshold makes sense (must be positive)
		if self.snr_threshold_db <= sdr_scanner.constants.HYSTERESIS_DB:

			logger.error(f"CONFIG ERROR: Band '{band_name}' has snr_threshold_db ({self.snr_threshold_db} dB) <= HYSTERESIS_DB ({sdr_scanner.constants.HYSTERESIS_DB} dB)")
			logger.error(f"This would result in snr_threshold_off_db = {self.snr_threshold_off_db} dB")
			logger.error(f"Channels would never turn OFF because SNR rarely drops to 0 or below")
			logger.error(f"Please set snr_threshold_db to at least {sdr_scanner.constants.HYSTERESIS_DB + 0.1} dB")
			raise ValueError(f"Invalid snr_threshold_db for band '{band_name}': must be > {sdr_scanner.constants.HYSTERESIS_DB} dB")

		self.sdr_gain_db = self.band_config.sdr_gain_db

		self.modulation = self.band_config.modulation
		self.recording_enabled = self.band_config.recording_enabled
		self.audio_sample_rate = self.recording_config.audio_sample_rate
		self.buffer_size_seconds = self.recording_config.buffer_size_seconds
		self.disk_flush_interval = self.recording_config.disk_flush_interval_seconds
		self.audio_output_dir = self.recording_config.audio_output_dir
		self.fade_in_ms = self.recording_config.fade_in_ms
		self.fade_out_ms = self.recording_config.fade_out_ms
		self.soft_limit_drive = self.recording_config.soft_limit_drive
		self.hold_time_seconds = self.recording_config.recording_hold_time_ms / 1000.0

		self.can_demod = self.modulation in sdr_scanner.dsp.demodulation.DEMODULATORS

		# Check if recording is possible (enabled and demodulator available)
		self.can_record = self.recording_enabled and self.can_demod

		self.sdr_device_sample_size = self.scanner_config.sdr_device_sample_size
		self.band_time_slice_ms = self.scanner_config.band_time_slice_ms
		self.sample_queue_maxsize = self.scanner_config.sample_queue_maxsize

		# Calculate all channel frequencies based on start/end/spacing
		self.all_channels = self._calculate_channels()
		# Allow user to exclude specific channels by index (e.g., skip known interference)
		excluded_indices = set(self.band_config.exclude_channel_indices or [])
		out_of_range = sorted(idx for idx in excluded_indices if idx >= len(self.all_channels))
		if out_of_range:
			logger.warning(
				f"Ignoring out-of-range excluded channel indices for band '{band_name}': "
				f"{', '.join(str(idx) for idx in out_of_range)}"
			)
			excluded_indices -= set(out_of_range)

		self.channels = [
			freq for idx, freq in enumerate(self.all_channels)
			if idx not in excluded_indices
		]
		self.channel_original_indices = {
			freq: idx for idx, freq in enumerate(self.all_channels)
			if idx not in excluded_indices
		}
		self.num_channels = len(self.channels)

		# Track last time signal was above detection threshold for each channel
		# Used for "Hold Time" (hang time) logic to prevent early truncation
		self.channel_last_active_time: dict[float, float] = {ch_freq: 0.0 for ch_freq in self.channels}

		# Calculate edge margin - add padding on each side to avoid filter rolloff
		# Filters attenuate signals near the edge of the passband, so we need extra space
		self.band_edge_margin_hz = self.channel_spacing / 2

		# Calculate center frequency (midpoint of the band) where SDR will be tuned
		self.center_freq = (self.freq_start + self.freq_end) / 2
		# Required bandwidth includes the band span plus one channel width plus margin on each end
		self.required_bandwidth = self.freq_end - self.freq_start + self.channel_width + (2 * self.band_edge_margin_hz)

		# Channel state tracking: True = on, False = off
		self.channel_states: dict[float, bool] = {ch_freq: False for ch_freq in self.channels}

		# Stuck channel tracking
		self.channel_start_times: dict[float, float] = {}
		self.channel_last_warning_times: dict[float, float] = {}

		# Channel recorders: one per active channel
		self.channel_recorders: dict[float, sdr_scanner.recording.ChannelRecorder] = {}

		# Callbacks for channel state changes (ON/OFF)
		self.state_callbacks: list[typing.Callable] = []

		# Callbacks for completed recordings (Finalized)
		self.recording_callbacks: list[typing.Callable] = []

		# SDR device
		self.sdr: typing.Any | None = None

		# Pre-computed values (initialized in _precompute_fft_params)
		self.samples_per_slice: int = 0
		self.fft_size: int = 0
		self.window: numpy.typing.NDArray[numpy.float64] | None = None
		self.freqs: numpy.typing.NDArray[numpy.float64] | None = None
		self.channel_indices: dict[float, tuple[int, int]] = {}
		self.channel_bin_starts: numpy.typing.NDArray[numpy.int32] | None = None
		self.channel_bin_ends: numpy.typing.NDArray[numpy.int32] | None = None
		self.channel_dc_masks: list[numpy.typing.NDArray[numpy.bool_] | None] = []
		self.channel_list_index: dict[float, int] = {}
		self.noise_indices: list[tuple[int, int]] = []
		self.dc_mask: numpy.typing.NDArray[numpy.bool_] | None = None
		self.noise_mask: numpy.typing.NDArray[numpy.bool_] | None = None
		self.channel_filter_sos: numpy.typing.NDArray[numpy.float64] | None = None

		# Per-channel filter state for continuous processing (prevents clicks at block boundaries)
		# complex128 (float64 real+imag) prevents rounding drift in long-running sessions.
		self.channel_filter_zi: dict[float, numpy.typing.NDArray[numpy.complex128]] = {}

		# Per-channel demodulator state (last IQ sample and de-emphasis filter state)
		self.channel_demod_state: dict[float, dict] = {}

		# Pre-computed angular frequencies for frequency shifting (computed in _precompute_fft_params).
		self.channel_omega: dict[float, complex] = {}

		# Cumulative sample counter for continuous phase in frequency shifting
		# This ensures the oscillator used for frequency shifting doesn't reset between blocks
		self.sample_counter: int = 0

		# EMA-smoothed noise floor (dB) and warmup counter.
		# Smoothing eliminates per-slice jitter; warmup absorbs SDR startup transients.
		self._noise_floor_ema: float | None = None
		self._warmup_remaining: int = sdr_scanner.constants.NOISE_FLOOR_WARMUP_SLICES

		# Sample queue for async streaming
		self.sample_queue: asyncio.Queue | None = None
		self.loop: asyncio.AbstractEventLoop | None = None

		logger.info(f"Initialized scanner for band '{band_name}'")
		logger.info(f"Frequency range: {self.freq_start/1e6:.5f} - {self.freq_end/1e6:.5f} MHz")
		logger.info(f"Number of channels: {self.num_channels}")
		if excluded_indices:
			excluded_list = ", ".join(str(idx) for idx in sorted(excluded_indices))
			logger.info(f"Excluded channels: {excluded_list}")
		logger.info(f"Center frequency: {self.center_freq/1e6:.5f} MHz")
		logger.info(f"Required bandwidth: {self.required_bandwidth/1e6:.5f} MHz (inc. {self.band_edge_margin_hz/1e3:.1f}kHz edge margin)")
		logger.info(f"Sample rate: {self.sample_rate/1e6:.3f} MHz")
		logger.info(f"SNR threshold: {self.snr_threshold_db} dB ON / {self.snr_threshold_off_db} dB OFF (hysteresis)")
		logger.info(f"SDR Gain: {self.sdr_gain_db}")
		logger.info(f"SDR Device: {self.device_type} (index {self.device_index})")
		logger.info(f"Modulation: {self.modulation}")

		if self.can_record:
			status = f"ENABLED ({self.audio_sample_rate} Hz mono WAV to {self.audio_output_dir})"
		elif self.recording_enabled:
			status = f"DISABLED (no demodulator for {self.modulation})"
		else:
			status = "DISABLED"
		logger.info(f"Recording: {status}")


	def _calculate_channels (self) -> list[float]:

		"""
		Calculate all channel center frequencies in the band.

		Uses inclusive bounds from freq_start to freq_end with channel_spacing.
		Returns an ordered list for consistent indexing and logging.

		Returns:
			List of channel center frequencies in Hz
		"""

		# Use integer indexing to avoid floating-point accumulation drift.
		# Repeated `freq += spacing` would accumulate rounding error over many channels.
		n_channels = int(round((self.freq_end - self.freq_start) / self.channel_spacing)) + 1
		return [self.freq_start + i * self.channel_spacing for i in range(n_channels)]

	def add_state_callback (self, callback: typing.Callable) -> None:
		
		"""
		Add a callback function to be called when a channel changes state.

		The callback should accept (band_name: str, channel_index: int, is_active: bool, snr_db: float).
		Synchronous and asynchronous callbacks are both supported and will be
		executed on the main event loop.
		"""
		
		self.state_callbacks.append(callback)

	def add_recording_callback (self, callback: typing.Callable) -> None:
		
		"""
		Add a callback function to be called when a recording is finished and saved.

		The callback should accept (band_name: str, channel_index: int, file_path: str).
		Synchronous and asynchronous callbacks are both supported and will be
		executed on the main event loop.
		"""
		
		self.recording_callbacks.append(callback)

	def _precompute_fft_params (self) -> None:

		"""
		Pre-compute FFT windowing, frequency bins, and channel index ranges.

		This sets up the PSD analysis pipeline: samples per slice, Welch segment
		size, FFT window, frequency bins, channel bin ranges, and masks for DC
		spike exclusion and noise estimation. It is called once after the SDR
		sample rate is known.
		"""

		# Calculate samples per time slice
		time_slice_seconds = self.band_time_slice_ms / 1000.0
		self.samples_per_slice = int(self.sample_rate * time_slice_seconds)

		# Adjust to be multiple of device sample size
		self.samples_per_slice = -(-self.samples_per_slice // self.sdr_device_sample_size) * self.sdr_device_sample_size

		# FFT size per segment for Welch averaging (smaller segments reduce variance).
		self.fft_size = self.samples_per_slice // sdr_scanner.constants.WELCH_SEGMENTS

		# Warn if FFT size is not a power of two, which can impact performance.
		if (self.fft_size & (self.fft_size - 1) != 0) or self.fft_size == 0:
			logger.warning(
				f"FFT size ({self.fft_size}) is not a power of two. "
				"This may lead to slightly slower processing on some CPU architectures. "
				"To optimize, adjust band_time_slice_ms or sample_rate so that "
				"(sample_rate * time_slice_ms / 8000) is a power of two."
			)

		# Window function reduces spectral leakage in the FFT bins.
		self.window = scipy.signal.get_window('hann', self.fft_size).astype(numpy.float64)

		# Frequency bins (shifted so DC is in the middle for easy masking).
		freqs_unshifted = numpy.fft.fftfreq(self.fft_size, d=1.0/self.sample_rate)
		self.freqs = self.center_freq + numpy.fft.fftshift(freqs_unshifted)

		# Calculate observable frequency range based on sample rate
		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		# Calculate actual band span required (with margins).
		band_span = self.required_bandwidth

		# Check if band is too wide for sample rate (prevents silent/invalid scanning).
		if band_span > observable_span:
			logger.error(f"CONFIG ERROR: Band '{self.band_name}' spans {band_span/1e6:.2f} MHz but sample rate is only {self.sample_rate/1e6:.2f} MHz")
			logger.error(f"Band frequency range: {self.freq_start/1e6:.3f} - {self.freq_end/1e6:.3f} MHz (inc. margins)")
			logger.error(f"Observable frequency range: {observable_min_freq/1e6:.3f} - {observable_max_freq/1e6:.3f} MHz")
			logger.error(f"")
			logger.error(f"To fix this, you can either:")
			logger.error(f"  1. Split this band into multiple smaller bands of ~{observable_span*0.8/1e6:.1f} MHz each in sdr_scanner.config.yaml")
			logger.error(f"  2. Increase the sample_rate for this band (if your SDR hardware supports it)")
			logger.error(f"  3. Use a different SDR with higher bandwidth capability")
			logger.error(f"")
			raise ValueError(f"Band '{self.band_name}' is too wide ({band_span/1e6:.2f} MHz) for sample rate ({self.sample_rate/1e6:.2f} MHz)")

		# Pre-compute channel index ranges for fast per-channel power extraction.
		freq_resolution = self.sample_rate / self.fft_size
		self.channel_indices = {}
		channels_outside_range = []

		for channel_freq in self.channels:
			channel_half_width = self.channel_width / 2
			low_freq = channel_freq - channel_half_width
			high_freq = channel_freq + channel_half_width

			# Find indices covering the channel bandwidth.
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
			logger.warning(f"These channels will not be scanned. Check your band configuration in sdr_scanner.config.yaml")

		# Pre-compute noise estimation regions (gaps between channels).
		self._compute_noise_regions()

		# Pre-compute per-channel bin ranges once to avoid repeated dict lookups in hot paths.
		self.channel_list_index = {freq: idx for idx, freq in enumerate(self.channels)}
		self.channel_bin_starts = numpy.zeros(self.num_channels, dtype=numpy.int32)
		self.channel_bin_ends = numpy.zeros(self.num_channels, dtype=numpy.int32)
		self.channel_dc_masks = [None] * self.num_channels

		center_bin = self.fft_size // 2
		for idx, channel_freq in enumerate(self.channels):
			idx_start, idx_end = self.channel_indices[channel_freq]
			self.channel_bin_starts[idx] = idx_start
			self.channel_bin_ends[idx] = idx_end

			if idx_end > idx_start and idx_start <= center_bin < idx_end:
				# Exclude DC spike bins for channels that cross DC.
				local_dc_start = max(0, center_bin - sdr_scanner.constants.DC_SPIKE_BINS - idx_start)
				local_dc_end = min(idx_end - idx_start, center_bin + sdr_scanner.constants.DC_SPIKE_BINS + 1 - idx_start)
				mask = numpy.ones(idx_end - idx_start, dtype=bool)
				mask[local_dc_start:local_dc_end] = False
				self.channel_dc_masks[idx] = mask

		# Pre-compute DC spike mask for global noise estimation.
		self.dc_mask = numpy.ones(self.fft_size, dtype=bool)
		dc_start = max(0, center_bin - sdr_scanner.constants.DC_SPIKE_BINS)
		dc_end = min(self.fft_size, center_bin + sdr_scanner.constants.DC_SPIKE_BINS + 1)
		self.dc_mask[dc_start:dc_end] = False
		self.noise_mask = None
		if self.noise_indices:
			noise_mask = numpy.zeros(self.fft_size, dtype=bool)
			for idx_start, idx_end in self.noise_indices:
				noise_mask[idx_start:idx_end] = True
			noise_mask &= self.dc_mask
			if numpy.any(noise_mask):
				self.noise_mask = noise_mask

		# Pre-compute channel extraction filter for recording demodulation.
		if self.can_demod and self.recording_enabled:
			cutoff_freq = self.channel_width / 2
			normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
			self.channel_filter_sos = scipy.signal.butter(5, normalized_cutoff, btype='low', output='sos')

		# Pre-compute angular frequencies for frequency shifting.
		self.channel_omega = {}
		self.channel_phase_array = None # Shared if all channels have same slice size

		for channel_freq in self.channels:
			freq_offset = channel_freq - self.center_freq
			# Omega is the phase increment per sample in radians.
			self.channel_omega[channel_freq] = -2j * numpy.pi * freq_offset / self.sample_rate

		# Pre-compute a static phase array (0, 1, 2, ... N-1) to avoid re-allocating it in the hot path.
		self.phase_index_array = numpy.arange(self.samples_per_slice, dtype=numpy.float64)

		logger.info(f"FFT size: {self.fft_size} bins, frequency resolution: {freq_resolution:.1f} Hz")
		logger.info(f"Welch segments: {sdr_scanner.constants.WELCH_SEGMENTS}, samples per slice: {self.samples_per_slice}")
		logger.info(f"DC spike exclusion: {sdr_scanner.constants.DC_SPIKE_BINS * 2 + 1} bins around center")

	def _compute_noise_regions (self) -> None:

		"""
		Compute the index ranges for noise estimation.
		Uses the gaps between channels and areas outside the channel band.

		The noise regions intentionally avoid channel bins and the DC spike so
		noise floor estimation is stable even in busy bands.
		"""

		self.noise_indices = []

		sorted_channels = sorted(self.all_channels)

		# Calculate observable frequency range
		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		# Region before first channel (use edge of observable range, not band definition).
		first_channel = sorted_channels[0]
		first_channel_low = first_channel - self.channel_width / 2

		# Use the higher of: (observable minimum) or (band start - margin).
		noise_start_freq = max(observable_min_freq, self.freq_start - self.band_edge_margin_hz)

		band_start_idx = numpy.searchsorted(self.freqs, noise_start_freq)
		first_channel_idx = numpy.searchsorted(self.freqs, first_channel_low)

		gap_hz = (first_channel_low - noise_start_freq) / 1e3
		if first_channel_idx > band_start_idx:
			self.noise_indices.append((band_start_idx, first_channel_idx))
			logger.debug(f"Edge margin BEFORE first channel: {gap_hz:.1f} kHz ({first_channel_idx - band_start_idx} bins)")
		else:
			logger.debug(f"No gap before first channel (gap would be {gap_hz:.1f} kHz)")

		# Gaps between channels.

		inter_channel_gaps = 0

		for i in range(len(sorted_channels) - 1):

			ch1 = sorted_channels[i]
			ch2 = sorted_channels[i + 1]

			ch1_high = ch1 + self.channel_width / 2
			ch2_low = ch2 - self.channel_width / 2

			# Only use gap if there's actually space between channels.

			if ch2_low <= ch1_high:
				continue

			idx_start = numpy.searchsorted(self.freqs, ch1_high)
			idx_end = numpy.searchsorted(self.freqs, ch2_low)

			if idx_end > idx_start:

				gap_hz = (ch2_low - ch1_high) / 1e3
				self.noise_indices.append((idx_start, idx_end))
				inter_channel_gaps += 1
				logger.debug(f"Inter-channel gap {inter_channel_gaps}: {gap_hz:.1f} kHz ({idx_end - idx_start} bins)")

		# Region after last channel (use edge of observable range, not band definition).
		last_channel = sorted_channels[-1]
		last_channel_high = last_channel + self.channel_width / 2

		# Use the lower of: (observable maximum) or (band end + margin).
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

		# Store current settings for restoration after calibration
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

		logger.info(f"Calibrating SDR using known signal at {known_freq/1e6:.3f} MHz within {bandwidth/1e3:.0f} kHz bandwidth. This will take a few seconds...")

		for iteration in range(iterations, 0, -1):

			logger.debug(f"Calibration measurement {iterations - iteration + 1}/{iterations}...")

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

		# Bail out if no valid measurements were captured
		if not freq_correction_ppm_list:
			logger.warning("Calibration failed: no valid measurements collected")
			return

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

		"""
		Configure the SDR device with calculated parameters.

		Sets sample rate, center frequency, and gain, then optionally performs
		frequency calibration. FFT parameters are derived after the device is
		ready so all downstream processing uses the actual sample rate.
		"""

		logger.info("Setting up SDR device...")

		self.sdr = sdr_scanner.devices.create_device(self.device_type, self.device_index)
		self.sdr.sample_rate = self.sample_rate
		self.sdr.center_freq = self.center_freq

		# Per-element gain takes priority over overall gain for devices with
		# multiple gain stages (e.g., AirSpy R2: LNA, Mixer, VGA).
		if self.band_config.sdr_gain_elements and hasattr(self.sdr, 'gain_elements'):
			self.sdr.gain_elements = self.band_config.sdr_gain_elements
		elif self.sdr_gain_db == 'auto' or self.sdr_gain_db is None:
			try:
				self.sdr.gain = 'auto'
			except Exception:
				self.sdr.gain = None
		else:
			self.sdr.gain = self.sdr_gain_db

		# Apply device-specific settings (bias tee, clock source, etc.)
		if self.band_config.sdr_device_settings and hasattr(self.sdr, 'device_settings'):
			self.sdr.device_settings = self.band_config.sdr_device_settings

		serial = getattr(self.sdr, 'serial', None)

		if serial:
			logger.info(f"SDR serial: {serial}")

		logger.info("SDR device configured successfully")

		# Calibrate frequency offset if calibration frequency is provided

		calibration_freq = self.scanner_config.calibration_frequency_hz

		if calibration_freq is not None and hasattr(self.sdr, 'read_samples') and hasattr(self.sdr, 'freq_correction'):
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

	def _safe_queue_put (self, samples: numpy.typing.NDArray[numpy.complex64]) -> None:

		"""
		Safely put samples in queue, dropping them if queue is full
		This runs on the event loop thread, not the callback thread
		"""
		try:
			self.sample_queue.put_nowait(samples)
		except asyncio.QueueFull:
			# Drop samples if consumer is behind
			logger.warning("Sample queue full; dropping samples")

	def _sdr_callback (self, samples: numpy.typing.NDArray[numpy.complex64], _context: typing.Any) -> None:

		"""
		Callback for async SDR streaming (runs in librtlsdr background thread)

		Args:
			samples: IQ samples from SDR
			_context: Context object (unused)
		"""

		if self.loop and self.sample_queue:
			# Thread-safe: schedule queue put on the event loop.
			# The copy avoids buffer reuse in librtlsdr but costs memory bandwidth.
			# ascontiguousarray ensures optimal cache-line alignment for faster processing
			self.loop.call_soon_threadsafe(self._safe_queue_put, numpy.ascontiguousarray(samples))

	async def _sample_band_async (self) -> typing.AsyncGenerator[numpy.typing.NDArray[numpy.complex64], None]:
		"""
		Asynchronously sample the band using background streaming

		Yields:
			numpy.ndarray: Complex IQ samples for the time slice
		"""
		logger.info(f"Time slice: {self.band_time_slice_ms} ms ({self.samples_per_slice} samples)")

		while True:
			# Get samples from queue (filled by background thread).
			# A None sentinel signals that the streaming task has ended.
			samples = await self.sample_queue.get()
			if samples is None:
				return
			yield samples

	def _calculate_psd_data (self, samples: numpy.typing.NDArray[numpy.complex64], include_segment_psd: bool = True) -> tuple[numpy.typing.NDArray[numpy.float64], list[numpy.typing.NDArray[numpy.float64]] | None]:
		"""
		Calculate both averaged Welch PSD and per-segment PSDs.
		Uses vectorized batched FFT for performance.

		Returns:
			(psd_welch_db, segment_psds_db)
		"""

		segment_size = self.fft_size
		hop_size = segment_size // 2
		n_segments = (len(samples) - segment_size) // hop_size + 1
		if n_segments <= 0:
			raise ValueError("Not enough samples for PSD calculation")

		# Create overlapping segment views without copying data.
		samples_contig = numpy.ascontiguousarray(samples)
		segment_shape = (n_segments, segment_size)
		segment_strides = (samples_contig.strides[0] * hop_size, samples_contig.strides[0])
		segments = numpy.lib.stride_tricks.as_strided(samples_contig, shape=segment_shape, strides=segment_strides)

		# Apply window and run a batched FFT across segments.
		windowed = segments * self.window
		fft_results = scipy.fft.fft(windowed, axis=1)

		# Magnitude squared: faster than abs() ** 2 for complex arrays.
		mag_sq = fft_results.real ** 2 + fft_results.imag ** 2

		# Average and convert to dB for Welch.
		psd_avg = numpy.mean(mag_sq, axis=0)
		psd_welch_db = numpy.fft.fftshift(10.0 * numpy.log10(psd_avg + 1e-12))

		segment_psds_db = None

		if include_segment_psd:
			segment_psds_db_arr = 10.0 * numpy.log10(mag_sq + 1e-12)
			segment_psds_db = [numpy.fft.fftshift(segment_psds_db_arr[i]) for i in range(n_segments)]

		return psd_welch_db, segment_psds_db

	def _find_transition_index (self, samples:numpy.typing.NDArray[numpy.complex64], channel_freq:float, turning_on: bool, segment_psd: list[numpy.typing.NDArray[numpy.float64]] | None, segment_noise_floors: list[float] | None) -> int:

		"""
		Find the sample index within a chunk where a channel turns ON or OFF.
		Returns a conservative boundary based on per-segment SNR.
		"""

		if not segment_psd:
			return 0 if turning_on else len(samples)

		segment_size = self.fft_size
		hop_size = segment_size // 2
		threshold = self.snr_threshold_db if turning_on else self.snr_threshold_off_db

		if turning_on:
			# Per-segment SNR scan is CPU-heavy but only used for transition localization.
			for i, psd_db in enumerate(segment_psd):
				noise_floor = segment_noise_floors[i] if segment_noise_floors else self._estimate_noise_floor(psd_db)
				snr_db = self._get_channel_power(psd_db, channel_freq) - noise_floor
				if snr_db > threshold:
					return min(len(samples), i * hop_size)
			return 0

		for i, psd_db in enumerate(segment_psd):
			noise_floor = segment_noise_floors[i] if segment_noise_floors else self._estimate_noise_floor(psd_db)
			snr_db = self._get_channel_power(psd_db, channel_freq) - noise_floor
			if snr_db <= threshold:
				return min(len(samples), i * hop_size)

		return len(samples)

	@staticmethod
	def _refine_trim_on_audio (audio: numpy.typing.NDArray[numpy.float32], turning_on: bool) -> tuple[numpy.typing.NDArray[numpy.float32], int]:

		"""
		Refine a coarse PSD-based trim to sample-level precision on demodulated audio.

		Scans the audio for the first (turn-ON) or last (turn-OFF) sample that
		exceeds an amplitude threshold, then adds padding samples around that
		point.  The fade is later applied only to the padding region so that
		actual signal content (including attack transients) is never attenuated.

		Returns:
			(trimmed_audio, pad_samples) — the refined audio slice and the number
			of padding samples at the fade end (start for turn-ON, end for turn-OFF).
		"""

		threshold = sdr_scanner.constants.TRIM_AMPLITUDE_THRESHOLD
		n = len(audio)
		if n == 0:
			return audio, 0

		above = numpy.where(numpy.abs(audio) >= threshold)[0]

		if len(above) == 0:
			# No sample exceeds threshold — return as-is with no padding info
			return audio, 0

		if turning_on:
			first = int(above[0])
			pad = min(first, sdr_scanner.constants.TRIM_PRE_SAMPLES)
			start = first - pad
			return audio[start:], pad
		else:
			last = int(above[-1])
			remaining = n - 1 - last
			pad = min(remaining, sdr_scanner.constants.TRIM_POST_SAMPLES)
			end = last + 1 + pad
			return audio[:end], pad

	def _prepare_channel_transition (self, samples:numpy.typing.NDArray[numpy.complex64], channel_freq:float, channel_index:int, snr_db: float, is_active:bool, current_state:bool, segment_psd:list[numpy.typing.NDArray[numpy.float64]] | None, segment_noise_floors: list[float] | None, loop:asyncio.AbstractEventLoop) -> tuple[int, int, int, bool, bool]:

		"""
		Compute trim boundaries and update state for a channel transition.
		Returns trim_start, trim_end, sample_offset, turning_on, turning_off.
		"""

		turning_on = is_active and not current_state
		turning_off = (not is_active) and current_state

		trim_start = 0
		trim_end = len(samples)
		sample_offset = 0

		if turning_on or turning_off:

			transition_idx = self._find_transition_index(samples, channel_freq, turning_on, segment_psd, segment_noise_floors)
			transition_idx = max(0, min(len(samples), transition_idx))

			if turning_on:

				trim_start = transition_idx
				sample_offset = transition_idx

				# Record channel activity start for stuck detection.
				self.channel_start_times[channel_freq] = time.time()

				if channel_freq in self.channel_last_warning_times:
					del self.channel_last_warning_times[channel_freq]

			else:

				trim_end = transition_idx

				# Clean up tracking state when channel turns off.
				if channel_freq in self.channel_start_times:
					del self.channel_start_times[channel_freq]

				if channel_freq in self.channel_last_warning_times:
					del self.channel_last_warning_times[channel_freq]

			self.channel_states[channel_freq] = is_active

			state_str = "ON" if is_active else "OFF"
			channel_mhz = channel_freq / 1e6

			logger.info(f"Channel {channel_index} {state_str} (f = {channel_mhz:.5f} MHz, SNR = {snr_db:.1f}dB, recording: {'YES' if self.can_record else 'NO'})")
			logger.debug("".join("X" if self.channel_states[ch] else "-" for ch in self.channels))

			if turning_on and self.can_record:

				if channel_freq in self.channel_filter_zi:
					del self.channel_filter_zi[channel_freq]

				if channel_freq in self.channel_demod_state:
					del self.channel_demod_state[channel_freq]

				self._start_channel_recording(channel_freq, channel_index, snr_db, loop)

			# Trigger state change callbacks on the main event loop
			for callback in self.state_callbacks:

				if asyncio.iscoroutinefunction(callback):
					asyncio.run_coroutine_threadsafe(callback(self.band_name, channel_index, is_active, snr_db), loop)

				else:
					loop.call_soon_threadsafe(callback, self.band_name, channel_index, is_active, snr_db)

		return trim_start, trim_end, sample_offset, turning_on, turning_off

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

		# Get channel bins, excluding any that fall in DC spike region.
		channel_bins = psd_db[idx_start:idx_end]

		mask = None
		channel_idx = self.channel_list_index.get(channel_freq)
		if channel_idx is not None:
			mask = self.channel_dc_masks[channel_idx]

		if mask is not None:
			channel_bins = channel_bins[mask]
			if channel_bins.size == 0:
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

		if self.noise_mask is None:
			# Fallback to percentile method if no gaps defined
			return numpy.percentile(psd_db, 25)

		# Precomputed mask keeps noise estimation in numpy (no Python per-gap loops).
		noise_samples = psd_db[self.noise_mask]
		if noise_samples.size == 0:
			return numpy.percentile(psd_db, 25)

		# Use median of noise samples - robust to outliers
		return numpy.median(noise_samples)

	def _get_channel_powers (self, psd_db: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:

		"""
		Vectorized channel power extraction for all channels in scan order.
		Uses cumulative sums for the common case and falls back to masked mean
		for any channel that overlaps the DC spike.
		"""

		if self.channel_bin_starts is None or self.channel_bin_ends is None:
			return numpy.array([self._get_channel_power(psd_db, ch) for ch in self.channels], dtype=numpy.float64)

		counts = self.channel_bin_ends - self.channel_bin_starts
		powers = numpy.full(self.num_channels, -numpy.inf, dtype=numpy.float64)
		valid = counts > 0
		if numpy.any(valid):
			# Prefix sums avoid a Python loop per channel in the hot path.
			# We use a 1-indexed cumulative sum array to simplify the logic.
			csum = numpy.concatenate(([0.0], numpy.cumsum(psd_db)))
			sums = csum[self.channel_bin_ends[valid]] - csum[self.channel_bin_starts[valid]]
			powers[valid] = sums / counts[valid]

		# Handle DC spike masks for overlapping channels
		for idx, mask in enumerate(self.channel_dc_masks):
			if mask is None:
				continue
			idx_start = int(self.channel_bin_starts[idx])
			idx_end = int(self.channel_bin_ends[idx])
			if idx_end <= idx_start:
				powers[idx] = -numpy.inf
				continue
			channel_bins = psd_db[idx_start:idx_end]
			channel_bins = channel_bins[mask]
			powers[idx] = numpy.mean(channel_bins) if channel_bins.size else -numpy.inf

		return powers

	def _extract_channel_iq (self, samples: numpy.typing.NDArray[numpy.complex64], channel_freq: float, sample_offset: int = 0) -> numpy.typing.NDArray[numpy.complex64]:

		"""
		Extract IQ samples for a specific channel by frequency shifting and filtering.
		Uses internal filter state to maintain continuity across blocks.
		"""

		# Frequency shift to baseband using pre-computed angular frequency.
		n_samples = len(samples)
		omega = self.channel_omega.get(channel_freq)

		if omega is None:
			freq_offset = channel_freq - self.center_freq
			omega = -2j * numpy.pi * freq_offset / self.sample_rate

		# Compute oscillator with continuous phase based on cumulative sample count.
		start_sample = self.sample_counter + sample_offset
		start_phase = omega * start_sample

		# Optimization: use pre-computed phase array if chunk size matches.
		if n_samples == self.samples_per_slice and sample_offset == 0:
			phase_arr = self.phase_index_array
		else:
			phase_arr = numpy.arange(n_samples, dtype=numpy.float64)

		samples_shifted = samples * numpy.exp(start_phase + omega * phase_arr)

		# Initialize filter state if needed
		if channel_freq not in self.channel_filter_zi:
			zi = scipy.signal.sosfilt_zi(self.channel_filter_sos)
			self.channel_filter_zi[channel_freq] = (zi * 0.0).astype(numpy.complex128)

		# Low-pass filter with state preservation
		filtered, self.channel_filter_zi[channel_freq] = scipy.signal.sosfilt(
			self.channel_filter_sos,
			samples_shifted,
			zi=self.channel_filter_zi[channel_freq]
		)

		return filtered

	def _start_channel_recording (self, channel_freq:float, channel_index:int, snr_db:float, loop:asyncio.AbstractEventLoop) -> None:

		"""
		Start recording a channel

		Args:
			channel_freq: Channel center frequency in Hz
			channel_index: Channel index number
			snr_db: Passed for info only, used to generate filename_suffix
			loop: Event loop to use for creating async tasks
		"""

		# Create recorder instance

		filename_suffix = f"{snr_db:.1f}" + "dB_" + self.device_type + "_" + str(self.device_index)

		# Find the initial noise floor to provide a stable reference for noise reduction
		# This is better than let the recorder guess from short audio chunks.
		initial_noise_floor = getattr(self, '_last_noise_floor_db', None)

		channel_recorder = sdr_scanner.recording.ChannelRecorder(
			channel_freq=channel_freq,
			channel_index=channel_index,
			band_name=self.band_name,
			audio_sample_rate=self.audio_sample_rate,
			buffer_size_seconds=self.buffer_size_seconds,
			disk_flush_interval_seconds=self.disk_flush_interval,
			audio_output_dir=self.audio_output_dir,
			modulation=self.modulation,
			filename_suffix=filename_suffix,
			soft_limit_drive=self.soft_limit_drive,
			noise_reduction_enabled=self.recording_config.noise_reduction_enabled
		)

		# Pass the band-wide noise floor if we have it
		if initial_noise_floor is not None:
			channel_recorder.initial_noise_floor_db = initial_noise_floor

		# Start the async flush task using the provided event loop
		channel_recorder.flush_task = asyncio.run_coroutine_threadsafe(
			channel_recorder._flush_to_disk_periodically(),
			loop
		)

		# Store recorder
		self.channel_recorders[channel_freq] = channel_recorder

	async def _stop_channel_recording (self, channel_freq: float) -> None:
		
		"""
		Stop recording a channel and close the file

		Args:
			channel_freq: Channel center frequency in Hz
		"""
		
		if channel_freq not in self.channel_recorders:
			return

		channel_recorder = self.channel_recorders[channel_freq]

		# Close recorder (flushes buffer and closes WAV file)
		await channel_recorder.close()

		# Trigger recording finished callbacks on the main event loop
		# Run these after close() so we are sure the file is finished and metadata is appended.
		ch_idx = channel_recorder.channel_index
		filepath = channel_recorder.filepath

		loop = self.loop or asyncio.get_running_loop()

		for callback in self.recording_callbacks:

			if asyncio.iscoroutinefunction(callback):

				asyncio.run_coroutine_threadsafe(callback(self.band_name, ch_idx, filepath), loop)

			else:

				loop.call_soon_threadsafe(callback, self.band_name, ch_idx, filepath)

		# Remove from dictionary
		del self.channel_recorders[channel_freq]

	def _process_samples (self, samples: numpy.typing.NDArray[numpy.complex64], loop: asyncio.AbstractEventLoop) -> None:

		"""
		Process a single IQ slice to detect active channels and manage recording.

		Computes PSD, estimates noise floor, applies per-channel detection with
		hysteresis, and when recording is enabled, demodulates only the active
		channel regions to keep CPU use predictable.
		"""

		start_time = time.perf_counter()
		try:
			# Phase 1: sanity check for ADC saturation.
			clipping_threshold = 0.95
			# Subsample to keep the check cheap for large slices.
			subsample_step = max(1, len(samples) // 4096)
			subsamples = samples[::subsample_step]
			clipping_count = numpy.sum(
				(subsamples.real > clipping_threshold) | (subsamples.real < -clipping_threshold) |
				(subsamples.imag > clipping_threshold) | (subsamples.imag < -clipping_threshold)
			)
			clipping_percentage = clipping_count / len(subsamples) * 100

			if clipping_percentage > 0.1:
				logger.warning(f"ADC SATURATION: {clipping_percentage:.1f}% samples clipping. Reduce gain.")

			# Phase 1.5: Remove DC offset (center spike) from SDR.
			# Using a simple mean subtraction is common and effective for 8-bit SDRs.
			# This is done on the whole slice before any frequency shifting.
			samples = samples - numpy.mean(samples)

			# Phase 2: compute PSD (Welch) for detection.
			# Welch PSD is the primary CPU cost; segment PSDs computed lazily only when needed.
			# This saves 30-40% of FFT overhead when no channels are transitioning.
			psd_db, segment_psds = self._calculate_psd_data(samples, include_segment_psd=False)

			# Phase 3: estimate noise floor from gaps, not the whole band.
			# Raw per-slice estimate is EMA-smoothed to eliminate jitter.
			raw_noise_floor_db = self._estimate_noise_floor(psd_db)
			if self._noise_floor_ema is None:
				# Seed the EMA on the first slice (avoids slow ramp from zero)
				self._noise_floor_ema = raw_noise_floor_db
			else:
				alpha = sdr_scanner.constants.NOISE_FLOOR_EMA_ALPHA
				self._noise_floor_ema = alpha * raw_noise_floor_db + (1.0 - alpha) * self._noise_floor_ema
			noise_floor_db = self._noise_floor_ema
			self._last_noise_floor_db = noise_floor_db

			# Warmup: absorb SDR startup transients before enabling detection.
			if self._warmup_remaining > 0:
				self._warmup_remaining -= 1
				self.sample_counter += len(samples)
				if self._warmup_remaining == 0:
					logger.info("Noise floor warmup complete (%.1f dB). Detection enabled.", noise_floor_db)
				return

			# Phase 4: bulk energy check avoids per-channel work when quiet.
			# We use a threshold that's strictly lower than the lowest possible detection threshold.
			bulk_threshold_db = max(2.0, self.snr_threshold_off_db - 2.0)
			
			if self.dc_mask is not None:
				max_power = numpy.max(psd_db[self.dc_mask])
			else:
				max_power = numpy.max(psd_db)
				
			if max_power < noise_floor_db + bulk_threshold_db and not any(self.channel_states.values()):
				# Fast path: spectrum is quiet and no channels are currently active
				self.sample_counter += len(samples)
				return

			# Phase 5: compute per-channel metrics.
			# Vectorized channel powers reduce per-channel Python overhead in busy bands.
			channel_powers = self._get_channel_powers(psd_db)

			# Lazy-computed segment data for transition localization.
			segment_psds = None
			segment_noise_floors = None

			# Phase 6: handle transitions, demodulation, and recording.
			now = time.time()
			for i, channel_freq in enumerate(self.channels):
				idx = self.channel_original_indices.get(channel_freq, -1)
				snr_db = channel_powers[i] - noise_floor_db
				current_state = self.channel_states[channel_freq]

				# Stuck channel detection: warn if PTT seems stuck or interference is constant.
				if current_state and self.config.scanner.stuck_channel_threshold_seconds:

					ch_start_time = self.channel_start_times.get(channel_freq)

					if ch_start_time:

						duration = now - ch_start_time

						if duration > self.config.scanner.stuck_channel_threshold_seconds:

							last_warn = self.channel_last_warning_times.get(channel_freq, 0)

							# Rate limit warnings to every 60 seconds to avoid flooding.

							if now - last_warn > 60:

								ch_idx = self.channel_original_indices.get(channel_freq, -1)

								logger.warning(
									f"STUCK CHANNEL WARNING: Channel {ch_idx} ({channel_freq/1e6:.4f} MHz) "
									f"has been active for {duration:.0f} seconds"
								)

								self.channel_last_warning_times[channel_freq] = now
				
				threshold = self.snr_threshold_off_db if current_state else self.snr_threshold_db
				above_threshold = snr_db > threshold

				# Update last active time if signal is strong
				if above_threshold:
					self.channel_last_active_time[channel_freq] = now
				
				# Channel is "active" if signal is strong OR we are within the hold time window
				is_active = above_threshold or (now - self.channel_last_active_time.get(channel_freq, 0) < self.hold_time_seconds)

				# Compute segment PSDs lazily only when a transition is detected
				# This avoids expensive per-segment FFT when channels are stable
				if self.can_record and (is_active != current_state) and segment_psds is None:
					_, segment_psds = self._calculate_psd_data(samples, include_segment_psd=True)
					if segment_psds:
						segment_noise_floors = [self._estimate_noise_floor(psd) for psd in segment_psds]

				trim_start, trim_end, offset, turning_on, turning_off = self._prepare_channel_transition(
					samples, channel_freq, idx, snr_db,
					is_active, current_state, segment_psds, segment_noise_floors, loop
				)

				if (is_active or turning_off) and channel_freq in self.channel_recorders:
					# Demodulate only when we are actively recording to avoid wasted CPU.
					if trim_end > trim_start:
						channel_iq = self._extract_channel_iq(samples[trim_start:trim_end], channel_freq, sample_offset=offset)
						demod_func = sdr_scanner.dsp.demodulation.DEMODULATORS[self.modulation]
						demod_state = None if turning_on else self.channel_demod_state.get(channel_freq)

						audio, new_state = demod_func(channel_iq, self.sample_rate, self.audio_sample_rate, state=demod_state)

						# Sample-level trim refinement + fade-only-padding:
						# Refine the coarse PSD boundary on the demodulated audio,
						# then apply fades only to the padding region (not signal content).
						if turning_on:
							audio, pad_samples = self._refine_trim_on_audio(audio, turning_on=True)
							if self.fade_in_ms:
								audio = sdr_scanner.dsp.filters.apply_fade(audio, self.audio_sample_rate, self.fade_in_ms, None, pad_in_samples=pad_samples)
						elif turning_off:
							audio, pad_samples = self._refine_trim_on_audio(audio, turning_on=False)
							if self.fade_out_ms:
								audio = sdr_scanner.dsp.filters.apply_fade(audio, self.audio_sample_rate, None, self.fade_out_ms, pad_out_samples=pad_samples)

						if not turning_off:
							self.channel_demod_state[channel_freq] = new_state

						recorder = self.channel_recorders.get(channel_freq)
						if recorder:
							recorder.append_audio(audio)
						else:
							logger.warning(f"Channel {channel_index}: no recorder found, audio discarded")

				if turning_off:
					self.channel_filter_zi.pop(channel_freq, None)
					self.channel_demod_state.pop(channel_freq, None)
					if channel_freq in self.channel_recorders:
						asyncio.run_coroutine_threadsafe(self._stop_channel_recording(channel_freq), loop)

			# Update sample counter for continuous phase tracking
			self.sample_counter += len(samples)
		finally:
			elapsed = time.perf_counter() - start_time
			expected = len(samples) / self.sample_rate if self.sample_rate else 0.0
			if expected > 0.0 and elapsed > expected * 1.05:
				ratio = elapsed / expected
				logger.warning(
					f"Processing overrun: {elapsed:.3f}s for {expected:.3f}s slice "
					f"({ratio:.2f}x)"
				)

	async def scan (self) -> None:

		"""
		Main scanning loop.

		Sets up the SDR and async pipeline, streams IQ slices in the background,
		and offloads CPU-heavy processing to the executor to keep the event loop
		responsive.
		"""

		logger.info("Starting scan...")

		try:
			self._setup_sdr()

			# Initialize async components
			self.loop = asyncio.get_running_loop()
			self.sample_queue = asyncio.Queue(maxsize=self.sample_queue_maxsize)

			# Start async SDR streaming in background thread (non-blocking)
			# This must run in an executor because read_samples_async blocks
			async def start_streaming () -> None:
				await self.loop.run_in_executor(
					None,
					self.sdr.read_samples_async,
					self._sdr_callback,
					self.samples_per_slice
				)

			# Start streaming task in background.
			# Store the reference so errors don't vanish silently.
			streaming_task = asyncio.create_task(start_streaming())

			def _on_streaming_done (task: asyncio.Task) -> None:
				exc = task.exception() if not task.cancelled() else None
				if exc:
					logger.error(f"SDR streaming task failed: {exc}", exc_info=exc)
					# Signal the sample queue to unblock the processing loop
					if self.sample_queue:
						self.sample_queue.put_nowait(None)

			streaming_task.add_done_callback(_on_streaming_done)

			logger.info("Started async SDR streaming")

			async for samples in self._sample_band_async():
				# CPU-heavy processing stays off the event loop to keep async I/O responsive.
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

			try:
				await asyncio.shield(self._cleanup_sdr())
			except asyncio.CancelledError:
				logger.debug("Cleanup completed despite task cancellation")
