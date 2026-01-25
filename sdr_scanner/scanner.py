import asyncio
import logging
import threading
import numpy
import numpy.lib.stride_tricks
import numpy.typing
import scipy.fft
import scipy.signal
import time
import typing

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

	def __init__ (self, config_path: str='config.yaml', band_name: str='pmr', device_type: str='rtlsdr', device_index: int=0, config: typing.Any|None=None) -> None:

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

		self.can_demod = self.modulation in sdr_scanner.dsp.demodulation.DEMODULATORS

		# Check if recording is possible (enabled and demodulator available)
		self.can_record = self.recording_enabled and self.can_demod

		# Scanner parameters
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

		# Calculate edge margin - add padding on each side to avoid filter rolloff
		# Filters attenuate signals near the edge of the passband, so we need extra space
		self.band_edge_margin_hz = self.channel_spacing / 2

		# Calculate center frequency (midpoint of the band) where SDR will be tuned
		self.center_freq = (self.freq_start + self.freq_end) / 2
		# Required bandwidth includes the band span plus one channel width plus margin on each end
		self.required_bandwidth = self.freq_end - self.freq_start + self.channel_width + (2 * self.band_edge_margin_hz)

		# Channel state tracking: True = on, False = off
		self.channel_states: dict[float, bool] = {ch_freq: False for ch_freq in self.channels}

		# Channel recorders: one per active channel
		self.channel_recorders: dict[float, sdr_scanner.recording.ChannelRecorder] = {}

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
		self.channel_dc_mask_indices: list[int] = []
		self.channel_list_index: dict[float, int] = {}
		self.noise_indices: list[tuple[int, int]] = []
		self.dc_mask: numpy.typing.NDArray[numpy.bool_] | None = None
		self.noise_mask: numpy.typing.NDArray[numpy.bool_] | None = None
		self.channel_filter_sos: numpy.typing.NDArray[numpy.float64] | None = None

		# Per-channel filter state for continuous processing (prevents clicks at block boundaries)
		self.channel_filter_zi: dict[float, numpy.typing.NDArray[numpy.complex64]] = {}

		# Per-channel demodulator state (last IQ sample and de-emphasis filter state)
		self.channel_demod_state: dict[float, dict] = {}

		# Pre-computed angular frequencies for frequency shifting (computed in _precompute_fft_params)
		# These are used to shift each channel to baseband (0 Hz) for demodulation
		self.channel_omega: dict[float, complex] = {}

		# Cumulative sample counter for continuous phase in frequency shifting
		# This ensures the oscillator used for frequency shifting doesn't reset between blocks
		self.sample_counter: int = 0

		# Track dropped samples to keep phase continuity without cross-thread updates
		# When the queue is full, we drop samples but must advance the phase counter accordingly
		self.dropped_samples = 0
		self.drop_lock = threading.Lock()

		# Queue monitoring for backpressure warnings
		# Track when we last warned about queue depth to avoid log spam
		self.last_queue_warning_time = 0.0
		self.queue_warning_interval = 5.0  # Warn at most once every 5 seconds

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
		Calculate all channel frequencies in the band.

		Generates equally-spaced channel center frequencies from freq_start to freq_end
		using the configured channel_spacing. For example, PMR446 has 16 channels
		spaced 12.5 kHz apart from 446.00625 MHz to 446.19375 MHz.

		Returns:
			List of channel center frequencies in Hz
		"""

		if self.channel_spacing <= 0:
			return []

		if self.freq_end < self.freq_start:
			return []

		# Use integer stepping to avoid cumulative floating-point drift
		# (repeated addition of floats introduces rounding errors)
		span = self.freq_end - self.freq_start

		# Allow a tiny tolerance so the final channel isn't missed due to float rounding
		tolerance = self.channel_spacing * 1e-6
		max_steps = int(numpy.floor((span + tolerance) / self.channel_spacing))

		return [
			self.freq_start + (index * self.channel_spacing)
			for index in range(max_steps + 1)
		]

	def _precompute_fft_params (self) -> None:

		"""
		Pre-compute FFT parameters, frequency bins, and channel index mappings.

		This method is called once after the SDR sample rate is known. It calculates
		the FFT size based on the configured time slice duration, generates the
		frequency bins that correspond to each FFT output, and pre-computes which
		bins correspond to each channel. This avoids repeated calculations in the
		hot path (per-sample processing).

		Also computes window functions (for spectral leakage reduction), DC spike
		masks (to exclude the DC offset artifact), and noise estimation regions
		(gaps between channels where we can measure the noise floor).
		"""

		# Calculate how many samples we process in each time slice
		time_slice_seconds = self.band_time_slice_ms / 1000.0
		self.samples_per_slice = int(self.sample_rate * time_slice_seconds)

		# Round up to a multiple of device sample size for efficient USB transfers
		# The -(-a // b) idiom is ceiling division: equivalent to ceil(a / b)
		self.samples_per_slice = -(-self.samples_per_slice // self.sdr_device_sample_size) * self.sdr_device_sample_size

		# FFT size per segment for Welch averaging
		# Welch method: split data into overlapping segments, compute PSD for each, then average
		# This reduces variance (noise) in the power estimate at the cost of frequency resolution
		self.fft_size = self.samples_per_slice // sdr_scanner.constants.WELCH_SEGMENTS

		# Window function (Hann window) reduces spectral leakage
		# Spectral leakage: FFT assumes signal repeats periodically; window tapers edges to reduce artifacts
		# Float32 keeps FFT inputs in complex64 to reduce CPU and memory
		self.window = scipy.signal.get_window('hann', self.fft_size).astype(numpy.float32)

		# Generate frequency bins that correspond to each FFT output
		# fftfreq gives frequencies from -Fs/2 to +Fs/2 relative to center
		freqs_unshifted = numpy.fft.fftfreq(self.fft_size, d=1.0/self.sample_rate)
		# fftshift rearranges from [0, +, -, -] to [-, -, 0, +, +] for intuitive ordering
		# Then add center_freq to get absolute frequencies in Hz
		self.freqs = self.center_freq + numpy.fft.fftshift(freqs_unshifted)

		# Calculate what frequency range is actually observable with this sample rate
		# Nyquist theorem: we can observe up to sample_rate/2 on each side of center
		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		# Check if the configured band fits within the observable range
		band_span = self.required_bandwidth

		# Validate that the band isn't too wide for the sample rate
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

		# Pre-compute which FFT bins correspond to each channel
		# This is done once here to avoid searching through frequency arrays repeatedly
		freq_resolution = self.sample_rate / self.fft_size
		self.channel_indices = {}
		channels_outside_range = []

		for channel_freq in self.channels:
			# Each channel occupies a bandwidth (e.g., 12.5 kHz for PMR)
			channel_half_width = self.channel_width / 2
			low_freq = channel_freq - channel_half_width
			high_freq = channel_freq + channel_half_width

			# Find which FFT bins fall within this channel's frequency range
			indices = numpy.where((self.freqs >= low_freq) & (self.freqs <= high_freq))[0]
			if len(indices) > 0:
				# Store as (start_index, end_index) for slicing
				self.channel_indices[channel_freq] = (indices[0], indices[-1] + 1)
			else:
				# Channel is outside observable range (shouldn't happen if validation passed)
				channels_outside_range.append(channel_freq)
				self.channel_indices[channel_freq] = (0, 0)

		# Warn about channels outside range (shouldn't happen if band span check passed)
		if channels_outside_range:
			logger.warning(f"CONFIG WARNING: {len(channels_outside_range)} channels fall outside observable frequency range:")
			for ch_freq in channels_outside_range:
				logger.warning(f"  - Channel {ch_freq/1e6:.5f} MHz is outside {observable_min_freq/1e6:.3f} - {observable_max_freq/1e6:.3f} MHz")
			logger.warning(f"These channels will not be scanned. Check your band configuration in sdr_scanner.config.yaml")

		# Pre-compute noise estimation regions (gaps between channels)
		self._compute_noise_regions()

		# Pre-compute per-channel bin ranges as arrays to enable vectorized operations
		# Using arrays instead of repeated dict lookups significantly speeds up the hot path
		self.channel_list_index = {freq: idx for idx, freq in enumerate(self.channels)}
		self.channel_bin_starts = numpy.zeros(self.num_channels, dtype=numpy.int32)
		self.channel_bin_ends = numpy.zeros(self.num_channels, dtype=numpy.int32)
		self.channel_dc_masks = [None] * self.num_channels

		# Track only channels that need DC masking to avoid unnecessary None checks per slice
		self.channel_dc_mask_indices = []

		# DC spike: most SDR receivers have a spike at the center frequency (0 Hz offset)
		# This is caused by LO (local oscillator) leakage and I/Q imbalance
		center_bin = self.fft_size // 2
		for idx, channel_freq in enumerate(self.channels):
			idx_start, idx_end = self.channel_indices[channel_freq]
			self.channel_bin_starts[idx] = idx_start
			self.channel_bin_ends[idx] = idx_end

			# If this channel includes the center frequency, we need to mask out the DC spike
			if idx_end > idx_start and idx_start <= center_bin < idx_end:
				# Calculate local indices relative to this channel's start
				local_dc_start = max(0, center_bin - sdr_scanner.constants.DC_SPIKE_BINS - idx_start)
				local_dc_end = min(idx_end - idx_start, center_bin + sdr_scanner.constants.DC_SPIKE_BINS + 1 - idx_start)
				# Create a mask: True = include, False = exclude from averaging
				mask = numpy.ones(idx_end - idx_start, dtype=bool)
				mask[local_dc_start:local_dc_end] = False
				self.channel_dc_masks[idx] = mask

				# Keep a list of channels that need masking for efficient iteration
				self.channel_dc_mask_indices.append(idx)

		# Pre-compute global DC spike mask (for all operations on the full spectrum)
		# This mask is True everywhere except around the center frequency
		self.dc_mask = numpy.ones(self.fft_size, dtype=bool)
		dc_start = max(0, center_bin - sdr_scanner.constants.DC_SPIKE_BINS)
		dc_end = min(self.fft_size, center_bin + sdr_scanner.constants.DC_SPIKE_BINS + 1)
		self.dc_mask[dc_start:dc_end] = False

		# Pre-compute noise estimation mask (bins in gaps between channels, excluding DC)
		# This lets us quickly extract noise floor estimates using vectorized operations
		self.noise_mask = None
		if self.noise_indices:
			noise_mask = numpy.zeros(self.fft_size, dtype=bool)
			for idx_start, idx_end in self.noise_indices:
				noise_mask[idx_start:idx_end] = True
			# Exclude DC spike from noise estimation (it's not representative of actual noise)
			noise_mask &= self.dc_mask
			if numpy.any(noise_mask):
				self.noise_mask = noise_mask

		# Pre-compute channel extraction filter (used when demodulating for recording)
		# This is a low-pass filter that isolates a single channel after frequency shifting
		if self.can_demod and self.recording_enabled:
			# Cutoff at half the channel width (e.g., 6.25 kHz for a 12.5 kHz channel)
			cutoff_freq = self.channel_width / 2
			# Normalize to Nyquist frequency (sample_rate / 2) as required by scipy
			normalized_cutoff = cutoff_freq / (self.sample_rate / 2)
			# 5th order Butterworth: good balance of rolloff steepness and phase linearity
			# SOS (second-order sections) format is more numerically stable than ba (poles/zeros)
			self.channel_filter_sos = scipy.signal.butter(5, normalized_cutoff, btype='low', output='sos')

		# Pre-compute angular frequencies for frequency shifting
		# Frequency shifting: multiply by e^(-j*2*pi*f_offset*t) to move channel to baseband (0 Hz)
		# Store as complex64 to keep phase math lightweight on constrained CPUs
		for channel_freq in self.channels:
			freq_offset = channel_freq - self.center_freq
			# Angular frequency: omega = -2*pi*f / Fs (negative because we're shifting down)
			omega = -2j * numpy.pi * freq_offset / self.sample_rate
			self.channel_omega[channel_freq] = numpy.complex64(omega)

		logger.info(f"FFT size: {self.fft_size} bins, frequency resolution: {freq_resolution:.1f} Hz")
		logger.info(f"Welch segments: {sdr_scanner.constants.WELCH_SEGMENTS}, samples per slice: {self.samples_per_slice}")
		logger.info(f"DC spike exclusion: {sdr_scanner.constants.DC_SPIKE_BINS * 2 + 1} bins around center")

	def _compute_noise_regions (self) -> None:

		"""
		Compute FFT bin ranges that contain only noise (no signal).

		Identifies "quiet" regions in the spectrum: gaps between channels and margins
		before the first/after the last channel. These regions are used to estimate
		the noise floor, which is more accurate than using a percentile of the entire
		spectrum (which may include weak signals that would skew the estimate).

		The noise floor is the baseline power level with no signal present, used as
		the reference when calculating SNR (signal-to-noise ratio).
		"""

		self.noise_indices = []

		# Can't identify gaps if there are no channels configured
		if not self.channels:
			logger.warning("No channels configured after exclusions; noise estimation regions disabled")
			return

		sorted_channels = sorted(self.channels)

		observable_span = self.sample_rate
		observable_min_freq = self.center_freq - observable_span / 2
		observable_max_freq = self.center_freq + observable_span / 2

		first_channel = sorted_channels[0]
		first_channel_low = first_channel - self.channel_width / 2
		noise_start_freq = max(observable_min_freq, self.freq_start - self.band_edge_margin_hz)

		band_start_idx = numpy.searchsorted(self.freqs, noise_start_freq)
		first_channel_idx = numpy.searchsorted(self.freqs, first_channel_low)

		gap_hz = (first_channel_low - noise_start_freq) / 1e3

		if first_channel_idx > band_start_idx:
			self.noise_indices.append((band_start_idx, first_channel_idx))
			logger.debug(f"Edge margin BEFORE first channel: {gap_hz:.1f} kHz ({first_channel_idx - band_start_idx} bins)")
		else:
			logger.debug(f"No gap before first channel (gap would be {gap_hz:.1f} kHz)")

		inter_channel_gaps = 0

		for i in range(len(sorted_channels) - 1):
			ch1 = sorted_channels[i]
			ch2 = sorted_channels[i + 1]

			ch1_high = ch1 + self.channel_width / 2
			ch2_low = ch2 - self.channel_width / 2

			if ch2_low <= ch1_high:
				continue

			idx_start = numpy.searchsorted(self.freqs, ch1_high)
			idx_end = numpy.searchsorted(self.freqs, ch2_low)

			if idx_end > idx_start:
				gap_hz = (ch2_low - ch1_high) / 1e3
				self.noise_indices.append((idx_start, idx_end))
				inter_channel_gaps += 1
				logger.debug(f"Inter-channel gap {inter_channel_gaps}: {gap_hz:.1f} kHz ({idx_end - idx_start} bins)")

		last_channel = sorted_channels[-1]
		last_channel_high = last_channel + self.channel_width / 2
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
		Calibrate SDR frequency offset using a known strong signal.

		Inexpensive SDR receivers (especially RTL-SDR) have crystal oscillators that
		drift from their nominal frequency, causing all received signals to appear
		shifted by a small amount (typically ±100 PPM or parts per million). This
		calibration measures the offset by tuning to a known signal (e.g., a strong
		FM broadcast station) and measuring where it actually appears in the spectrum.

		The measured offset is expressed in PPM and applied to the SDR's frequency
		correction setting, which shifts all subsequent tuning to compensate.

		Args:
			known_freq: Known signal frequency in Hz (e.g., 93.7 MHz for FM broadcast)
			bandwidth: Bandwidth to sample in Hz (default: 300 kHz, enough for FM)
			iterations: Number of measurements to average (default: 10 for statistical robustness)
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
		magnitude_db = None

		logger.info(f"Calibrating SDR using known signal at {known_freq/1e6:.3f} MHz within {bandwidth/1e3:.0f} kHz bandwidth. This will take a few seconds...")

		for iteration in range(iterations, 0, -1):
			logger.debug(f"Calibration measurement {iterations - iteration + 1}/{iterations}...")

			samples = self.sdr.read_samples(sample_size)

			# Apply Hanning window to reduce spectral leakage
			window = numpy.hanning(sample_size)
			fft_result = numpy.fft.fftshift(numpy.fft.fft(samples * window))
			# Generate frequency axis (relative to center frequency)
			freqs = numpy.fft.fftshift(numpy.fft.fftfreq(sample_size, 1 / self.sdr.sample_rate))
			# Convert to dB scale for easier analysis
			magnitude_db = 20 * numpy.log10(numpy.abs(fft_result) + 1e-10)

			# Only search within ±50 kHz of center (signal should be close)
			search_range_hz = 50e3
			freq_mask = numpy.abs(freqs) < search_range_hz

			if numpy.sum(freq_mask) == 0:
				logger.warning(f"No frequency bins within ±{search_range_hz/1e3:.0f} kHz search range")
				continue

			# Find the strongest peak in the search range
			peak_index_local = numpy.argmax(magnitude_db[freq_mask])
			freqs_filtered = freqs[freq_mask]
			measured_freq = self.sdr.center_freq + freqs_filtered[peak_index_local]
			peak_mag = magnitude_db[freq_mask][peak_index_local]

			peak_magnitudes.append(peak_mag)

			# Calculate frequency error in parts per million (PPM)
			# PPM = (measured - expected) / expected * 1e6
			freq_error_ppm = (measured_freq - known_freq) / known_freq * 1e6
			freq_correction_ppm_list.append(freq_error_ppm)

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

		# Remove outliers using IQR (Interquartile Range) method
		# This is robust against occasional bad measurements (e.g., interference)
		q1 = numpy.percentile(freq_correction_ppm_list, 25)  # 25th percentile
		q3 = numpy.percentile(freq_correction_ppm_list, 75)  # 75th percentile
		iqr = q3 - q1  # Interquartile range
		# Standard outlier threshold: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
		filtered_ppm = [x for x in freq_correction_ppm_list if q1 - 1.5*iqr <= x <= q3 + 1.5*iqr]

		# Use median of filtered values (more robust to outliers than mean)
		if len(filtered_ppm) == 0:
			logger.warning("All calibration measurements were outliers - using unfiltered data")
			filtered_ppm = freq_correction_ppm_list

		# Round to nearest integer PPM (SDR hardware only accepts integer PPM values)
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
		Initialize and configure the SDR hardware.

		Creates the SDR device object, sets the sample rate and center frequency,
		configures the gain (either auto or manual), and optionally calibrates
		the frequency offset if a calibration signal is configured.

		After hardware setup, pre-computes all FFT parameters and bin mappings
		that will be used during processing.
		"""

		logger.info("Setting up SDR device...")

		self.sdr = sdr_scanner.devices.create_device(self.device_type, self.device_index)
		self.sdr.sample_rate = self.sample_rate
		self.sdr.center_freq = self.center_freq

		if self.sdr_gain_db == 'auto' or self.sdr_gain_db is None:
			try:
				self.sdr.gain = 'auto'
			except Exception:
				self.sdr.gain = None
		else:
			self.sdr.gain = self.sdr_gain_db

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
		"""
		Clean up resources and shut down gracefully.

		Stops all active recordings (flushes buffers, closes files), then
		closes the SDR device. Called when the scan loop exits (either due
		to user interrupt or error).

		Exceptions during cleanup are logged but not raised, since we're
		already shutting down and errors here are expected (e.g., when
		interrupted by Ctrl+C, the SDR may already be in a bad state).
		"""
		# Stop and finalize all active recordings first
		for channel_freq in list(self.channel_recorders.keys()):
			await self._stop_channel_recording(channel_freq)

		# Close the SDR device and release hardware resources
		if self.sdr:
			try:
				self.sdr.close()
				logger.info("SDR device closed")
			except Exception as e:
				logger.warning(f"Error closing SDR device (this is normal on interrupt): {e}")

	def _safe_queue_put (self, samples: numpy.typing.NDArray[numpy.complex64]) -> None:
		"""
		Safely enqueue samples, dropping them if the consumer can't keep up.

		This is called by the SDR callback via call_soon_threadsafe, so it runs
		on the event loop thread (not the SDR background thread). It uses put_nowait
		to avoid blocking if the queue is full.

		If samples are dropped, we track the count so we can advance the phase
		counter accordingly (maintaining phase continuity for frequency shifting).

		Queue depth monitoring warns when the queue is filling up, giving early
		notice that processing may be falling behind. Warnings are throttled to
		avoid log spam (at most one warning per 5 seconds).
		"""
		# Check queue depth for early warning (throttled to avoid log spam)
		current_time = time.time()
		queue_size = self.sample_queue.qsize()
		max_size = self.sample_queue_maxsize
		fill_ratio = queue_size / max_size if max_size > 0 else 0.0

		# Warn when queue is 70% full (early warning of potential overrun)
		if fill_ratio >= 0.7 and (current_time - self.last_queue_warning_time) >= self.queue_warning_interval:
			logger.warning(f"Sample queue backpressure: {queue_size}/{max_size} ({fill_ratio:.0%}) - processing may be falling behind")
			self.last_queue_warning_time = current_time

		try:
			self.sample_queue.put_nowait(samples)
		except asyncio.QueueFull:
			# Processing is falling behind - drop this block of samples
			# This will cause gaps in recordings but prevents unbounded memory growth
			logger.warning(f"Sample queue full ({queue_size}/{max_size}); dropping {len(samples)} samples")

			# Track how many samples were dropped for phase continuity
			# The processing thread will advance sample_counter to compensate
			with self.drop_lock:
				self.dropped_samples += len(samples)

	def _sdr_callback (self, samples: numpy.typing.NDArray[numpy.complex64], _context: typing.Any) -> None:
		"""
		Callback for async SDR streaming (runs in librtlsdr background thread).

		This is called automatically by the SDR driver when new samples are available.
		It runs in a separate thread created by the SDR library, not our event loop,
		so we must use thread-safe operations to communicate with the main processing.

		The samples are IQ (in-phase/quadrature) data: complex numbers where the real
		part is I and the imaginary part is Q. This represents the signal as a vector
		that can rotate (showing frequency and phase changes).

		Args:
			samples: IQ samples from SDR (complex64 array)
			_context: Context object (unused, required by callback signature)
		"""
		if self.loop and self.sample_queue:
			# Thread-safe: schedule queue put on the event loop
			# call_soon_threadsafe allows the background thread to interact with asyncio
			# The copy() avoids buffer reuse issues in librtlsdr (they may reuse the buffer)
			# Use wrapper function to catch QueueFull exceptions in the event loop
			self.loop.call_soon_threadsafe(self._safe_queue_put, samples.copy())

	async def _sample_band_async (self) -> typing.AsyncGenerator[numpy.typing.NDArray[numpy.complex64], None]:
		"""
		Asynchronous generator that yields sample blocks from the SDR.

		This is the bridge between the SDR background thread (which calls
		_sdr_callback) and the main async processing loop. It pulls samples
		from the queue and yields them to the caller.

		The queue decouples the SDR hardware (which produces samples at a fixed
		rate) from the processing (which may have variable latency). If processing
		falls behind, the queue fills up and old samples are dropped.

		Yields:
			Complex IQ sample arrays, one block per time slice (e.g., 100ms worth)
		"""
		logger.info(f"Time slice: {self.band_time_slice_ms} ms ({self.samples_per_slice} samples)")

		while True:
			# Wait for next sample block from the queue (async, non-blocking)
			samples = await self.sample_queue.get()
			yield samples

	def _calculate_psd_data (self, samples: numpy.typing.NDArray[numpy.complex64], include_segment_psd: bool = True) -> tuple[numpy.typing.NDArray[numpy.float64], list[numpy.typing.NDArray[numpy.float64]] | None]:

		"""
		Calculate power spectral density (PSD) using Welch's method.

		PSD shows how power is distributed across frequencies. Welch's method splits
		the data into overlapping segments, computes the FFT of each segment, and
		averages the results. This reduces noise variance at the cost of some
		frequency resolution.

		The overlapping segments (50% overlap) provide better averaging without
		losing too much data between segments.

		Args:
			include_segment_psd: If True, also return individual segment PSDs (used
			                     for transition detection). If False, save CPU time.

		Returns:
			Tuple of (averaged_psd_db, segment_psds_db):
			- averaged_psd_db: Welch-averaged PSD in dB (lower noise, smoother)
			- segment_psds_db: List of individual segment PSDs in dB, or None if not requested
		"""

		segment_size = self.fft_size
		hop_size = segment_size // 2  # 50% overlap
		n_segments = (len(samples) - segment_size) // hop_size + 1

		if n_segments <= 0:
			raise ValueError("Not enough samples for PSD calculation")

		# Create overlapping segments view using stride tricks (no memory copy)
		# stride_tricks creates a view into the existing array with different shape/strides
		samples_contig = numpy.ascontiguousarray(samples)
		segment_shape = (n_segments, segment_size)
		segment_strides = (samples_contig.strides[0] * hop_size, samples_contig.strides[0])
		segments = numpy.lib.stride_tricks.as_strided(samples_contig, shape=segment_shape, strides=segment_strides)

		# Apply window function to each segment to reduce spectral leakage
		windowed = segments * self.window
		# Compute FFT for all segments at once (vectorized, much faster than a loop)
		fft_results = scipy.fft.fft(windowed, axis=1)

		# Compute magnitude squared (power): real²+ imag² is faster than abs()²
		mag_sq = fft_results.real ** 2 + fft_results.imag ** 2

		# Welch averaging: take the mean across all segments
		psd_avg = numpy.mean(mag_sq, axis=0)
		# Convert to dB scale and shift zero frequency to center
		psd_welch_db = numpy.fft.fftshift(10.0 * numpy.log10(psd_avg + 1e-12))

		# Optionally compute per-segment PSDs (only when needed to save CPU)
		segment_psds_db = None

		if include_segment_psd:
			segment_psds_db_arr = 10.0 * numpy.log10(mag_sq + 1e-12)
			segment_psds_db = [numpy.fft.fftshift(segment_psds_db_arr[i]) for i in range(n_segments)]

		return psd_welch_db, segment_psds_db

	def _find_transition_index (self, samples: numpy.typing.NDArray[numpy.complex64], channel_freq: float, turning_on: bool, segment_psd: list[numpy.typing.NDArray[numpy.float64]] | None, segment_noise_floors: list[float] | None) -> int:

		"""
		Find the sample index where a channel state transition occurs.

		When a channel turns ON or OFF, we want to know exactly where in the
		sample block this happened. This lets us start/stop recording at the
		right moment, avoiding recording silence or cutting off the beginning.

		We examine each segment's SNR to find when it crosses the threshold.
		This is CPU-intensive (requires per-segment SNR calculation) but only
		happens during transitions, not during steady-state operation.

		Args:
			samples: IQ sample block being analyzed
			channel_freq: Frequency of the channel in question
			turning_on: True if channel is turning ON, False if turning OFF
			segment_psd: Individual segment PSDs for fine-grained analysis
			segment_noise_floors: Pre-computed noise floor for each segment

		Returns:
			Sample index where the transition occurred (0 to len(samples))
		"""

		if not segment_psd:
			return 0 if turning_on else len(samples)

		segment_size = self.fft_size
		hop_size = segment_size // 2
		# Use appropriate threshold (hysteresis: different for ON vs OFF)
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

	def _prepare_channel_transition (self, samples: numpy.typing.NDArray[numpy.complex64], channel_freq: float, channel_index: int, snr_db: float, is_active: bool, current_state: bool, segment_psd: list[numpy.typing.NDArray[numpy.float64]] | None, segment_noise_floors: list[float] | None, loop: asyncio.AbstractEventLoop) -> tuple[int, int, int, bool, bool]:

		"""
		Handle channel state transitions (ON to OFF or OFF to ON).

		When a channel's SNR crosses the threshold, we need to:
		1. Find exactly where in the sample block the transition occurred
		2. Update the channel state
		3. Start/stop recordings as appropriate
		4. Log the state change
		5. Return trimming boundaries for precise audio extraction

		Args:
			samples: IQ sample block being processed
			channel_freq: Frequency of the channel that's transitioning
			channel_index: Channel number for logging
			snr_db: Current SNR for this channel
			is_active: New state (True = ON, False = OFF)
			current_state: Old state
			segment_psd: Per-segment PSDs for finding transition point
			segment_noise_floors: Noise floor for each segment
			loop: Event loop for starting async recording tasks

		Returns:
			Tuple of (trim_start, trim_end, sample_offset, turning_on, turning_off):
			- trim_start/end: sample indices to process (excludes silence before/after)
			- sample_offset: offset for phase continuity in frequency shifting
			- turning_on/off: boolean flags indicating the type of transition
		"""

		turning_on = is_active and not current_state
		turning_off = (not is_active) and current_state

		# Default: process entire block
		trim_start = 0
		trim_end = len(samples)
		sample_offset = 0

		if turning_on or turning_off:

			# Find the exact sample where the transition occurred
			transition_idx = self._find_transition_index(samples, channel_freq, turning_on, segment_psd, segment_noise_floors)
			transition_idx = max(0, min(len(samples), transition_idx))

			if turning_on:
				# Only process samples after the transition (skip silence before)
				trim_start = transition_idx
				sample_offset = transition_idx

			else:
				# Only process samples before the transition (skip silence after)
				trim_end = transition_idx

			# Update channel state
			self.channel_states[channel_freq] = is_active

			# Log the state change
			state_str = "ON" if is_active else "OFF"
			channel_mhz = channel_freq / 1e6

			logger.info(f"Channel {channel_index} {state_str} (f = {channel_mhz:.5f} MHz, SNR = {snr_db:.1f}dB, recording: {'YES' if self.can_record else 'NO'})")

			# Visual indicator of which channels are currently active
			channels_on_string = "".join("X" if self.channel_states[ch] else "-" for ch in self.channels)
			logger.info(f"Channels on: {channels_on_string}")

			# Start recording when channel turns on
			if turning_on and self.can_record:

				# Clear any stale filter/demodulator state from previous recording
				if channel_freq in self.channel_filter_zi:
					del self.channel_filter_zi[channel_freq]

				if channel_freq in self.channel_demod_state:
					del self.channel_demod_state[channel_freq]

				# Create recorder and start background flush task
				self._start_channel_recording(channel_freq, channel_index, snr_db, loop)

		return trim_start, trim_end, sample_offset, turning_on, turning_off

	def _get_channel_power (self, psd_db: numpy.typing.NDArray[numpy.float64], channel_freq: float) -> float:

		"""
		Extract the average power of a specific channel from the PSD.

		Uses pre-computed FFT bin indices to quickly slice out the frequency range
		that corresponds to this channel, then averages the power across those bins.

		If the channel includes the DC bin (center frequency), we exclude the DC
		spike region from the average to avoid artificially inflating the power reading.

		Args:
			psd_db: Power spectral density in dB (FFT output, shifted)
			channel_freq: Center frequency of the channel in Hz

		Returns:
			Average power of the channel in dB, or -inf if channel has no valid bins
		"""

		idx_start, idx_end = self.channel_indices[channel_freq]

		if idx_end <= idx_start:
			return -numpy.inf

		# Extract the bins that correspond to this channel's frequency range
		channel_bins = psd_db[idx_start:idx_end]

		# Apply DC spike mask if this channel overlaps the center frequency
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
		Estimate the noise floor by sampling quiet regions of the spectrum.

		The noise floor is the baseline power level when no signal is present.
		It's caused by thermal noise, amplifier noise, and environmental RF noise.

		We measure it by looking at gaps between channels where no signal should
		be present. This is more accurate than using a percentile of the entire
		spectrum, which could be skewed by weak signals that are below our
		detection threshold.

		Args:
			psd_db: Power spectral density in dB (shifted FFT output)

		Returns:
			Estimated noise floor in dB
		"""

		if self.noise_mask is None:
			# Fallback: use 25th percentile of entire spectrum if no gaps are defined
			# This assumes most of the spectrum is noise, not signal
			return numpy.percentile(psd_db, 25)

		# Extract power readings from pre-identified noise regions (vectorized, fast)
		noise_samples = psd_db[self.noise_mask]
		if noise_samples.size == 0:
			return numpy.percentile(psd_db, 25)

		# Use median instead of mean - more robust to occasional interference
		return numpy.median(noise_samples)

	def _get_channel_powers (self, psd_db: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:

		"""
		Extract power for all channels simultaneously (vectorized for performance).

		This is a performance-critical hot path called every time slice. Instead of
		looping through channels in Python (slow), we use numpy vectorized operations
		(fast) to compute all channel powers at once.

		Uses a clever cumulative sum trick: to get the sum of psd_db[a:b], we compute
		cumsum[b] - cumsum[a] rather than calling sum() for each channel.

		For channels that overlap the DC spike, we fall back to masked averaging
		(slower but necessary to exclude the DC spike artifact).

		Returns:
			Array of channel powers in dB, in the same order as self.channels
		"""

		# Fallback for edge cases where vectorization isn't set up
		if self.channel_bin_starts is None or self.channel_bin_ends is None:
			return numpy.array([self._get_channel_power(psd_db, ch) for ch in self.channels], dtype=numpy.float64)

		# Calculate how many bins each channel spans
		counts = self.channel_bin_ends - self.channel_bin_starts
		powers = numpy.full(self.num_channels, -numpy.inf, dtype=numpy.float64)
		valid = counts > 0
		if numpy.any(valid):
			# Cumulative sum trick: allows us to compute range sums in O(1) per channel
			# For a range [start:end], sum = cumsum[end] - cumsum[start]
			csum = numpy.concatenate(([0.0], numpy.cumsum(psd_db)))
			sums = csum[self.channel_bin_ends[valid]] - csum[self.channel_bin_starts[valid]]
			powers[valid] = sums / counts[valid]

		# Special handling for channels that overlap the DC spike (must exclude DC bins)
		# Only iterate over channels that actually need masking (typically 0 or 1)
		for idx in self.channel_dc_mask_indices:
			mask = self.channel_dc_masks[idx]
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
		Extract and isolate a single channel for demodulation.

		The SDR is tuned to the center of the band and receives all channels at once.
		To isolate a single channel, we:
		1. Frequency shift: multiply by a complex exponential to move the channel
		   to baseband (0 Hz). This is like heterodyning in analog radios.
		2. Low-pass filter: remove all other channels, keeping only our target.

		The frequency shift must maintain continuous phase across sample blocks,
		otherwise we'd introduce clicks/discontinuities in the audio. We track
		the phase using sample_counter.

		Filter state is preserved between blocks for the same reason - the filter
		has internal state (delay line) that must be continuous.

		Args:
			samples: IQ samples from the SDR (centered at center_freq)
			channel_freq: Frequency of the channel to extract
			sample_offset: Offset into the current block (for transition detection)

		Returns:
			Filtered IQ samples with the channel shifted to baseband
		"""

		# Frequency shift to baseband using pre-computed angular frequency
		n_samples = len(samples)
		omega = self.channel_omega.get(channel_freq)

		if omega is None:
			# Fallback if omega wasn't pre-computed (shouldn't happen normally)
			freq_offset = channel_freq - self.center_freq
			omega = -2j * numpy.pi * freq_offset / self.sample_rate

		# Generate complex oscillator with continuous phase across blocks
		# Formula: e^(j*omega*n) where n is the absolute sample number
		# Keep phase math in float32/complex64 to avoid unnecessary upcasts (performance)
		start_sample = self.sample_counter + sample_offset
		start_phase = omega * numpy.float32(start_sample)
		phases = start_phase + omega * numpy.arange(n_samples, dtype=numpy.float32)
		# Multiply by oscillator to shift frequency (complex multiplication rotates the vector)
		samples_shifted = samples * numpy.exp(phases)

		# Initialize filter state if this is a new recording (channel just turned on)
		if channel_freq not in self.channel_filter_zi:
			zi = scipy.signal.sosfilt_zi(self.channel_filter_sos)
			# Start with zero state (no history) to avoid transients
			self.channel_filter_zi[channel_freq] = (zi * 0.0).astype(numpy.complex64)

		# Apply low-pass filter to remove unwanted channels, preserving filter state
		# The filter state (zi) represents the internal delay line - must be continuous
		filtered, self.channel_filter_zi[channel_freq] = scipy.signal.sosfilt(
			self.channel_filter_sos,
			samples_shifted,
			zi=self.channel_filter_zi[channel_freq]
		)

		return filtered

	def _start_channel_recording (self, channel_freq: float, channel_index: int, snr_db: float, loop: asyncio.AbstractEventLoop) -> None:

		"""
		Start recording audio from a newly active channel.

		Creates a ChannelRecorder instance that manages buffered audio recording
		to WAV file. The recorder runs an async background task that periodically
		flushes the audio buffer to disk (to avoid blocking the main processing).

		Args:
			channel_freq: Channel center frequency in Hz (e.g., 446.00625e6)
			channel_index: Channel number for display/logging (e.g., 0-15 for PMR)
			snr_db: Signal strength in dB, included in filename for reference
			loop: Event loop to use for launching the background flush task
		"""

		# Build filename suffix with signal strength and device info
		filename_suffix = f"{snr_db:.1f}" + "dB_" + self.device_type + "_" + str(self.device_index)

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
			soft_limit_drive=self.soft_limit_drive
		)

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

		# Remove from dictionary
		del self.channel_recorders[channel_freq]

	def _process_samples (self, samples: numpy.typing.NDArray[numpy.complex64], loop: asyncio.AbstractEventLoop) -> None:

		"""
		Main signal processing: detect active channels and record audio.

		This is called for every time slice (typically every 100ms). It:
		1. Checks for ADC clipping (gain too high)
		2. Computes the power spectral density (FFT-based frequency analysis)
		3. Estimates the noise floor
		4. Detects which channels are active based on SNR
		5. Starts/stops recordings as channels turn on/off
		6. Demodulates and records audio for active channels

		Performance-critical: must process samples faster than they arrive or we'll
		drop data. Various optimizations are used (vectorization, lazy evaluation,
		early exits, etc.) to keep processing time under the time slice duration.

		Args:
			samples: IQ sample block from SDR (one time slice worth)
			loop: Event loop for async operations (starting/stopping recordings)
		"""

		start_time = time.perf_counter()

		try:

			# Account for any samples that were dropped due to queue overflow
			# We still need to advance the phase counter to maintain continuity
			with self.drop_lock:
				dropped_samples = self.dropped_samples
				self.dropped_samples = 0

			if dropped_samples:
				# Advance phase counter to keep oscillator synchronized
				self.sample_counter += dropped_samples

			# Quick clipping check: subsample to avoid checking all samples (performance)
			# Clipping indicates the gain is too high and the ADC is saturating
			clipping_threshold = 0.95  # IQ samples are normalized to ±1.0
			subsample_step = max(1, len(samples) // 4096)
			subsamples = samples[::subsample_step]
			clipping_count = numpy.sum(
				(subsamples.real > clipping_threshold) | (subsamples.real < -clipping_threshold) |
				(subsamples.imag > clipping_threshold) | (subsamples.imag < -clipping_threshold)
			)
			clipping_percentage = clipping_count / len(subsamples) * 100

			if clipping_percentage > 0.1:
				logger.warning(f"ADC SATURATION: {clipping_percentage:.1f}% samples clipping. Reduce gain.")

			# Compute Welch-averaged PSD (frequency analysis)
			# Skip per-segment PSDs initially - only compute them if we detect a transition
			# This saves significant CPU time when the spectrum is quiet
			psd_db, _ = self._calculate_psd_data(samples, include_segment_psd=False)

			# Estimate noise floor from quiet regions (gaps between channels)
			noise_floor_db = self._estimate_noise_floor(psd_db)

			# Optimization: skip expensive per-channel analysis if entire spectrum is quiet
			# Check if the strongest signal anywhere is below our detection threshold
			# Use a threshold slightly below snr_threshold_off_db to avoid missing signals
			bulk_threshold_db = max(2.0, self.snr_threshold_off_db - 2.0)

			# Find the peak power in the spectrum (excluding DC spike)
			if self.dc_mask is not None:
				max_power = numpy.max(psd_db[self.dc_mask])
			else:
				max_power = numpy.max(psd_db)

			# Fast path: if spectrum is quiet and nothing is recording, skip all processing
			if max_power < noise_floor_db + bulk_threshold_db and not any(self.channel_states.values()):
				# Just advance the sample counter and return early
				self.sample_counter += len(samples)
				return

			# Extract power for all channels at once (vectorized for performance)
			channel_powers = self._get_channel_powers(psd_db)

			# Segment PSDs are only needed for precise transition timing (CPU-intensive)
			# Compute them lazily: only when we actually detect a transition
			segment_psds = None
			segment_noise_floors = None

			# Main channel processing loop: check each channel for activity
			for i, channel_freq in enumerate(self.channels):
				# Calculate SNR for this channel (signal power minus noise floor)
				snr_db = channel_powers[i] - noise_floor_db
				current_state = self.channel_states[channel_freq]

				# Hysteresis: use different thresholds for turning on vs turning off
				# This prevents rapid toggling when SNR hovers near the threshold
				threshold = self.snr_threshold_off_db if current_state else self.snr_threshold_db
				is_active = snr_db > threshold

				# Detect state changes (channel turning on or off)
				turning_on = is_active and not current_state
				turning_off = (not is_active) and current_state

				# Lazy evaluation: only compute expensive per-segment PSDs when we detect a transition
				# and recording is enabled (transitions need precise timing to avoid cutting audio)
				if (turning_on or turning_off) and self.can_record and segment_psds is None:
					_, segment_psds = self._calculate_psd_data(samples, include_segment_psd=True)
					# Compute noise floor for each segment (used in transition detection)
					segment_noise_floors = [self._estimate_noise_floor(psd) for psd in segment_psds]

				# Default: process the entire sample block
				trim_start = 0
				trim_end = len(samples)
				offset = 0

				# Handle transitions: find exactly where in the block the transition occurred
				if turning_on or turning_off:
					# Look up original channel index for logging (before exclusions were applied)
					idx = self.channel_original_indices.get(channel_freq, -1)

					# Find transition point and update state
					trim_start, trim_end, offset, turning_on, turning_off = self._prepare_channel_transition(
						samples, channel_freq, idx, snr_db,
						is_active, current_state, segment_psds, segment_noise_floors, loop
					)

				# Process audio for active channels or channels that are turning off (fade out)
				if (is_active or turning_off) and channel_freq in self.channel_recorders:
					# Only demodulate if we're actually recording (saves CPU when not recording)
					if trim_end > trim_start:
						# Extract and isolate this channel (frequency shift + filter)
						channel_iq = self._extract_channel_iq(samples[trim_start:trim_end], channel_freq, sample_offset=offset)

						# Demodulate IQ to audio (e.g., FM demodulation)
						demod_func = sdr_scanner.dsp.demodulation.DEMODULATORS[self.modulation]
						# Preserve demodulator state for continuity (e.g., last phase for FM)
						# Reset state when turning on (fresh start)
						demod_state = None if turning_on else self.channel_demod_state.get(channel_freq)

						# Demodulation converts IQ samples to real audio samples
						audio, new_state = demod_func(channel_iq, self.sample_rate, self.audio_sample_rate, state=demod_state)

						# Apply fades to avoid clicks at start/end of recording
						if turning_on and self.fade_in_ms:
							audio = sdr_scanner.dsp.filters.apply_fade(audio, self.audio_sample_rate, self.fade_in_ms, 0.0)
						elif turning_off and self.fade_out_ms:
							audio = sdr_scanner.dsp.filters.apply_fade(audio, self.audio_sample_rate, 0.0, self.fade_out_ms)

						# Save demodulator state for next block (don't save when turning off)
						if not turning_off:
							self.channel_demod_state[channel_freq] = new_state

						# Add audio to the recording buffer (buffered writes for efficiency)
						self.channel_recorders[channel_freq].append_audio(audio)

				# Clean up resources when channel turns off
				if turning_off:
					# Clear filter state (no longer needed)
					self.channel_filter_zi.pop(channel_freq, None)
					# Clear demodulator state (no longer needed)
					self.channel_demod_state.pop(channel_freq, None)
					# Stop recording and close file (async operation)
					if channel_freq in self.channel_recorders:
						asyncio.run_coroutine_threadsafe(self._stop_channel_recording(channel_freq), loop)

			# Update sample counter for continuous phase tracking across blocks
			self.sample_counter += len(samples)
		finally:
			# Performance monitoring: check if we're keeping up with real-time processing
			# If processing takes longer than the time slice, we'll eventually drop samples
			elapsed = time.perf_counter() - start_time
			expected = len(samples) / self.sample_rate if self.sample_rate else 0.0
			# Allow 5% margin before warning (some variance is normal)
			if expected > 0.0 and elapsed > expected * 1.05:
				ratio = elapsed / expected
				logger.warning(
					f"Processing overrun: {elapsed:.3f}s for {expected:.3f}s slice "
					f"({ratio:.2f}x)"
				)

	async def scan (self) -> None:

		"""
		Main scanning loop - continuously monitors the band for activity.

		This is the main entry point for the scanner. It:
		1. Sets up the SDR device and computes all necessary parameters
		2. Starts async streaming from the SDR (runs in background thread)
		3. Continuously processes incoming samples to detect active channels
		4. Runs until interrupted (Ctrl+C) or an error occurs

		The scanning is fully asynchronous: the SDR streams samples in a background
		thread, the main loop processes them, and recordings are managed asynchronously.
		This allows efficient concurrent operation without blocking.
		"""
		logger.info("Starting scan...")

		try:
			# Configure SDR hardware and pre-compute all processing parameters
			self._setup_sdr()

			# Set up async infrastructure
			self.loop = asyncio.get_running_loop()
			# Queue to pass samples from SDR callback thread to processing
			self.sample_queue = asyncio.Queue(maxsize=self.sample_queue_maxsize)

			# Start async SDR streaming in background thread (non-blocking)
			# read_samples_async is a blocking C library call, so it must run in an executor
			# (thread pool) to avoid blocking the event loop
			async def start_streaming():
				await self.loop.run_in_executor(
					None,
					self.sdr.read_samples_async,
					self._sdr_callback,
					self.samples_per_slice
				)

			# Launch streaming task in the background (doesn't block)
			asyncio.create_task(start_streaming())

			logger.info("Started async SDR streaming")

			# Main processing loop: continuously process incoming samples
			async for samples in self._sample_band_async():
				# Run CPU-heavy processing in thread pool to keep event loop responsive
				# This allows async I/O (file writes, etc.) to continue without blocking
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
				pass
