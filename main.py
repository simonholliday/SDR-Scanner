import asyncio
import collections
import datetime
import logging
import math
import numpy
import numpy.typing
import os
import rtlsdr
import scipy.signal
import soundfile
import struct
import threading
import time
import typing
import uuid
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NFM demodulation constants
NFM_DEEMPHASIS_TAU = 300e-6  # 300µs time constant (standard for NFM)
NFM_DEVIATION_HZ = 5000      # Max frequency deviation for normalization

# AM AGC constants
AM_AGC_ALPHA = 0.05          # AGC smoothing factor (lower = slower response)

class ChannelSpec:

	def __init__ (
		self,
		band_name: str,
		channel_index: int,
		channel_freq: float,
		modulation: str = "Unknown",
	) -> None:

		pass

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
		audio_output_dir: str,
		modulation: str = "Unknown",
		broadcast_wav_format: bool = True,
		description: str = "",
		originator: str = "SDR Scanner",
		include_coding_history: bool = True
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
			broadcast_wav_format: Enable Broadcast WAV format with BEXT chunk
			description: Description string for BEXT metadata
			originator: Originator string for BEXT metadata
			include_coding_history: Include coding history in BEXT
		"""

		self.channel_freq = channel_freq
		self.channel_index = channel_index
		self.band_name = band_name
		self.audio_sample_rate = audio_sample_rate
		self.disk_flush_interval = disk_flush_interval_seconds
		self.modulation = modulation
		self.broadcast_wav_format = broadcast_wav_format

		# Calculate maximum buffer size in samples
		max_buffer_samples = int(buffer_size_seconds * audio_sample_rate)

		# Create circular buffer (deque with maxlen drops oldest when full)
		self.audio_buffer: collections.deque = collections.deque(maxlen=max_buffer_samples)

		# Recording start time
		self.start_time = datetime.datetime.now()

		# Calculate TimeReference: samples since midnight (for multi-file sync)
		midnight = self.start_time.replace(hour=0, minute=0, second=0, microsecond=0)
		seconds_since_midnight = (self.start_time - midnight).total_seconds()
		self.time_reference = int(seconds_since_midnight * audio_sample_rate)

		# Broadcast WAV metadata
		self.description = description if description else f"{band_name.upper()} Channel {channel_index} - {channel_freq/1e6:.5f} MHz"
		self.originator = originator
		self.originator_reference = str(uuid.uuid4())[:32]  # Truncate to 32 chars
		self.include_coding_history = include_coding_history

		date_str = self.start_time.strftime("%Y-%m-%d")
		time_str = self.start_time.strftime("%H-%M-%S")
		filename = f"{date_str}_{time_str}_{band_name}_{channel_index}.wav"
		self.filepath = os.path.join(audio_output_dir, date_str, filename)

		# Create output directory with date subdirectory if needed
		os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

		# Open WAV file for writing using soundfile (supports Broadcast WAV)
		if self.broadcast_wav_format:
			# Prepare BEXT metadata for soundfile
			# Note: soundfile uses 'extra_info' parameter for BEXT chunk
			coding_history = ""
			if self.include_coding_history:
				coding_history = (
					f"A=PCM,F={audio_sample_rate},W=16,M=mono,T={modulation};"
					f"Frequency={channel_freq/1e6:.5f}MHz\r\n"
				)

			# Open file with soundfile
			self.wav_file = soundfile.SoundFile(
				self.filepath,
				mode='w',
				samplerate=audio_sample_rate,
				channels=1,
				subtype='PCM_16',
				format='WAV'
			)

			# Store BEXT metadata to write on close
			self.bext_metadata = {
				'description': self.description,
				'originator': self.originator,
				'originator_reference': self.originator_reference,
				'origination_date': self.start_time.strftime('%Y-%m-%d'),
				'origination_time': self.start_time.strftime('%H:%M:%S'),
				'time_reference': self.time_reference,
				'version': 1,
				'coding_history': coding_history
			}
		else:
			# Fallback to standard WAV (backward compatibility)
			self.wav_file = soundfile.SoundFile(
				self.filepath,
				mode='w',
				samplerate=audio_sample_rate,
				channels=1,
				subtype='PCM_16',
				format='WAV'
			)
			self.bext_metadata = None

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

		# soundfile expects float32 samples in range [-1.0, 1.0] for PCM_16 output
		# It will automatically convert to int16 internally
		with self._write_lock:

			# Write to WAV file (soundfile handles float32 to int16 conversion)
			self.wav_file.write(samples)

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

		# Close WAV file (this writes headers)
		self.wav_file.close()

		# Write BEXT chunk if BWF format enabled
		if self.broadcast_wav_format and self.bext_metadata:
			self._write_bext_chunk()

		duration_seconds = self.total_samples_written / self.audio_sample_rate
		logger.info(f"Stopped recording channel {self.channel_index} (f = {self.channel_freq/1e6:.5f} MHz) - Duration: {duration_seconds:.1f}s, File: {self.filepath}")

	def _write_bext_chunk(self) -> None:

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

		# Build BEXT chunk
		# BEXT chunk format (EBU Tech 3285)
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

	sr = int(round(sample_rate))
	ar = int(audio_sample_rate)
	if sr <= 0 or ar <= 0:
		return signal.astype(numpy.float32), state

	if sr == ar:
		return signal.astype(numpy.float32), state

	# If the ratio isn't an integer, we need rational resampling instead of decimation.
	decimation_factor = sr // ar

	if sr % ar != 0:
		# Use rational resampling when rates are not integer-divisible.
		# We keep a small overlap from the previous block to reduce boundary clicks.
		g = math.gcd(sr, ar)
		up = ar // g
		down = sr // g
		prev = state.get('resample_prev', numpy.array([], dtype=signal.dtype))
		if prev.size > 0:
			signal = numpy.concatenate([prev, signal])
		resampled = scipy.signal.resample_poly(signal, up, down).astype(numpy.float32)
		if prev.size > 0:
			# Drop samples corresponding to the prepended overlap.
			out_prev = int(numpy.ceil(prev.size * up / down))
			resampled = resampled[out_prev:]
		# Store tail for the next block.
		state['resample_prev'] = signal[-64:].copy()
		return resampled, state

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

	# Downsample with phase continuity for stream-safe decimation.
	start_idx = state['decimate_phase']
	audio_samples = filtered[start_idx::decimation_factor].astype(numpy.float32)

	remaining = (len(filtered) - start_idx) % decimation_factor
	state['decimate_phase'] = (decimation_factor - remaining) % decimation_factor

	return audio_samples, state

def _apply_fade(
	audio: numpy.typing.NDArray[numpy.float32],
	sample_rate: int,
	fade_in_ms: float | None,
	fade_out_ms: float | None
) -> numpy.typing.NDArray[numpy.float32]:

	"""
	Apply linear fade-in/out to a 1-D audio array.
	If either duration is None, fading is disabled.
	"""

	if fade_in_ms is None or fade_out_ms is None:
		return audio

	n_samples = len(audio)
	if n_samples == 0:
		return audio

	fade_in_len = int(sample_rate * (fade_in_ms / 1000.0))
	fade_out_len = int(sample_rate * (fade_out_ms / 1000.0))

	fade_in_len = min(fade_in_len, n_samples)
	fade_out_len = min(fade_out_len, n_samples - fade_in_len)

	if fade_in_len > 0:
		# Smoothstep curve for an S-shaped fade-in.
		ramp_in = numpy.linspace(0.0, 1.0, fade_in_len, endpoint=True, dtype=audio.dtype)
		ramp_in = ramp_in * ramp_in * (3.0 - 2.0 * ramp_in)
		audio[:fade_in_len] *= ramp_in

	if fade_out_len > 0:
		# Smoothstep curve for an S-shaped fade-out.
		ramp_out = numpy.linspace(0.0, 1.0, fade_out_len, endpoint=True, dtype=audio.dtype)
		ramp_out = ramp_out * ramp_out * (3.0 - 2.0 * ramp_out)
		ramp_out = 1.0 - ramp_out
		audio[-fade_out_len:] *= ramp_out

	return audio

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

	# Hysteresis margin in dB - channel turns ON at threshold, OFF at threshold minus HYSTERESIS_DB
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
		self.fade_in_ms = self.recording_config.get('fade_in_ms', None)
		self.fade_out_ms = self.recording_config.get('fade_out_ms', None)

		# Broadcast WAV format parameters
		self.broadcast_wav_format = self.recording_config.get('broadcast_wav_format', True)
		self.default_description = self.recording_config.get('default_description', '')
		self.originator = self.recording_config.get('originator', 'SDR Scanner')
		self.include_coding_history = self.recording_config.get('include_coding_history', True)

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
		if self.sdr_gain_db == 'auto' or self.sdr_gain_db is None:
			try:
				self.sdr.gain = 'auto'
			except Exception:
				self.sdr.gain = None
		else:
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
			try:
				self.loop.call_soon_threadsafe(self.sample_queue.put_nowait, samples.copy())
			except asyncio.QueueFull:
				# Drop samples if consumer is behind to avoid callback thread exceptions
				logger.debug("Sample queue full; dropping samples")

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

	def _calculate_psd_segments (
		self,
		samples: numpy.typing.NDArray[numpy.complex64]
	) -> list[numpy.typing.NDArray[numpy.float64]]:

		"""
		Calculate per-segment PSDs for transition localization.
		Uses the same segment size and overlap as Welch.
		"""

		n_samples = len(samples)
		segment_size = self.fft_size
		hop_size = segment_size // 2
		n_segments = (n_samples - segment_size) // hop_size + 1

		if n_segments <= 0:
			return []

		segment_psd = []
		for i in range(n_segments):
			start = i * hop_size
			end = start + segment_size
			segment = samples[start:end]
			windowed = segment * self.window
			fft_result = numpy.fft.fft(windowed)
			psd_db = 10 * numpy.log10(numpy.abs(fft_result) ** 2 + 1e-12)
			segment_psd.append(numpy.fft.fftshift(psd_db))

		return segment_psd

	def _find_transition_index(
		self,
		samples: numpy.typing.NDArray[numpy.complex64],
		channel_freq: float,
		turning_on: bool,
		segment_psd: list[numpy.typing.NDArray[numpy.float64]] | None
	) -> int:

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
			for i, psd_db in enumerate(segment_psd):
				snr_db = self._get_channel_power(psd_db, channel_freq) - self._estimate_noise_floor(psd_db)
				if snr_db > threshold:
					return min(len(samples), i * hop_size)
			return 0

		for i, psd_db in enumerate(segment_psd):
			snr_db = self._get_channel_power(psd_db, channel_freq) - self._estimate_noise_floor(psd_db)
			if snr_db <= threshold:
				return min(len(samples), i * hop_size)

		return len(samples)

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

	def _extract_channel_iq (
		self,
		samples: numpy.typing.NDArray[numpy.complex64],
		channel_freq: float,
		sample_offset: int = 0
	) -> numpy.typing.NDArray[numpy.complex64]:

		"""
		Extract IQ samples for a specific channel by frequency shifting and filtering

		Args:
			samples: Full bandwidth IQ samples from SDR
			channel_freq: Center frequency of channel to extract
			sample_offset: Offset into the chunk for phase continuity

		Returns:
			Filtered IQ samples centered at baseband for the channel
		"""

		# Frequency shift to baseband using continuous phase (prevents clicks at block boundaries)
		freq_offset = channel_freq - self.center_freq
		n_samples = len(samples)
		t = (self.sample_counter + sample_offset + numpy.arange(n_samples)) / self.sample_rate
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
			audio_output_dir=self.audio_output_dir,
			modulation=self.modulation,
			broadcast_wav_format=self.broadcast_wav_format,
			description=self.default_description,
			originator=self.originator,
			include_coding_history=self.include_coding_history
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

		channel_metrics: dict[float, dict[str, typing.Any]] = {}
		transitions = []

		# Determine next state for each channel
		for channel_index, channel_freq in enumerate(self.channels):
			channel_power_db = self._get_channel_power(psd_db, channel_freq)
			snr_db = channel_power_db - noise_floor_db
			current_state = self.channel_states[channel_freq]

			if current_state:
				is_active = snr_db > self.snr_threshold_off_db
			else:
				is_active = snr_db > self.snr_threshold_db

			channel_metrics[channel_freq] = {
				'index': channel_index,
				'snr_db': snr_db,
				'is_active': is_active,
				'current_state': current_state
			}

			if is_active != current_state:
				transitions.append(channel_freq)

		segment_psd = None
		if transitions and self.can_record:
			# Only compute per-segment PSDs when transitions occur.
			segment_psd = self._calculate_psd_segments(samples)

		# Apply state changes and feed audio, trimming transition chunks if needed.
		for channel_freq in self.channels:
			metrics = channel_metrics[channel_freq]
			channel_index = metrics['index']
			snr_db = metrics['snr_db']
			is_active = metrics['is_active']
			current_state = metrics['current_state']

			turning_on = is_active and not current_state
			turning_off = (not is_active) and current_state

			trim_start = 0
			trim_end = len(samples)
			sample_offset = 0

			if turning_on or turning_off:
				transition_idx = self._find_transition_index(samples, channel_freq, turning_on, segment_psd)
				transition_idx = max(0, min(len(samples), transition_idx))
				if turning_on:
					trim_start = transition_idx
					sample_offset = transition_idx
				else:
					trim_end = transition_idx

				self.channel_states[channel_freq] = is_active
				state_str = "ON" if is_active else "OFF"
				channel_mhz = channel_freq / 1e6
				logger.info(f"Channel {channel_index} {state_str} (f = {channel_mhz:.5f} MHz, SNR = {snr_db:.1f}dB)")

				if turning_on:
					if self.can_record:
						if channel_freq in self.channel_filter_zi:
							del self.channel_filter_zi[channel_freq]
						if channel_freq in self.channel_demod_state:
							del self.channel_demod_state[channel_freq]
						self._start_channel_recording(channel_freq, channel_index, loop)

			# Feed audio samples to active recordings (include trimmed transition chunks).
			if (is_active or turning_off) and channel_freq in self.channel_recorders:
				if trim_end > trim_start:
					channel_iq = self._extract_channel_iq(
						samples[trim_start:trim_end],
						channel_freq,
						sample_offset=sample_offset
					)

					demodulator = DEMODULATORS[self.modulation]
					demod_state = None if turning_on else self.channel_demod_state.get(channel_freq, None)
					audio_samples, new_state = demodulator(
						channel_iq,
						self.sample_rate,
						self.audio_sample_rate,
						state=demod_state
					)

					if (turning_on or turning_off) and self.fade_in_ms is not None and self.fade_out_ms is not None:
						if turning_on and self.fade_in_ms > 0:
							audio_samples = _apply_fade(
								audio_samples,
								self.audio_sample_rate,
								self.fade_in_ms,
								0.0
							)
						elif turning_off and self.fade_out_ms > 0:
							audio_samples = _apply_fade(
								audio_samples,
								self.audio_sample_rate,
								0.0,
								self.fade_out_ms
							)

					if not turning_off:
						self.channel_demod_state[channel_freq] = new_state

					recorder = self.channel_recorders[channel_freq]
					recorder.append_audio(audio_samples)

			if turning_off:
				if channel_freq in self.channel_filter_zi:
					del self.channel_filter_zi[channel_freq]
				if channel_freq in self.channel_demod_state:
					del self.channel_demod_state[channel_freq]
				if channel_freq in self.channel_recorders:
					asyncio.run_coroutine_threadsafe(self._stop_channel_recording(channel_freq), loop)

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
