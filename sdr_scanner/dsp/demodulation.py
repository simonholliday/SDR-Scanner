"""
Digital demodulation for common radio modulation schemes.

Provides demodulation functions for:
- NFM (Narrow FM): Used in PMR, amateur radio, public safety
- AM (Amplitude Modulation): Used in airband, marine VHF, broadcast AM

All demodulators accept IQ samples (already filtered to channel bandwidth)
and maintain state for continuous operation across multiple blocks.
"""

import numpy
import numpy.typing
import scipy.ndimage
import scipy.signal
import typing

import sdr_scanner.constants
import sdr_scanner.dsp.filters


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

	# Handle empty input
	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# FM demodulation: instantaneous frequency is the derivative of phase
	# Phase difference between consecutive samples gives frequency offset
	# Formula: angle(IQ[n] * conj(IQ[n-1])) = phase[n] - phase[n-1]
	# This is the "angle difference" or "polar discriminator" method
	if 'last_iq' not in state:
		state['last_iq'] = iq_samples[0]

	# Prepend last sample from previous block for continuous phase tracking
	iq_with_prev = numpy.concatenate(([state['last_iq']], iq_samples))
	demod = numpy.angle(iq_with_prev[1:] * numpy.conj(iq_with_prev[:-1]))
	state['last_iq'] = iq_samples[-1]

	# De-emphasis: undo the pre-emphasis applied by the transmitter
	# Transmitters boost high frequencies to improve SNR (signal compresses better)
	# Receivers apply matching low-pass filter (de-emphasis) to restore flat response
	# This is a simple 1-pole IIR filter: H(z) = alpha / (1 - (1-alpha)*z^-1)
	tau = sdr_scanner.constants.NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + sample_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	# Remove DC offset: the demodulated signal may have a DC component
	# This can come from receiver frequency offset or transmitter carrier drift
	demod_dc_blocked = demod_deemph - numpy.mean(demod_deemph)

	# Normalize to [-1, 1] range based on expected deviation
	# The phase difference is in radians, we convert to normalized audio
	# For NFM with ±5 kHz deviation: max phase change = 2*pi * (5000 / sample_rate)
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * sdr_scanner.constants.NFM_DEVIATION_HZ / sample_rate)

	# Clip to prevent occasional over-deviation from causing distortion
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	return sdr_scanner.dsp.filters.decimate_audio(demod_normalized, sample_rate, audio_sample_rate, state)


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

	# Handle empty input
	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# AM demodulation: extract envelope using magnitude of IQ samples
	# The audio signal is encoded in the amplitude variations of the carrier
	# Formula: |IQ| = sqrt(I² + Q²) gives the instantaneous amplitude
	demod = numpy.abs(iq_samples)

	# Decimate to audio sample rate (envelope doesn't need full RF sample rate)
	audio, state = sdr_scanner.dsp.filters.decimate_audio(demod, sample_rate, audio_sample_rate, state)

	if len(audio) == 0:
		return audio.astype(numpy.float32, copy=False), state

	# Remove DC component: envelope detection creates a large DC offset
	# The carrier amplitude becomes DC, we only want the audio modulation
	# Use 30 Hz high-pass filter to remove DC while preserving voice (>100 Hz)
	cutoff_hz = 30.0

	# Create or reuse DC blocking filter (1st order Butterworth high-pass)
	if state.get('am_dc_fs') != audio_sample_rate or 'am_dc_sos' not in state:
		state['am_dc_sos'] = scipy.signal.butter(1, cutoff_hz, btype='highpass', fs=audio_sample_rate, output='sos')
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0
		state['am_dc_fs'] = audio_sample_rate
	elif 'am_dc_zi' not in state:
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0

	# Apply DC blocking filter with state preservation
	audio, state['am_dc_zi'] = scipy.signal.sosfilt(
		state['am_dc_sos'],
		audio,
		zi=state['am_dc_zi']
	)

	# Automatic Gain Control (AGC): compensate for varying signal strengths
	# AM signals can vary widely in amplitude (distant vs nearby transmitters)
	# AGC normalizes loudness by estimating signal level and adjusting gain
	# Uses vectorized filtering instead of per-sample loop for performance
	env = numpy.abs(audio)
	attack_ms = sdr_scanner.constants.AM_AGC_ATTACK_MS
	release_ms = sdr_scanner.constants.AM_AGC_RELEASE_MS
	floor = sdr_scanner.constants.AM_AGC_FLOOR

	# Convert time constants from milliseconds to sample counts
	attack_samples = max(1, int(audio_sample_rate * (attack_ms / 1000.0)))
	release_samples = max(1, int(audio_sample_rate * (release_ms / 1000.0)))

	# Two-stage envelope tracking:
	# 1. Fast attack: use maximum filter to quickly respond to loud signals
	#    This prevents distortion from sudden peaks (clipping prevention)
	# 2. Slow release: use averaging filter to gradually increase gain
	#    This sounds more natural and avoids "pumping" artifacts
	peak_env = scipy.ndimage.maximum_filter1d(env, size=attack_samples, mode='nearest')
	smooth_env = scipy.ndimage.uniform_filter1d(peak_env, size=release_samples, mode='nearest')
	level_arr = numpy.maximum(smooth_env, floor)

	# Smooth transition between blocks: blend previous level with current
	# Without blending, there could be a discontinuity at block boundaries
	# This would create an audible click or pop in the audio
	prev_level = state.get('am_agc_level')

	if prev_level is not None and len(level_arr) > 0:
		blend_len = min(attack_samples, len(level_arr))
		blend = numpy.linspace(0.0, 1.0, blend_len, dtype=numpy.float32)
		level_arr[:blend_len] = prev_level * (1.0 - blend) + level_arr[:blend_len] * blend

	# Apply AGC: divide signal by estimated level to normalize amplitude
	# This makes quiet signals louder and loud signals quieter
	output = (audio / level_arr).astype(numpy.float32)
	state['am_agc_level'] = float(level_arr[-1]) if len(level_arr) > 0 else floor

	# Apply output gain and clip to prevent distortion
	# AM demodulation can produce values outside [-1, 1] range
	output *= sdr_scanner.constants.AM_OUTPUT_GAIN
	output = numpy.clip(output, -1.0, 1.0)

	return output.astype(numpy.float32, copy=False), state


# Dictionary of available demodulators
DEMODULATORS: dict[str, typing.Callable] = {
	'NFM': demodulate_nfm,
	'AM': demodulate_am,
	# Future demodulators can be added here:
	# 'WFM': demodulate_wfm,
}


def get_demodulator (modulation: str) -> typing.Callable:

	"""
	Get demodulator function for a specific modulation type

	Args:
		modulation: Modulation type (e.g., 'NFM', 'AM')

	Returns:
		Demodulator function

	Raises:
		KeyError: If modulation type is not supported
	"""

	if modulation not in DEMODULATORS:
		available = ', '.join(DEMODULATORS.keys())
		raise KeyError(f"Unsupported modulation '{modulation}'. Available: {available}")

	return DEMODULATORS[modulation]
