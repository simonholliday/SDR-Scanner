"""
Demodulation functions for various modulation types
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
	Demodulate Narrow FM (NFM) from IQ samples with state preservation.

	Uses a polar discriminator for instantaneous frequency, then applies
	de-emphasis and DC blocking. Output is normalized and decimated to audio rate.

	Performance and SNR optimized by decimating IQ to a lower rate (near 48 kHz)
	using an integer factor before processing.
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# 1. Decimate IQ samples to an integer-multiple IF rate to reduce CPU and complexity.
	# The IF rate is kept high enough (3-4x audio rate) for reliable FM math and de-emphasis.
	target_if_rate = audio_sample_rate * sdr_scanner.constants.NFM_IF_OVERSAMPLE
	if_decimation = max(1, int(round(sample_rate / target_if_rate)))
	
	# Optimization: if the total ratio to audio is an integer, pick a factor of it.
	sr_int = int(round(sample_rate))
	if sr_int % audio_sample_rate == 0:
		total_factor = sr_int // audio_sample_rate
		# Find largest divisor of total_factor that is <= our calculated if_decimation
		for i in range(if_decimation, 0, -1):
			if total_factor % i == 0:
				if_decimation = i
				break
	
	if_rate = sample_rate / if_decimation

	iq_if, state = sdr_scanner.dsp.filters.decimate_iq(
		iq_samples,
		sample_rate,
		if_rate,
		state,
		state_prefix='nfm_iq_decim_'
	)

	if len(iq_if) == 0:
		return numpy.array([], dtype=numpy.float32), state

	# 2. IQ DC Removal (Robustness against ADC saturation)
	# Massive DC offsets from saturation "blind" the discriminator.
	# We subtract the mean of the block to re-center it.
	iq_if = iq_if - numpy.mean(iq_if)

	# 3. FM demodulation: instantaneous frequency is the phase difference per sample.
	if 'nfm_last_iq' not in state:
		state['nfm_last_iq'] = iq_if[0]

	iq_with_prev = numpy.concatenate(([state['nfm_last_iq']], iq_if))
	demod = numpy.angle(iq_with_prev[1:] * numpy.conj(iq_with_prev[:-1]))
	state['nfm_last_iq'] = iq_if[-1]

	# 4. De-emphasis compensates for transmitter pre-emphasis.
	tau = sdr_scanner.constants.NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + if_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	# 5. DC removal.
	if 'nfm_dc_sos' not in state:
		state['nfm_dc_sos'] = scipy.signal.butter(1, 30.0, btype='highpass', fs=if_rate, output='sos')
		state['nfm_dc_zi'] = scipy.signal.sosfilt_zi(state['nfm_dc_sos']) * 0.0

	demod_dc_blocked, state['nfm_dc_zi'] = scipy.signal.sosfilt(
		state['nfm_dc_sos'], demod_deemph, zi=state['nfm_dc_zi']
	)

	# 6. Normalize based on deviation and sample rate.
	# Increasing denominator slightly or adding gain here helps with low levels.
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * sdr_scanner.constants.NFM_DEVIATION_HZ / if_rate)
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# 7. Decimate to final audio rate.
	return sdr_scanner.dsp.filters.decimate_audio(
		demod_normalized,
		if_rate,
		audio_sample_rate,
		state
	)


def demodulate_am (
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Demodulate Amplitude Modulation (AM) from IQ samples with state preservation.

	Performs envelope detection, removes DC, and applies AGC.
	Efficiently converts to a real signal (envelope) before decimation.
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# 1. AM envelope detection: magnitude of IQ gives the audio envelope.
	# Done at the full RF rate for maximum timing precision before decimation.
	demod = numpy.abs(iq_samples)

	# 2. Decimate to final audio rate.
	audio, state = sdr_scanner.dsp.filters.decimate_audio(
		demod,
		sample_rate,
		audio_sample_rate,
		state
	)
	
	if len(audio) == 0:
		return audio.astype(numpy.float32, copy=False), state

	# 3. DC removal at audio rate.
	cutoff_hz = 30.0
	if state.get('am_dc_fs') != audio_sample_rate:
		state['am_dc_sos'] = scipy.signal.butter(1, cutoff_hz, btype='highpass', fs=audio_sample_rate, output='sos')
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0
		state['am_dc_fs'] = audio_sample_rate
	elif 'am_dc_zi' not in state:
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0

	audio, state['am_dc_zi'] = scipy.signal.sosfilt(
		state['am_dc_sos'],
		audio,
		zi=state['am_dc_zi']
	)

	# 4. Vectorized AGC.
	env = numpy.abs(audio)
	attack_ms = sdr_scanner.constants.AM_AGC_ATTACK_MS
	release_ms = sdr_scanner.constants.AM_AGC_RELEASE_MS
	floor = sdr_scanner.constants.AM_AGC_FLOOR

	attack_samples = max(1, int(audio_sample_rate * (attack_ms / 1000.0)))
	release_samples = max(1, int(audio_sample_rate * (release_ms / 1000.0)))

	peak_env = scipy.ndimage.maximum_filter1d(env, size=attack_samples, mode='nearest')
	smooth_env = scipy.ndimage.uniform_filter1d(peak_env, size=release_samples, mode='nearest')
	level_arr = numpy.maximum(smooth_env, floor)

	prev_level = state.get('am_agc_level')
	if prev_level is not None and len(level_arr) > 0:
		blend_len = min(attack_samples, len(level_arr))
		blend = numpy.linspace(0.0, 1.0, blend_len, dtype=numpy.float32)
		level_arr[:blend_len] = prev_level * (1.0 - blend) + level_arr[:blend_len] * blend

	output = (audio / level_arr).astype(numpy.float32)
	state['am_agc_level'] = float(level_arr[-1]) if len(level_arr) > 0 else floor
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
