"""
Demodulation functions for various modulation types
"""

import numpy
import numpy.typing
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

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	if 'last_iq' not in state:
		state['last_iq'] = iq_samples[0]

	iq_with_prev = numpy.concatenate(([state['last_iq']], iq_samples))
	demod = numpy.angle(iq_with_prev[1:] * numpy.conj(iq_with_prev[:-1]))
	state['last_iq'] = iq_samples[-1]

	tau = sdr_scanner.constants.NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + sample_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	demod_dc_blocked = demod_deemph - numpy.mean(demod_deemph)

	demod_normalized = demod_dc_blocked / (2 * numpy.pi * sdr_scanner.constants.NFM_DEVIATION_HZ / sample_rate)

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

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	demod = numpy.abs(iq_samples)

	audio, state = sdr_scanner.dsp.filters.decimate_audio(demod, sample_rate, audio_sample_rate, state)

	if len(audio) == 0:
		return audio.astype(numpy.float32, copy=False), state

	cutoff_hz = 30.0

	if state.get('am_dc_fs') != audio_sample_rate or 'am_dc_sos' not in state:
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

	env = numpy.abs(audio)
	attack_ms = sdr_scanner.constants.AM_AGC_ATTACK_MS
	release_ms = sdr_scanner.constants.AM_AGC_RELEASE_MS

	if attack_ms <= 0:
		attack_coeff = 0.0
	else:
		attack_coeff = numpy.exp(-1.0 / (audio_sample_rate * (attack_ms / 1000.0)))

	if release_ms <= 0:
		release_coeff = 0.0
	else:
		release_coeff = numpy.exp(-1.0 / (audio_sample_rate * (release_ms / 1000.0)))

	level = state.get('am_agc_level')

	if level is None:
		level = max(float(numpy.mean(env)), sdr_scanner.constants.AM_AGC_FLOOR)

	output = numpy.empty_like(audio, dtype=numpy.float32)
	min_update = sdr_scanner.constants.AM_AGC_MIN_UPDATE_LEVEL
	floor = sdr_scanner.constants.AM_AGC_FLOOR

	for i in range(env.size):
		sample_env = float(env[i])

		if sample_env >= min_update:
			if sample_env > level:
				level = attack_coeff * level + (1.0 - attack_coeff) * sample_env
			else:
				level = release_coeff * level + (1.0 - release_coeff) * sample_env

			if level < floor:
				level = floor

		output[i] = audio[i] / level if level > 0.0 else audio[i]

	state['am_agc_level'] = level

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
