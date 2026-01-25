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
	tau = sdr_scanner.constants.NFM_DEEMPHASIS_TAU
	alpha = 1.0 / (1.0 + sample_rate * tau)

	if 'deemph_zi' not in state:
		state['deemph_zi'] = scipy.signal.lfilter_zi([alpha], [1, alpha - 1]) * 0.0

	demod_deemph, state['deemph_zi'] = scipy.signal.lfilter(
		[alpha], [1, alpha - 1], demod, zi=state['deemph_zi']
	)

	# DC removal using stateful highpass filter (prevents boundary artifacts)
	if 'nfm_dc_sos' not in state:
		state['nfm_dc_sos'] = scipy.signal.butter(1, 30.0, btype='highpass', fs=sample_rate, output='sos')
		state['nfm_dc_zi'] = scipy.signal.sosfilt_zi(state['nfm_dc_sos']) * 0.0

	demod_dc_blocked, state['nfm_dc_zi'] = scipy.signal.sosfilt(
		state['nfm_dc_sos'], demod_deemph, zi=state['nfm_dc_zi']
	)

	# Normalize to approximate [-1, 1] range
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * sdr_scanner.constants.NFM_DEVIATION_HZ / sample_rate)

	# Clip to [-1, 1] range
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# Decimate to audio sample rate
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

	# AM demodulation - extract magnitude (envelope detection)
	demod = numpy.abs(iq_samples)

	# Decimate to audio sample rate first to reduce CPU load.
	audio, state = sdr_scanner.dsp.filters.decimate_audio(demod, sample_rate, audio_sample_rate, state)
	if len(audio) == 0:
		return audio.astype(numpy.float32, copy=False), state

	# DC removal at audio rate with state to avoid boundary steps.
	cutoff_hz = 30.0
	# Reinitialize filter if sample rate changed
	if state.get('am_dc_fs') != audio_sample_rate:
		state['am_dc_sos'] = scipy.signal.butter(1, cutoff_hz, btype='highpass', fs=audio_sample_rate, output='sos')
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0
		state['am_dc_fs'] = audio_sample_rate
	# Initialize filter state if missing
	elif 'am_dc_zi' not in state:
		state['am_dc_zi'] = scipy.signal.sosfilt_zi(state['am_dc_sos']) * 0.0

	audio, state['am_dc_zi'] = scipy.signal.sosfilt(
		state['am_dc_sos'],
		audio,
		zi=state['am_dc_zi']
	)

	# Continuous AGC at audio rate using vectorized operations
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
		level = max(numpy.mean(env), sdr_scanner.constants.AM_AGC_FLOOR)

	min_update = sdr_scanner.constants.AM_AGC_MIN_UPDATE_LEVEL
	floor = sdr_scanner.constants.AM_AGC_FLOOR

	# Vectorized AGC using dual-filter approach
	# This replaces the sample-by-sample Python loop with fast numpy operations
	if 'am_agc_zi_attack' not in state:
		state['am_agc_zi_attack'] = scipy.signal.lfilter_zi([1 - attack_coeff], [1, -attack_coeff]) * level
	if 'am_agc_zi_release' not in state:
		state['am_agc_zi_release'] = scipy.signal.lfilter_zi([1 - release_coeff], [1, -release_coeff]) * level

	# Apply exponential smoothing filters
	attack_track, state['am_agc_zi_attack'] = scipy.signal.lfilter(
		[1 - attack_coeff], [1, -attack_coeff], env, zi=state['am_agc_zi_attack']
	)
	release_track, state['am_agc_zi_release'] = scipy.signal.lfilter(
		[1 - release_coeff], [1, -release_coeff], env, zi=state['am_agc_zi_release']
	)

	# Take minimum of attack and release tracks for proper AGC behavior
	level_track = numpy.minimum(attack_track, release_track)

	# Apply floor and min_update threshold
	level_track = numpy.maximum(level_track, floor)
	mask = env >= min_update
	if not numpy.all(mask):
		# Propagate last valid level where envelope is too low
		last_valid = level
		for i in range(len(level_track)):
			if mask[i]:
				last_valid = level_track[i]
			else:
				level_track[i] = last_valid

	# Apply gain reduction with safety against division by zero
	output = numpy.where(level_track > 1e-10, audio / level_track, audio)

	state['am_agc_level'] = level_track[-1]

	output *= sdr_scanner.constants.AM_OUTPUT_GAIN

	# Clip to [-1.0, 1.0] range
	output = numpy.clip(output, -1.0, 1.0)
	return output.astype(numpy.float32, copy=False), state


# Dictionary of available demodulators
DEMODULATORS: dict[str, typing.Callable] = {
	'NFM': demodulate_nfm,
	'AM': demodulate_am,
	# Future demodulators can be added here:
	# 'WFM': demodulate_wfm,
}
