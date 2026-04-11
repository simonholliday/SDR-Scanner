"""
Demodulation functions for various modulation types
"""

import functools
import logging
import typing

import numpy
import numpy.typing
import scipy.ndimage
import scipy.signal

import substation.constants
import substation.dsp.filters

logger = logging.getLogger(__name__)


def _pick_if_decimation (sample_rate: float, audio_sample_rate: int, oversample: float) -> int:

	"""
	Choose an integer decimation factor for the IF stage that produces a
	clean intermediate rate.

	The naive choice — round(sample_rate / target_if_rate) — is correct
	when sample_rate is an integer multiple of audio_sample_rate but
	produces pathological resampling otherwise.  For example, with
	sample_rate=2500000 and audio_sample_rate=16000, the naive answer is
	39, which yields if_rate=64103 — coprime with 2500000, so the
	downstream rational resampler tries to allocate a 50-million-tap
	filter (~400 MB) and locks the system up.

	This helper searches a window around the ideal decimation value for
	an integer divisor of sample_rate (so the IF decimation step uses
	fast integer downsampling), and returns the closest such divisor.
	If no integer divisor exists in the search window, it falls back to
	the rounded value (which will trigger rational resampling — but the
	defensive backstop in _decimate_common will catch the pathological
	cases).

	The selection prefers larger decimation factors when distances are
	tied: a higher decimation = lower IF rate = less downstream work.

	Args:
		sample_rate: SDR sample rate in Hz
		audio_sample_rate: Final audio rate in Hz
		oversample: How many times the audio rate the IF should sit at
			(typically 4 for NFM to give the FM discriminator headroom)

	Returns:
		An integer decimation factor.  Guaranteed >= 1.
	"""

	target_if_rate = audio_sample_rate * oversample
	ideal = max(1, int(round(sample_rate / target_if_rate)))

	sr_int = int(round(sample_rate))

	# Search a 2x window around the ideal value for integer divisors of
	# sample_rate.  The window is generous enough to find a clean divisor
	# for every common SDR sample rate while staying close to the
	# user-intended IF oversampling ratio.
	search_min = max(1, ideal // 2)
	search_max = max(search_min, ideal * 2)

	candidates = [d for d in range(search_min, search_max + 1) if sr_int % d == 0]

	if not candidates:
		# Pathological — no integer divisor in the window.  Fall back to
		# the rounded value; the rational-resample path will handle it
		# (or the backstop in _decimate_common will refuse and warn).
		return ideal

	# Pick the candidate closest to the ideal.  Tie-break by preferring
	# the larger decimation factor (lower IF rate, less downstream cost).
	return min(candidates, key=lambda d: (abs(d - ideal), -d))


def _apply_voice_agc (
	audio: numpy.typing.NDArray[numpy.float32],
	audio_sample_rate: int,
	state: dict,
	state_prefix: str,
) -> numpy.typing.NDArray[numpy.float32]:

	"""
	Vectorised AGC for voice-band audio.

	Smooths a peak-following envelope into a slow level estimate, then
	divides the audio by it.  The level estimate has a fast attack
	(catches transients) and slow release (avoids "pumping").  Cross-block
	continuity is preserved by blending the previous block's final level
	into the start of the current block over `attack_samples` samples.

	Shared between the AM and SSB demodulators because both produce a
	real audio envelope with similar amplitude variation characteristics.
	The state_prefix lets each caller keep its own AGC level so changing
	the band's modulation at runtime doesn't inherit stale state.
	"""

	if len(audio) == 0:
		return audio

	env = numpy.abs(audio)
	attack_ms = substation.constants.AM_AGC_ATTACK_MS
	release_ms = substation.constants.AM_AGC_RELEASE_MS
	floor = substation.constants.AM_AGC_FLOOR

	attack_samples = max(1, int(audio_sample_rate * (attack_ms / 1000.0)))
	release_samples = max(1, int(audio_sample_rate * (release_ms / 1000.0)))

	peak_env = scipy.ndimage.maximum_filter1d(env, size=attack_samples, mode='nearest')
	smooth_env = scipy.ndimage.uniform_filter1d(peak_env, size=release_samples, mode='nearest')
	level_arr = numpy.maximum(smooth_env, floor)

	level_key = state_prefix + 'agc_level'
	prev_level = state.get(level_key)

	if prev_level is not None and len(level_arr) > 0:
		blend_len = min(attack_samples, len(level_arr))
		blend = numpy.linspace(0.0, 1.0, blend_len, dtype=numpy.float32)
		level_arr[:blend_len] = prev_level * (1.0 - blend) + level_arr[:blend_len] * blend

	output = (audio / level_arr).astype(numpy.float32)
	state[level_key] = float(level_arr[-1]) if len(level_arr) > 0 else floor

	output *= substation.constants.AM_OUTPUT_GAIN
	return numpy.clip(output, -1.0, 1.0).astype(numpy.float32, copy=False)

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

	# 1. Decimate IQ samples to an IF rate that is an integer divisor of
	# the input sample rate AND keeps the IF at roughly NFM_IF_OVERSAMPLE x
	# the audio rate.  Picking a clean divisor here is essential — see the
	# _pick_if_decimation helper for the gory details.  Without this, a
	# 2.5 MHz AirSpy R2 → 16 kHz audio path would round to 39, producing
	# if_rate=64103 (coprime with 2500000) and a 50 million tap rational
	# resampling filter that locks the system up.
	if_decimation = _pick_if_decimation(
		sample_rate,
		audio_sample_rate,
		substation.constants.NFM_IF_OVERSAMPLE,
	)

	sr_int = int(round(sample_rate))
	if_rate = sr_int // if_decimation if sr_int % if_decimation == 0 else int(round(sample_rate / if_decimation))

	iq_if, state = substation.dsp.filters.decimate_iq(
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
	tau = substation.constants.NFM_DEEMPHASIS_TAU
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
	demod_normalized = demod_dc_blocked / (2 * numpy.pi * substation.constants.NFM_DEVIATION_HZ / if_rate)
	demod_normalized = numpy.clip(demod_normalized, -1.0, 1.0)

	# 7. Decimate to final audio rate.
	return substation.dsp.filters.decimate_audio(
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

	Pipeline: decimate IQ to a clean intermediate rate, take the envelope,
	decimate the envelope to audio rate, DC remove, apply AGC.

	The IF decimation step keeps the resampling chain manageable: without
	it, sample rates that don't have a small-gcd ratio with the audio rate
	(e.g. AirSpy R2 at 2.5 MHz → 16 kHz audio) would force decimate_audio
	into a multi-million-tap rational resampler.  See _pick_if_decimation
	for the helper that chooses a decimation factor.
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	# 1. Decimate IQ to a clean intermediate rate (chosen so the rest of the
	# chain runs as cheap integer / small-rational resampling).
	if_decimation = _pick_if_decimation(
		sample_rate,
		audio_sample_rate,
		substation.constants.AM_IF_OVERSAMPLE,
	)

	sr_int = int(round(sample_rate))
	if_rate = sr_int // if_decimation if sr_int % if_decimation == 0 else int(round(sample_rate / if_decimation))

	iq_if, state = substation.dsp.filters.decimate_iq(
		iq_samples,
		sample_rate,
		if_rate,
		state,
		state_prefix='am_iq_decim_',
	)

	if len(iq_if) == 0:
		return numpy.array([], dtype=numpy.float32), state

	# 2. AM envelope detection at the intermediate rate.  Voice AM
	# bandwidth is ~5 kHz so a 50-100 kHz IF rate has plenty of headroom
	# for the envelope to track without aliasing.
	demod = numpy.abs(iq_if)

	# 3. Decimate envelope to final audio rate.
	audio, state = substation.dsp.filters.decimate_audio(
		demod,
		if_rate,
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

	# 4. Vectorised AGC (shared helper, also used by demodulate_ssb).
	output = _apply_voice_agc(audio, audio_sample_rate, state, state_prefix='am_')

	return output, state


def demodulate_ssb (
	iq_samples: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	audio_sample_rate: int,
	sideband: str = 'USB',
	state: dict | None = None
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Demodulate Single Sideband (SSB) from IQ samples with state preservation.

	SSB transmissions occupy roughly 300 - 2700 Hz of audio bandwidth on
	one side of the carrier (USB = upper sideband, LSB = lower).  The
	channel-extracted IQ baseband from scanner.py is centered on the
	conventional SSB carrier frequency (a virtual point — there is no
	actual carrier in the transmission), so the wanted sideband sits in
	the positive frequencies of the IQ for USB and the negative
	frequencies for LSB.

	# ----- Algorithm choice ----------------------------------------------
	#
	# Three approaches were considered:
	#
	#   1. Frequency-shift + low-pass + take real part.  Simple but
	#      requires aggressive filtering to suppress the unwanted sideband
	#      sitting just on the other side of DC, and weak signals develop
	#      a "warbling" quality from filter ripple.
	#
	#   2. Hilbert / phasing method via scipy.signal.hilbert.  Clean
	#      mathematically (the IQ stream from a quadrature SDR is already
	#      analytic, in principle), but the FFT-based Hilbert transform
	#      has poor block-boundary continuity unless overlap-save is
	#      added, and that's a non-trivial amount of state to manage.
	#
	#   3. Weaver's method: complex frequency shift to centre the wanted
	#      sideband on DC, real-valued low-pass on each of I and Q, then
	#      shift back and take the real part.  This is the standard
	#      receiver implementation in commercial SDR software (Gqrx,
	#      SDR#, fldigi all use it or a close variant).  It composes
	#      cleanly with scipy's stateful sosfilt() so block-boundary
	#      continuity is automatic, the LPF cleanly rejects the unwanted
	#      sideband (which sits two bandwidths away from DC after the
	#      shift), and there are no FFTs in the audio path.
	#
	# We use Weaver's method.  The two oscillators are stateful (phase
	# carries across blocks) and the LPF uses scipy's standard sosfilt
	# state vectors, identical to the pattern used by demodulate_nfm and
	# demodulate_am.
	#
	# ----- Pipeline -------------------------------------------------------
	#
	#   IQ at sample_rate
	#     ↓ decimate_iq → IQ at audio_sample_rate
	#     ↓ multiply by exp(∓j 2π f_o t) (shift wanted sideband to DC)
	#     ↓ Butterworth LPF at ±SSB_AUDIO_HALF_BW_HZ on real and imag
	#     ↓ multiply by exp(±j 2π f_o t) (shift back)
	#     ↓ take real part → real audio
	#     ↓ DC removal (audio rate)
	#     ↓ shared voice AGC
	#   audio at audio_sample_rate
	#
	# The sign of the shift is what distinguishes USB from LSB:
	#   USB: shift down (negative), then back up (positive)
	#   LSB: shift up   (positive), then back down (negative)
	#
	# After the round-trip the wanted sideband ends up back at its
	# original IQ frequencies, while everything outside ±SSB_AUDIO_HALF_BW_HZ
	# (including the unwanted sideband and out-of-band noise) has been
	# rejected by the low-pass.

	Args:
		iq_samples: Channel-extracted complex IQ at sample_rate
		sample_rate: IQ sample rate in Hz
		audio_sample_rate: Final audio rate (typically 16 kHz)
		sideband: 'USB' or 'LSB' — selects which side of the carrier
			contains the wanted audio
		state: Per-channel state dict for cross-block continuity

	Returns:
		Tuple of (audio samples as float32 in [-1, 1], updated state dict)
	"""

	if len(iq_samples) == 0:
		return numpy.array([], dtype=numpy.float32), state if state else {}

	if state is None:
		state = {}

	if sideband not in ('USB', 'LSB'):
		raise ValueError(f"sideband must be 'USB' or 'LSB', got {sideband!r}")

	# 1. Decimate complex IQ to audio sample rate.
	# The channel filter upstream has already band-limited the IQ to
	# roughly the channel width (~4 kHz at 5 kHz spacing), so the 16 kHz
	# audio Nyquist is plenty.  Doing the rest of the pipeline at audio
	# rate keeps the LPF and frequency shifts cheap.
	iq_audio, state = substation.dsp.filters.decimate_iq(
		iq_samples,
		sample_rate,
		audio_sample_rate,
		state,
		state_prefix='ssb_iq_decim_'
	)

	if len(iq_audio) == 0:
		return numpy.array([], dtype=numpy.float32), state

	# 2. First frequency shift — bring the wanted sideband centre to DC.
	# USB: the audio sits in positive IQ frequencies → shift down
	# LSB: the audio sits in negative IQ frequencies → shift up
	# The oscillator phase is preserved across blocks so the join between
	# successive blocks doesn't introduce a click.
	f_o = substation.constants.SSB_AUDIO_CENTER_HZ
	shift_dir = -1.0 if sideband == 'USB' else +1.0

	n = len(iq_audio)
	sample_index = numpy.arange(n, dtype=numpy.float64)
	phase_offset = state.get('ssb_shift_phase', 0.0)
	step = 2.0 * numpy.pi * shift_dir * f_o / audio_sample_rate
	phases = step * sample_index + phase_offset
	iq_shifted = iq_audio * numpy.exp(1j * phases).astype(numpy.complex64)
	# Wrap to [0, 2π) so the float doesn't grow without bound.
	state['ssb_shift_phase'] = float((phase_offset + step * n) % (2.0 * numpy.pi))

	# 3. Low-pass filter at ±SSB_AUDIO_HALF_BW_HZ.
	# We apply the same real-valued Butterworth to the real and imaginary
	# parts independently — that combination is mathematically equivalent
	# to a complex bandpass on the original (pre-shift) IQ.  Each part
	# needs its own filter state vector.
	if state.get('ssb_lpf_fs') != audio_sample_rate:
		state['ssb_lpf_sos'] = scipy.signal.butter(
			substation.constants.SSB_LPF_ORDER,
			substation.constants.SSB_AUDIO_HALF_BW_HZ,
			btype='lowpass',
			fs=audio_sample_rate,
			output='sos',
		)
		zi_template = scipy.signal.sosfilt_zi(state['ssb_lpf_sos'])
		state['ssb_lpf_zi_real'] = zi_template * 0.0
		state['ssb_lpf_zi_imag'] = zi_template * 0.0
		state['ssb_lpf_fs'] = audio_sample_rate

	real_filt, state['ssb_lpf_zi_real'] = scipy.signal.sosfilt(
		state['ssb_lpf_sos'], iq_shifted.real, zi=state['ssb_lpf_zi_real']
	)
	imag_filt, state['ssb_lpf_zi_imag'] = scipy.signal.sosfilt(
		state['ssb_lpf_sos'], iq_shifted.imag, zi=state['ssb_lpf_zi_imag']
	)
	iq_filtered = (real_filt + 1j * imag_filt).astype(numpy.complex64)

	# 4. Second frequency shift — undo the first shift so the audio is
	# back at its original frequencies inside the IQ.  The sign is the
	# opposite of the first shift.
	phase_offset_back = state.get('ssb_unshift_phase', 0.0)
	step_back = 2.0 * numpy.pi * (-shift_dir) * f_o / audio_sample_rate
	phases_back = step_back * sample_index + phase_offset_back
	iq_unshifted = iq_filtered * numpy.exp(1j * phases_back).astype(numpy.complex64)
	state['ssb_unshift_phase'] = float((phase_offset_back + step_back * n) % (2.0 * numpy.pi))

	# 5. Take the real part — that is the audio.
	# Factor of 2 compensates for the fact that an analytic signal
	# carries half its energy in the real part and half in the imaginary
	# part; doubling the real part recovers the original amplitude.
	audio = (2.0 * iq_unshifted.real).astype(numpy.float32)

	# 6. DC removal at audio rate (high-pass at 30 Hz).
	cutoff_hz = 30.0
	if state.get('ssb_dc_fs') != audio_sample_rate:
		state['ssb_dc_sos'] = scipy.signal.butter(
			1, cutoff_hz, btype='highpass', fs=audio_sample_rate, output='sos'
		)
		state['ssb_dc_zi'] = scipy.signal.sosfilt_zi(state['ssb_dc_sos']) * 0.0
		state['ssb_dc_fs'] = audio_sample_rate

	audio, state['ssb_dc_zi'] = scipy.signal.sosfilt(
		state['ssb_dc_sos'], audio, zi=state['ssb_dc_zi']
	)

	# 7. AGC via the shared voice helper.
	output = _apply_voice_agc(audio, audio_sample_rate, state, state_prefix='ssb_')

	return output, state


# Dictionary of available demodulators.
# USB and LSB share demodulate_ssb via functools.partial — they're the
# same algorithm with different sideband flags, but exposed as separate
# top-level modulation types because that's how every other SDR program
# (Gqrx, SDR#, SDRConsole, fldigi, WSJT-X) labels them.  Users specifying
# a band as `modulation: USB` or `modulation: LSB` get the natural label
# without needing a separate sideband config field.
DEMODULATORS: dict[str, typing.Callable] = {
	'NFM': demodulate_nfm,
	'AM': demodulate_am,
	'USB': functools.partial(demodulate_ssb, sideband='USB'),
	'LSB': functools.partial(demodulate_ssb, sideband='LSB'),
	# Future demodulators can be added here:
	# 'WFM': demodulate_wfm,
}
