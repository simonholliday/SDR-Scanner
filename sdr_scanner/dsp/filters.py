"""
Audio processing and filtering functions
"""

import logging
import math
import numpy
import numpy.typing
import scipy.signal

logger = logging.getLogger(__name__)


def decimate_audio (
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
		if not state.get('resample_warned'):
			logger.warning(
				f"Non-integer resample ratio: {sr} -> {ar} Hz. "
				"Prefer a sample_rate that is an integer multiple of audio_sample_rate for lower CPU."
			)
			state['resample_warned'] = True

		# Use rational resampling when rates are not integer-divisible.
		# Keep enough overlap from the previous block to cover the FIR length.
		g = math.gcd(sr, ar)
		up = ar // g
		down = sr // g
		prev = state.get('resample_prev', numpy.array([], dtype=signal.dtype))
		last_ratio = state.get('resample_ratio')
		if last_ratio != (up, down):
			prev = numpy.array([], dtype=signal.dtype)
			state['resample_ratio'] = (up, down)

		current = signal
		if prev.size > 0:
			signal = numpy.concatenate([prev, current])
		resampled = scipy.signal.resample_poly(signal, up, down).astype(numpy.float32)
		if prev.size > 0:
			# Drop samples corresponding to the prepended overlap.
			out_prev = int(numpy.ceil(prev.size * up / down))
			resampled = resampled[out_prev:]

		# Store tail for the next block. Use FIR length to avoid boundary clicks.
		taps_len = state.get('resample_taps_len')
		if taps_len is None:
			taps_len = 10 * max(up, down) + 1
			state['resample_taps_len'] = taps_len
		overlap_len = min(current.size, int(taps_len))
		state['resample_prev'] = current[-overlap_len:].copy()
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


def apply_fade (
	audio: numpy.typing.NDArray[numpy.float32],
	sample_rate: int,
	fade_in_ms: float | None,
	fade_out_ms: float | None
) -> numpy.typing.NDArray[numpy.float32]:
	"""
	Apply linear fade-in/out to a 1-D audio array.
	Fades are applied independently - if one is None, only the other is applied.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		fade_in_ms: Fade-in duration in milliseconds (None to disable)
		fade_out_ms: Fade-out duration in milliseconds (None to disable)

	Returns:
		Audio signal with fades applied
	"""
	# Early exit only if both fades are disabled
	if fade_in_ms is None and fade_out_ms is None:
		return audio

	n_samples = len(audio)
	if n_samples == 0:
		return audio

	# Handle each fade independently
	fade_in_len = int(sample_rate * (fade_in_ms / 1000.0)) if fade_in_ms is not None else 0
	fade_out_len = int(sample_rate * (fade_out_ms / 1000.0)) if fade_out_ms is not None else 0

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
