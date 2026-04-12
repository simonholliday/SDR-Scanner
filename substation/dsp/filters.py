"""
Audio processing and filtering functions.

Provides sample rate conversion (decimation and resampling), audio fades,
and other signal conditioning needed for clean audio output from demodulated RF.

Key functions:
- decimate_audio: Convert from RF sample rate to audio sample rate with anti-aliasing
- decimate_iq: Convert complex iq samples to a lower sample rate (intermediate frequency)
- apply_fade: Apply smooth fade-in/fade-out to prevent clicks at recording boundaries
"""

import logging
import math

import numpy
import numpy.typing
import scipy.signal

logger = logging.getLogger(__name__)


# Track which resampling ratios have already been warned about to avoid log spam
_RESAMPLE_WARNED_RATIOS: set[tuple[int, int]] = set()

# Refuse to build a polyphase filter when max(up, down) exceeds this.  The
# filter length is approximately 2 * _RESAMPLE_HALF_LEN * max(up, down), so
# a max of 100_000 caps it at ~2 million taps (~16 MB of float64).  Anything
# beyond this is almost certainly a config bug — for example,
# sample_rate=2500000 with audio_sample_rate=16000 and a coprime intermediate
# IF rate produces max(up,down)=2500000, which would try to allocate hundreds
# of megabytes.  When this triggers we log a clear error and return
# zero-length output.
_RESAMPLE_MAX_FACTOR = 100_000

# Half-length of the Kaiser-windowed lowpass FIR used by the streaming
# polyphase resampler, in units of max(up, down).  A value of 10 gives a
# filter of length 2*10*max(up,down)+1, matching scipy.signal.resample_poly.
_RESAMPLE_HALF_LEN = 10


def _streaming_rational_resample (
	signal: numpy.typing.NDArray,
	up: int,
	down: int,
	state: dict,
	state_prefix: str,
) -> tuple[numpy.typing.NDArray, dict]:

	"""
	Stateful polyphase FIR rational resampler (up/down).

	Unlike scipy.signal.resample_poly, this function carries its filter
	state across calls so that block-by-block processing produces output
	identical to processing the entire signal at once — no zero-padding
	transients, no fractional-sample drift.

	The algorithm:
	  1.  Design a Kaiser-windowed lowpass FIR once and partition it into
	      `up` polyphase sub-filters.
	  2.  Maintain a short buffer of recent input samples (the FIR memory).
	  3.  For each output sample, pick the sub-filter whose fractional
	      delay matches the current output phase, dot-product it with the
	      buffer, and advance the input pointer by down/up.

	This is the standard polyphase resampler used by libsamplerate, soxr,
	and GStreamer's audioresample.
	"""

	fir_key = f'{state_prefix}poly_fir'
	buf_key = f'{state_prefix}poly_buf'
	phase_key = f'{state_prefix}poly_phase'
	offset_key = f'{state_prefix}poly_offset'

	if fir_key not in state:
		n_taps = 2 * _RESAMPLE_HALF_LEN * max(up, down) + 1
		cutoff = min(1.0 / up, 1.0 / down)
		fir = scipy.signal.firwin(n_taps, cutoff, window=('kaiser', 5.0)) * up
		sub_len = (n_taps + up - 1) // up
		padded = numpy.zeros(sub_len * up, dtype=numpy.float64)
		padded[:n_taps] = fir
		# polyphase[p] = h[p], h[p+up], h[p+2*up], ... (the p-th branch)
		# Flip each branch so dot(coeffs, seg) performs convolution not correlation.
		polyphase_branches = padded.reshape(sub_len, up).T
		state[fir_key] = polyphase_branches[:, ::-1].copy()
		state[buf_key] = numpy.zeros(sub_len, dtype=signal.dtype)
		state[phase_key] = 0
		state[offset_key] = 0

	polyphase: numpy.typing.NDArray = state[fir_key]
	sub_len = polyphase.shape[1]
	buf: numpy.typing.NDArray = state[buf_key]
	phase: int = state[phase_key]
	offset: int = state[offset_key]

	x = numpy.concatenate([buf, signal])
	n_in = len(signal)

	n_out_estimate = int(numpy.ceil((n_in - offset) * up / down))
	out = numpy.empty(max(0, n_out_estimate + up), dtype=signal.dtype)

	i = sub_len + offset
	k = 0
	while i < len(x):
		coeffs = polyphase[phase]
		start = i - sub_len
		seg = x[start:i]
		if numpy.iscomplexobj(seg):
			out[k] = numpy.dot(coeffs, seg.real) + 1j * numpy.dot(coeffs, seg.imag)
		else:
			out[k] = numpy.dot(coeffs, seg)
		k += 1
		phase += down
		i += phase // up
		phase = phase % up

	state[buf_key] = x[-sub_len:].copy()
	state[phase_key] = phase
	state[offset_key] = i - len(x)

	result = out[:k]
	if signal.dtype == numpy.float32 or signal.dtype == numpy.float64:
		return result.astype(numpy.float32), state
	return result.astype(numpy.complex64), state


def _decimate_common (
	signal: numpy.typing.NDArray,
	sample_rate: float,
	target_rate: int,
	state: dict,
	state_prefix: str = 'decimate_'
) -> tuple[numpy.typing.NDArray, dict]:

	"""
	Common implementation for signal decimation (audio or IQ).
	Preserves input dtype (float or complex).

	Args:
		signal: Input signal
		sample_rate: Input sample rate in Hz
		target_rate: Target sample rate in Hz
		state: State dict for continuity
		state_prefix: Prefix for state keys to avoid collisions in chained usage

	Returns:
		Tuple of (decimated_signal, updated_state)
	"""

	sr = int(round(sample_rate))
	ar = int(target_rate)
	
	if sr <= 0 or ar <= 0:
		return signal, state

	if sr == ar:
		return signal, state

	# Check if we can use simple integer decimation (much faster)
	# If the ratio isn't an integer, we need rational resampling instead
	decimation_factor = sr // ar

	if sr % ar != 0:
		g = math.gcd(sr, ar)
		up = ar // g
		down = sr // g

		if max(up, down) > 1_000:
			ratio = (sr, ar)
			if ratio not in _RESAMPLE_WARNED_RATIOS:
				logger.warning(
					f"Non-integer resample ratio: {sr} -> {ar} Hz "
					f"(up={up}, down={down}, ~{2 * _RESAMPLE_HALF_LEN * max(up, down) + 1} tap filter).  "
					"Prefer a sample_rate that is an integer multiple of target rate "
					"for lower CPU."
				)
				_RESAMPLE_WARNED_RATIOS.add(ratio)

		if max(up, down) > _RESAMPLE_MAX_FACTOR:
			logger.error(
				f"Refusing pathological resample {sr} -> {ar} Hz "
				f"(up={up}, down={down}, max={max(up, down)} > {_RESAMPLE_MAX_FACTOR}).  "
				f"This is almost certainly a configuration bug — pick a target rate "
				f"with a small-gcd ratio to the source rate."
			)
			return numpy.array([], dtype=signal.dtype), state

		return _streaming_rational_resample(signal, up, down, state, state_prefix)

	if decimation_factor <= 1:
		return signal, state

	# Anti-aliasing filter: remove frequencies above new Nyquist rate
	sos_key = f'{state_prefix}sos'
	zi_key = f'{state_prefix}zi'
	phase_key = f'{state_prefix}phase'

	if sos_key not in state:
		nyq_freq = target_rate / 2
		cutoff = nyq_freq * 0.8  # 80% of Nyquist leaves transition band
		state[sos_key] = scipy.signal.butter(8, cutoff, fs=sample_rate, output='sos')

	if zi_key not in state:
		# Use float64/complex128 for filter state to prevent rounding drift
		# in long-running sessions.  The output is downcast after filtering.
		zi_shape = (state[sos_key].shape[0], 2)
		if numpy.issubdtype(signal.dtype, numpy.complexfloating):
			state[zi_key] = numpy.zeros(zi_shape, dtype=numpy.complex128)
		else:
			state[zi_key] = numpy.zeros(zi_shape, dtype=numpy.float64)

	if phase_key not in state:
		state[phase_key] = 0

	# Apply anti-aliasing filter with state preservation
	filtered, state[zi_key] = scipy.signal.sosfilt(
		state[sos_key], signal, zi=state[zi_key]
	)

	# Downsample by keeping every Nth sample
	start_idx = state[phase_key]
	decimated = filtered[start_idx::decimation_factor]

	if signal.dtype == numpy.complex64 or signal.dtype == numpy.complex128:
		# Use copy=False to avoid extra alloc if possible, ensure complex64
		decimated = decimated.astype(numpy.complex64, copy=False)
	else:
		decimated = decimated.astype(numpy.float32, copy=False)

	# Calculate where to start next block to maintain phase continuity
	remaining = (len(filtered) - start_idx) % decimation_factor
	state[phase_key] = (decimation_factor - remaining) % decimation_factor

	return decimated, state


def decimate_audio (
	signal: numpy.typing.NDArray,
	sample_rate: float,
	audio_sample_rate: int,
	state: dict
) -> tuple[numpy.typing.NDArray[numpy.float32], dict]:

	"""
	Decimate signal to audio rate (float32).
	Wrapper around _decimate_common.
	"""
	
	# Use default 'decimate_' prefix for backward compatibility with existing state
	out, state = _decimate_common(signal, sample_rate, audio_sample_rate, state, state_prefix='decimate_')
	
	# Ensure float output (magnitude if complex, but usually audio is real)
	if numpy.iscomplexobj(out):
		out = numpy.abs(out)
		
	return out.astype(numpy.float32), state


def decimate_iq (
	signal: numpy.typing.NDArray[numpy.complex64],
	sample_rate: float,
	target_rate: int,
	state: dict,
	state_prefix: str = 'iq_decimate_'
) -> tuple[numpy.typing.NDArray[numpy.complex64], dict]:

	"""
	Decimate IQ signal (complex64).
	Wrapper around _decimate_common.
	"""

	return _decimate_common(signal, sample_rate, target_rate, state, state_prefix=state_prefix)


def apply_fade (
	audio: numpy.typing.NDArray[numpy.float32],
	sample_rate: int,
	fade_in_ms: float | None,
	fade_out_ms: float | None,
	pad_in_samples: int = 0,
	pad_out_samples: int = 0,
) -> numpy.typing.NDArray[numpy.float32]:

	"""
	Apply smooth fade-in/out to prevent clicks at recording boundaries.

	Modifies the input array in-place and returns it.

	Uses half-cosine S-curves for smooth transitions.  When ``pad_in_samples``
	or ``pad_out_samples`` are provided the fade is constrained to the padding
	region so that actual signal content (including attack transients) is never
	attenuated.  If the padding region is smaller than the requested fade
	duration the fade is shortened to fit the padding.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		fade_in_ms: Fade-in duration in milliseconds (None to disable)
		fade_out_ms: Fade-out duration in milliseconds (None to disable)
		pad_in_samples: Number of padding samples prepended before signal onset.
			When >0, fade-in is limited to this region.
		pad_out_samples: Number of padding samples appended after signal end.
			When >0, fade-out is limited to this region.

	Returns:
		Audio signal with fades applied
	"""

	if fade_in_ms is None and fade_out_ms is None:
		return audio

	n_samples = len(audio)
	if n_samples == 0:
		return audio

	# Convert fade durations from milliseconds to sample counts
	fade_in_len = int(sample_rate * (fade_in_ms / 1000.0)) if fade_in_ms is not None else 0
	fade_out_len = int(sample_rate * (fade_out_ms / 1000.0)) if fade_out_ms is not None else 0

	# Constrain fades to padding regions when specified (preserves signal content)
	if pad_in_samples > 0:
		fade_in_len = min(fade_in_len, pad_in_samples)
	if pad_out_samples > 0:
		fade_out_len = min(fade_out_len, pad_out_samples)

	# Ensure fades don't exceed audio length (fades can't overlap)
	fade_in_len = min(fade_in_len, n_samples)
	fade_out_len = min(fade_out_len, n_samples - fade_in_len)

	if fade_in_len > 0:
		# Half-cosine S-curve: (1 - cos(πt)) / 2
		# Zero first and second derivatives at endpoints — smoother than smoothstep
		ramp_in = (1.0 - numpy.cos(numpy.linspace(0.0, numpy.pi, fade_in_len, dtype=numpy.float64))) / 2.0
		audio[:fade_in_len] *= ramp_in.astype(audio.dtype)

	if fade_out_len > 0:
		# Same half-cosine, inverted (1.0 -> 0.0)
		ramp_out = (1.0 + numpy.cos(numpy.linspace(0.0, numpy.pi, fade_out_len, dtype=numpy.float64))) / 2.0
		audio[-fade_out_len:] *= ramp_out.astype(audio.dtype)

	return audio
