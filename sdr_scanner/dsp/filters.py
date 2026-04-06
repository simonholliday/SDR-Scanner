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
		# Warn once per process for each unique ratio to avoid log spam on every channel activation
		ratio = (sr, ar)
		if ratio not in _RESAMPLE_WARNED_RATIOS:
			logger.warning(
				f"Non-integer resample ratio: {sr} -> {ar} Hz. "
				"Prefer a sample_rate that is an integer multiple of target rate for lower CPU."
			)
			_RESAMPLE_WARNED_RATIOS.add(ratio)

		# Rational resampling: upsample by 'up', then downsample by 'down'
		# Example: 2 MHz -> 48 kHz is not integer-divisible
		# GCD(2000000, 48000) = 16000, so up=3, down=125
		g = math.gcd(sr, ar)
		up = ar // g
		down = sr // g
		
		prev_key = f'{state_prefix}resample_prev'
		ratio_key = f'{state_prefix}resample_ratio'
		taps_key = f'{state_prefix}resample_taps_len'
		
		prev = state.get(prev_key, numpy.array([], dtype=signal.dtype))
		last_ratio = state.get(ratio_key)
		
		if last_ratio != (up, down):
			prev = numpy.array([], dtype=signal.dtype)
			state[ratio_key] = (up, down)

		current = signal
		# Prepend previous block's tail to provide context for FIR filter
		if prev.size > 0:
			signal = numpy.concatenate([prev, current])
			
		# Perform rational resampling using polyphase filters
		# resample_poly handles both float and complex correctly
		resampled = scipy.signal.resample_poly(signal, up, down)
		
		# Ensure output matches input type (scipy allows float->float, complex->complex)
		if signal.dtype == numpy.float32 or signal.dtype == numpy.float64:
			resampled = resampled.astype(numpy.float32)
		else:
			resampled = resampled.astype(numpy.complex64)

		if prev.size > 0:
			# Drop output samples that came from the prepended overlap
			out_prev = int(numpy.ceil(prev.size * up / down))
			resampled = resampled[out_prev:]

		# Save tail of current block for next iteration
		taps_len = state.get(taps_key)
		if taps_len is None:
			# Polyphase FIR length is approximately 10 * max(up, down)
			taps_len = 10 * max(up, down) + 1
			state[taps_key] = taps_len
			
		overlap_len = min(current.size, int(taps_len))
		state[prev_key] = current[-overlap_len:].copy()
		return resampled, state

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
