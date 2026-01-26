"""
Audio processing and filtering functions.

Provides sample rate conversion (decimation and resampling), audio fades,
and other signal conditioning needed for clean audio output from demodulated RF.

Key functions:
- decimate_audio: Convert from RF sample rate to audio sample rate with anti-aliasing
- apply_fade: Apply smooth fade-in/fade-out to prevent clicks at recording boundaries
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

	Decimation reduces the sample rate by an integer factor (e.g., 2 MHz -> 48 kHz).
	This requires anti-aliasing filtering before downsampling to prevent aliasing
	(high frequencies folding back into the audible range).

	When the ratio is not an integer, uses rational resampling (upsample then downsample)
	which is more expensive but handles arbitrary rate conversions.

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

	# Check if we can use simple integer decimation (much faster)
	# If the ratio isn't an integer, we need rational resampling instead
	decimation_factor = sr // ar

	if sr % ar != 0:
		# Warn once: non-integer ratios are much more CPU-intensive
		if not state.get('resample_warned'):
			logger.warning(
				f"Non-integer resample ratio: {sr} -> {ar} Hz. "
				"Prefer a sample_rate that is an integer multiple of audio_sample_rate for lower CPU."
			)
			state['resample_warned'] = True

		# Rational resampling: upsample by 'up', then downsample by 'down'
		# Example: 2 MHz -> 48 kHz is not integer-divisible
		# GCD(2000000, 48000) = 16000, so up=3, down=125
		# This means upsample by 3, then downsample by 125
		# Keep overlap from previous block to avoid boundary artifacts from FIR filter
		g = math.gcd(sr, ar)
		up = ar // g
		down = sr // g
		prev = state.get('resample_prev', numpy.array([], dtype=signal.dtype))
		last_ratio = state.get('resample_ratio')
		if last_ratio != (up, down):
			prev = numpy.array([], dtype=signal.dtype)
			state['resample_ratio'] = (up, down)

		current = signal
		# Prepend previous block's tail to provide context for FIR filter
		if prev.size > 0:
			signal = numpy.concatenate([prev, current])
		# Perform rational resampling using polyphase filters
		resampled = scipy.signal.resample_poly(signal, up, down).astype(numpy.float32)
		if prev.size > 0:
			# Drop output samples that came from the prepended overlap
			# We only want output corresponding to the current block
			out_prev = int(numpy.ceil(prev.size * up / down))
			resampled = resampled[out_prev:]

		# Save tail of current block for next iteration
		# Tail length must cover the FIR filter length to avoid clicks at boundaries
		taps_len = state.get('resample_taps_len')
		if taps_len is None:
			# Polyphase FIR length is approximately 10 * max(up, down)
			taps_len = 10 * max(up, down) + 1
			state['resample_taps_len'] = taps_len
		overlap_len = min(current.size, int(taps_len))
		state['resample_prev'] = current[-overlap_len:].copy()
		return resampled, state

	if decimation_factor <= 1:
		return signal.astype(numpy.float32), state

	# Anti-aliasing filter: remove frequencies above new Nyquist rate
	# Without this, high frequencies would "fold back" into audible range (aliasing)
	# Use 8th-order Butterworth for sharp cutoff with minimal passband ripple
	if 'decimate_sos' not in state:
		nyq_freq = audio_sample_rate / 2
		cutoff = nyq_freq * 0.8  # 80% of Nyquist leaves transition band
		state['decimate_sos'] = scipy.signal.butter(8, cutoff, fs=sample_rate, output='sos')

	if 'decimate_zi' not in state:
		state['decimate_zi'] = scipy.signal.sosfilt_zi(state['decimate_sos']) * 0.0

	if 'decimate_phase' not in state:
		state['decimate_phase'] = 0

	# Apply anti-aliasing filter with state preservation for continuous operation
	filtered, state['decimate_zi'] = scipy.signal.sosfilt(
		state['decimate_sos'], signal, zi=state['decimate_zi']
	)

	# Downsample by keeping every Nth sample
	# Track phase across blocks: if we start at sample 2, next block starts at (2+len)%N
	# This ensures no samples are dropped or duplicated at block boundaries
	start_idx = state['decimate_phase']
	audio_samples = filtered[start_idx::decimation_factor].astype(numpy.float32)

	# Calculate where to start next block to maintain phase continuity
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
	Apply smooth fade-in/out to prevent clicks at recording boundaries.

	Fades prevent audible clicks/pops when recordings start or stop abruptly.
	Uses smoothstep curves (S-shaped) instead of linear for more natural sound.
	If either duration is None, fading is disabled.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		fade_in_ms: Fade-in duration in milliseconds (None to disable)
		fade_out_ms: Fade-out duration in milliseconds (None to disable)

	Returns:
		Audio signal with fades applied
	"""

	if fade_in_ms is None and fade_out_ms is None:
		return audio

	n_samples = len(audio)
	if n_samples == 0:
		return audio

	# Convert fade durations from milliseconds to sample counts
	# Handle None values by using 0 length (no fade)
	fade_in_len = int(sample_rate * (fade_in_ms / 1000.0)) if fade_in_ms is not None else 0
	fade_out_len = int(sample_rate * (fade_out_ms / 1000.0)) if fade_out_ms is not None else 0

	# Ensure fades don't exceed audio length (fades can't overlap)
	fade_in_len = min(fade_in_len, n_samples)
	fade_out_len = min(fade_out_len, n_samples - fade_in_len)

	if fade_in_len > 0:
		# Smoothstep curve: f(t) = t^2 * (3 - 2t)
		# This creates an S-shaped curve that's smoother than linear
		# Starts slow, accelerates in middle, slows at end
		ramp_in = numpy.linspace(0.0, 1.0, fade_in_len, endpoint=True, dtype=audio.dtype)
		ramp_in = ramp_in * ramp_in * (3.0 - 2.0 * ramp_in)
		audio[:fade_in_len] *= ramp_in

	if fade_out_len > 0:
		# Same smoothstep curve, but inverted (1.0 -> 0.0)
		ramp_out = numpy.linspace(0.0, 1.0, fade_out_len, endpoint=True, dtype=audio.dtype)
		ramp_out = ramp_out * ramp_out * (3.0 - 2.0 * ramp_out)
		ramp_out = 1.0 - ramp_out
		audio[-fade_out_len:] *= ramp_out

	return audio
