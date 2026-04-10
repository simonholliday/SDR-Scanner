"""
Noise reduction algorithms for improving audio quality.

Provides three noise reduction approaches:
1. apply_noisereduce: Uses the noisereduce library (spectral subtraction with noise profile)
2. apply_spectral_subtraction: Custom spectral subtraction implementation
3. apply_dynamics_curve: Per-sample dual-region expander (downward + upward)

The first two methods estimate the noise floor from quiet portions of the audio,
then subtract the noise spectrum from the signal spectrum. This reduces
background hiss and static while preserving speech/signals.

apply_dynamics_curve is a memoryless waveshaper that operates in dBFS:
quiet samples below a threshold are softly cut towards a silence floor
(downward expansion), and loud samples above the threshold are gently
boosted (upward expansion).  Both regions widen the dynamic range, so the
function as a whole is a dual-region expander.

Note: Aggressive noise reduction can introduce "musical noise" artifacts
(random burbling sounds). The spectral algorithms use smoothing to
minimize this.  apply_dynamics_curve is sample-level and can introduce
mild harmonic distortion on signals near the threshold; the curves are
shaped to keep this benign at sensible parameter values.
"""

from __future__ import annotations

import numpy
import numpy.lib.stride_tricks
import scipy.signal


def _frame_rms (audio: numpy.ndarray, frame_len: int, hop: int) -> numpy.ndarray:
	"""
	Calculate RMS (Root Mean Square) energy for overlapping frames.

	RMS measures the "loudness" of each frame, helping identify quiet (noise-only)
	vs active (signal+noise) portions of the audio.

	Uses stride tricks for efficient zero-copy windowing instead of loops.
	"""

	if audio.size < frame_len:
		return numpy.array([numpy.sqrt(numpy.mean(audio * audio))], dtype=numpy.float32)

	n_frames = 1 + (audio.size - frame_len) // hop

	# Stride tricks: create a 2D view of overlapping frames without copying data
	# EACH row is one frame of audio, frames overlap by (frame_len - hop) samples
	# This is much faster than a Python loop creating individual frame copies
	audio_contig = numpy.ascontiguousarray(audio, dtype=numpy.float32)
	frame_shape = (n_frames, frame_len)
	frame_strides = (audio_contig.strides[0] * hop, audio_contig.strides[0])
	frames = numpy.lib.stride_tricks.as_strided(audio_contig, shape=frame_shape, strides=frame_strides)

	# RMS = sqrt(mean(signal²))
	# Vectorized: compute all frame RMS values in one operation
	rms = numpy.sqrt(numpy.mean(frames * frames, axis=1)).astype(numpy.float32)

	return rms


def _noise_clip_from_percentile (
	audio: numpy.ndarray,
	sample_rate: int,
	frame_ms: float = 20.0,
	percentile: float = 20.0
) -> numpy.ndarray:
	"""
	Extract a noise profile from the quietest portions of the audio.

	Noise reduction requires a "noise sample" showing what the background noise
	sounds like. This function automatically finds the quietest frames
	(assumed to be noise-only, no speech/signal) and concatenates them.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		frame_ms: Frame duration in milliseconds for RMS calculation
		percentile: Percentile threshold for "quiet" frames (20 = quietest 20%)

	Returns:
		Concatenated audio from the quietest frames (noise profile)
	"""
	# Convert frame duration from milliseconds to samples
	frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
	hop = max(1, frame_len // 2)

	# Calculate RMS energy for each frame
	rms = _frame_rms(audio, frame_len, hop)
	if rms.size == 0:
		return audio[:0]

	# Find the RMS level that represents the Nth percentile (e.g., 20th percentile)
	# Frames quieter than this are assumed to be noise-only
	threshold = numpy.percentile(rms, percentile)

	# Collect all frames below the threshold
	noise_frames = []
	for i, value in enumerate(rms):
		if value <= threshold:
			start = i * hop
			noise_frames.append(audio[start:start + frame_len])

	# Fallback: if no frames are quiet enough, use first 0.5 seconds
	if not noise_frames:
		return audio[: min(audio.size, sample_rate // 2)]

	return numpy.concatenate(noise_frames)


def apply_noisereduce (
	audio: numpy.ndarray,
	sample_rate: int,
	frame_ms: float = 20.0,
	percentile: float = 20.0,
	prop_decrease: float = 0.8,
	freq_mask_smooth_hz: float = 300.0,
	time_mask_smooth_ms: float = 100.0,
	n_fft: int = 1024
) -> tuple[numpy.ndarray, numpy.ndarray | None]:

	"""
	Apply noise reduction using the noisereduce library.

	The noisereduce library uses spectral subtraction: it learns the noise spectrum
	from quiet portions, then subtracts that spectrum from the entire signal.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		frame_ms: Frame duration for noise profile estimation
		percentile: Percentile for identifying quiet frames (noise profile)
		prop_decrease: How much noise to subtract (0.0-1.0, higher = more aggressive)
		freq_mask_smooth_hz: Frequency smoothing to reduce musical noise
		time_mask_smooth_ms: Time smoothing to reduce musical noise
		n_fft: FFT size for spectral analysis

	Returns:
		Tuple of (noise_reduced_audio, None)
	"""

	# Skip processing for very short audio
	if audio.size == 0:
		return audio.astype(numpy.float32, copy=False), None
	if audio.size < n_fft:
		return audio.astype(numpy.float32, copy=False), None

	# Check if noisereduce library is available (optional dependency)
	try:
		import noisereduce
	except ImportError as exc:
		raise RuntimeError("noisereduce is required for noise reduction") from exc

	# Extract noise profile from quietest portions of the audio
	noise_clip = _noise_clip_from_percentile(audio, sample_rate, frame_ms, percentile)
	if noise_clip.size == 0:
		noise_clip = audio[: min(audio.size, sample_rate // 2)]

	# Apply noise reduction
	# stationary=False: allows noise floor to vary over time (better for real-world audio)
	reduced = noisereduce.reduce_noise(
		y=audio,
		sr=sample_rate,
		y_noise=noise_clip,
		stationary=False,
		prop_decrease=prop_decrease,
		freq_mask_smooth_hz=freq_mask_smooth_hz,
		time_mask_smooth_ms=time_mask_smooth_ms,
		n_fft=n_fft
	)
	return reduced.astype(numpy.float32, copy=False), None


def apply_spectral_subtraction (
	audio: numpy.ndarray,
	sample_rate: int,
	oversub: float = 0.7,
	floor: float = 0.06,
	noise_mag: numpy.ndarray | None = None,
	adaptive_noise_estimation: bool = False
) -> tuple[numpy.ndarray, numpy.ndarray]:

	"""
	Custom spectral subtraction implementation with gain smoothing.

	Spectral subtraction works in the frequency domain:
	1. Convert audio to spectrogram (STFT: Short-Time Fourier Transform)
	2. Estimate noise spectrum from quiet frames (or use provided noise_mag)
	3. Subtract noise spectrum from signal spectrum
	4. Apply smoothing to reduce "musical noise" artifacts
	5. Convert back to time domain (ISTFT)

	Two noise-frame selection strategies are available:
	- Default (``adaptive_noise_estimation=False``): uses the 20th percentile
	  of frame energy as the threshold.  Works well when the audio contains
	  a mix of speech and silence.
	- Adaptive (``adaptive_noise_estimation=True``): selects frames within
	  3 dB of the minimum frame energy.  More reliable when the audio chunk
	  is mostly speech/signal and there are few truly quiet frames.

	Args:
		audio: Input audio signal
		sample_rate: Audio sample rate in Hz
		oversub: Noise oversubtraction factor (>1 = more aggressive)
		floor: Minimum gain floor to prevent complete zeroing
		noise_mag: Pre-computed noise magnitude spectrum (optional)
		adaptive_noise_estimation: When True, use min-energy-based frame
			selection instead of the fixed 20th percentile heuristic.

	Returns:
		Tuple of (denoised_audio, noise_mag) where noise_mag can be reused
	"""

	if audio.size == 0:
		return audio.astype(numpy.float32, copy=False), (noise_mag if noise_mag is not None else numpy.array([], dtype=numpy.float32))

	# Choose frame length: ~20ms is typical for speech processing
	# Round up to nearest power of 2 for efficient FFT
	frame_len = max(256, int(sample_rate * 0.02))
	n_fft = 1 << (frame_len - 1).bit_length()

	if audio.size < n_fft:
		return audio.astype(numpy.float32, copy=False), (noise_mag if noise_mag is not None else numpy.array([], dtype=numpy.float32))

	# STFT: convert audio to time-frequency representation (spectrogram)
	hop = n_fft // 2
	_, _, zxx = scipy.signal.stft(
		audio,
		fs=sample_rate,
		window="hann",
		nperseg=n_fft,
		noverlap=n_fft - hop,
		boundary="zeros",
		padded=True
	)

	# Separate magnitude and phase
	magnitude = numpy.abs(zxx)
	phase = zxx / numpy.maximum(magnitude, 1e-10)

	# Use provided noise profile or estimate from quietest frames in current block
	if noise_mag is None or noise_mag.shape[0] != magnitude.shape[0]:

		# Calculate total energy per frame (sum across all frequencies)
		frame_energy = numpy.mean(magnitude * magnitude, axis=0)

		if adaptive_noise_estimation and frame_energy.size > 1:
			# Select frames within 3 dB of the minimum energy as noise.
			# More reliable than the percentile heuristic when most frames
			# contain signal rather than silence.
			min_energy = numpy.min(frame_energy)
			# 3 dB above the quietest frame — captures noise variation
			# without including speech frames.
			energy_threshold = min_energy * 2.0  # +3 dB in linear power
			noise_frames = magnitude[:, frame_energy <= energy_threshold]
			# Fall back to percentile if that selects too few frames
			if noise_frames.shape[1] < 2:
				energy_threshold = numpy.percentile(frame_energy, 20.0)
				noise_frames = magnitude[:, frame_energy <= energy_threshold]
		else:
			energy_threshold = numpy.percentile(frame_energy, 20.0)
			noise_frames = magnitude[:, frame_energy <= energy_threshold]

		# Average the quiet frames to get noise spectrum estimate
		if noise_frames.size == 0:
			noise_mag = numpy.median(magnitude, axis=1, keepdims=True)
		else:
			noise_mag = numpy.mean(noise_frames, axis=1, keepdims=True)

	# Spectral subtraction: magnitude_clean = magnitude_signal - α*magnitude_noise
	# oversub > 1.0 means we subtract more than the estimated noise (more aggressive)
	# floor prevents complete zeroing (some residual noise sounds more natural)
	subtracted = numpy.maximum(magnitude - noise_mag * oversub, noise_mag * floor)

	# Convert to gain mask: how much to attenuate each time-frequency bin
	gain = subtracted / numpy.maximum(magnitude, 1e-10)

	# Smooth the gain mask to reduce musical noise (random variations)
	smooth_kernel = numpy.ones((3, 5), dtype=numpy.float32)
	smooth_kernel /= smooth_kernel.sum()
	gain = scipy.signal.convolve2d(gain, smooth_kernel, mode="same", boundary="symm")
	gain = numpy.clip(gain, 0.0, 1.0)

	# Apply gain mask and restore phase
	zxx_denoised = magnitude * gain * phase

	# ISTFT: convert spectrogram back to time-domain audio
	_, denoised = scipy.signal.istft(
		zxx_denoised,
		fs=sample_rate,
		window="hann",
		nperseg=n_fft,
		noverlap=n_fft - hop,
		input_onesided=True,
		boundary=True
	)

	# Trim to original length (STFT/ISTFT may add padding)
	audio_out = denoised[: audio.size].astype(numpy.float32, copy=False)

	return audio_out, noise_mag


def apply_dynamics_curve (
	samples: numpy.ndarray,
	threshold_dbfs: float,
	cut_db: float = 6.0,
	boost_db: float = 1.5,
	floor_dbfs: float = -60.0,
	cut_curve: float = 0.5,
	boost_curve: float = 0.5,
) -> numpy.ndarray:

	"""
	Apply a dual-region expander transfer curve to an audio signal.

	This is a memoryless, sample-level waveshaper that maps each input
	sample's dBFS level to an output dBFS level via two smooth curves
	joined at ``threshold_dbfs``:

	- **Below the threshold (cut region — downward expansion).**  Samples
	  whose magnitude falls between ``floor_dbfs`` and ``threshold_dbfs``
	  are progressively reduced.  The reduction follows a smoothstep
	  S-curve in dB space, reaching exactly ``cut_db`` at the curve's
	  midpoint and ``2 * cut_db`` at the floor.  The curve has zero slope
	  at both endpoints, so there is no audible kink at the threshold or
	  the floor.  Below ``floor_dbfs`` the output is hard-zeroed (true
	  digital silence).

	- **Above the threshold (boost region — upward expansion).**  Samples
	  whose magnitude falls between ``threshold_dbfs`` and 0 dBFS are
	  gently boosted by up to ``boost_db`` at the curve's midpoint, with
	  zero boost at both ends.  The shape is ``sin²(π·t)`` so the curve
	  is tangent to unity at both the threshold and full scale, and there
	  is no clipping introduced *for any sensible configuration* (see the
	  defensive clamp below).

	The two regions together widen the overall dynamic range — quiet
	noise gets quieter, loud voice gets louder — making this useful as a
	noise-reduction polish step on top of spectral subtraction.

	The function is memoryless: it has no envelope follower, no attack
	or release, and no inter-block state.  Because the gain is computed
	from each sample's instantaneous magnitude, signals that cross the
	threshold within a waveform cycle pick up a small amount of harmonic
	distortion.  At the default ``cut_db`` and ``boost_db`` values this
	is benign; aggressive settings will produce a perceptible "edge" on
	the loudest syllables.

	**Defensive clamp.**  After the curves are applied the output is
	clamped to ``[-1, 1]`` as belt-and-braces protection against any
	configuration that would otherwise drive the boost region above 0
	dBFS.  With sensible defaults this clamp never engages, but it
	prevents speaker damage if the user picks adventurous parameter
	combinations.

	Args:
		samples: Input audio samples, float, normalised to [-1, 1].
		threshold_dbfs: Dividing line between the cut and boost regions.
			Samples at exactly this level pass through unchanged.
			Typical values: -20 to -35 dBFS for ATC airband audio.
		cut_db: Gain reduction at the midpoint of the cut S-curve.  The
			maximum reduction (at the floor) is twice this value.  Set to
			0.0 to disable the cut region entirely.
		boost_db: Maximum gain boost at the peak of the boost hump.  Set
			to 0.0 to disable the boost region entirely.
		floor_dbfs: Lower threshold.  Samples whose magnitude is at or
			below this level are output as exactly 0.0.
		cut_curve: Position of the steepest gradient within the cut
			S-curve, in (0, 1).  0.5 = symmetric (steepest at the
			midpoint).  Lower values shift the steepest part toward the
			threshold; higher values shift it toward the floor.
		boost_curve: Same concept for the boost hump.  0.5 = symmetric.
			Lower values shift the peak toward the threshold; higher
			values shift it toward 0 dBFS.

	Returns:
		A new numpy array of the same shape and dtype as the input,
		containing the processed samples.
	"""

	# Validate the level parameters defensively — the calling code is
	# expected to validate via Pydantic, but the function should also be
	# robust when called directly (e.g., from tests).

	if floor_dbfs >= threshold_dbfs:
		raise ValueError(f"floor_dbfs ({floor_dbfs}) must be strictly below threshold_dbfs ({threshold_dbfs})")

	if threshold_dbfs >= 0:
		raise ValueError(f"threshold_dbfs ({threshold_dbfs}) must be below 0 dBFS")

	if cut_db < 0 or boost_db < 0:
		raise ValueError(f"cut_db and boost_db must be non-negative (got cut_db={cut_db}, boost_db={boost_db})")

	if not (0.0 <= cut_curve <= 1.0) or not (0.0 <= boost_curve <= 1.0):
		raise ValueError(f"cut_curve and boost_curve must be in [0, 1] (got cut_curve={cut_curve}, boost_curve={boost_curve})")

	# Empty / pathological inputs short-circuit
	if samples.size == 0:
		return samples.astype(numpy.float32, copy=True)

	original_dtype = samples.dtype

	# Work in float32 for the whole pipeline; preserve the input dtype on output
	x = samples.astype(numpy.float32, copy=False)

	# Magnitudes and dBFS levels.  log10(0) would be -inf; we mask zeros
	# explicitly so they end up in the silence region rather than producing NaNs.
	mag = numpy.abs(x)
	with numpy.errstate(divide='ignore'):
		mag_db = 20.0 * numpy.log10(numpy.maximum(mag, 1e-30))

	# Region masks
	silence_mask = mag_db <= floor_dbfs
	cut_mask     = (mag_db > floor_dbfs) & (mag_db < threshold_dbfs)
	boost_mask   = (mag_db >= threshold_dbfs) & (mag_db < 0.0)
	# (anything at or above 0 dBFS passes through unchanged)

	# Start with passthrough — every region overrides as needed below
	new_db = mag_db.copy()

	# --- Cut region: downward-expansion S-curve in dB space ---
	# t goes from 0 at the threshold to 1 at the floor
	# (so larger t means deeper into the noise zone)

	if cut_db > 0.0 and numpy.any(cut_mask):

		t = (threshold_dbfs - mag_db[cut_mask]) / (threshold_dbfs - floor_dbfs)
		t = numpy.clip(t, 0.0, 1.0)

		# Skew the position of the steepest gradient via t^p.
		# For p < 1, small t maps "up" (e.g. 0.05^0.5 ≈ 0.22), so the
		# smoothstep accumulates quickly near t = 0 — i.e., the steepest
		# part of the dB-space curve sits near the threshold.  For p > 1,
		# the steepest part sits near the floor.
		# cut_curve = 0.5 → exponent = 1   (symmetric, smoothstep peak at t = 0.5)
		# cut_curve < 0.5 → exponent < 1  (steepest near threshold, t ≈ 0)
		# cut_curve > 0.5 → exponent > 1  (steepest near floor,     t ≈ 1)
		exponent = float(numpy.exp((cut_curve - 0.5) * 4.0))
		t_skewed = t ** exponent

		# Standard smoothstep: 3t² - 2t³.  Zero slope at both endpoints,
		# value 0.5 at t = 0.5.  Multiplying by (2 * cut_db) makes the
		# midpoint reduction exactly cut_db, matching the spec.
		smooth = t_skewed * t_skewed * (3.0 - 2.0 * t_skewed)
		reduction_db = 2.0 * cut_db * smooth

		new_db[cut_mask] = mag_db[cut_mask] - reduction_db

	# --- Boost region: upward-expansion hump in dB space ---
	# t goes from 0 at the threshold to 1 at 0 dBFS

	if boost_db > 0.0 and numpy.any(boost_mask):

		t = (mag_db[boost_mask] - threshold_dbfs) / (0.0 - threshold_dbfs)
		t = numpy.clip(t, 0.0, 1.0)

		# Same skew formulation as the cut region: small exponent shifts
		# the sin² peak toward the threshold (t ≈ 0); large exponent
		# shifts it toward 0 dBFS (t ≈ 1).
		exponent = float(numpy.exp((boost_curve - 0.5) * 4.0))
		t_skewed = t ** exponent

		# sin²(π·t) hump: peaks at boost_db when t_skewed = 0.5, zero at
		# both endpoints, smooth tangent to passthrough at threshold and
		# at 0 dBFS.
		hump = numpy.sin(numpy.pi * t_skewed) ** 2
		boost = boost_db * hump

		new_db[boost_mask] = mag_db[boost_mask] + boost

	# Convert dB back to linear magnitude
	new_mag = numpy.power(10.0, new_db / 20.0)

	# Defensive clamp: protect speakers from any configuration that would
	# otherwise push the boost region above 0 dBFS.  Costs essentially
	# nothing and never engages with sensible parameters.
	new_mag = numpy.minimum(new_mag, 1.0)

	# Restore the sign of the original samples
	output = numpy.sign(x) * new_mag

	# Hard zero below the floor (true digital silence)
	output[silence_mask] = 0.0

	return output.astype(original_dtype, copy=False)
