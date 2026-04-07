"""
Noise reduction algorithms for improving audio quality.

Provides two noise reduction approaches:
1. apply_noisereduce: Uses the noisereduce library (spectral subtraction with noise profile)
2. apply_spectral_subtraction: Custom spectral subtraction implementation

Both methods estimate the noise floor from quiet portions of the audio,
then subtract the noise spectrum from the signal spectrum. This reduces
background hiss and static while preserving speech/signals.

Note: Aggressive noise reduction can introduce "musical noise" artifacts
(random burbling sounds). The algorithms use smoothing to minimize this.
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
