"""
Noise reduction helpers for demodulated audio.
"""

from __future__ import annotations

import numpy
import numpy.lib.stride_tricks
import scipy.signal


def _frame_rms (audio: numpy.ndarray, frame_len: int, hop: int) -> numpy.ndarray:

	"""Compute RMS values for overlapping frames using vectorized operations."""

	if audio.size < frame_len:
		return numpy.array([numpy.sqrt(numpy.mean(audio * audio))], dtype=numpy.float32)

	n_frames = 1 + (audio.size - frame_len) // hop

	# Use numpy stride_tricks for efficient sliding window - no Python loops
	shape = (n_frames, frame_len)
	strides = (audio.strides[0] * hop, audio.strides[0])
	frames = numpy.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides, writeable=False)

	# Vectorized RMS computation across all frames at once
	return numpy.sqrt(numpy.mean(frames * frames, axis=1)).astype(numpy.float32)


def _noise_clip_from_percentile (
	audio: numpy.ndarray,
	sample_rate: int,
	frame_ms: float = 20.0,
	percentile: float = 20.0
) -> numpy.ndarray:
	frame_len = max(1, int(sample_rate * (frame_ms / 1000.0)))
	hop = max(1, frame_len // 2)
	rms = _frame_rms(audio, frame_len, hop)
	if rms.size == 0:
		return audio[:0]
	threshold = numpy.percentile(rms, percentile)
	noise_frames = []
	for i, value in enumerate(rms):
		if value <= threshold:
			start = i * hop
			noise_frames.append(audio[start:start + frame_len])
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
) -> numpy.ndarray:
	"""
Apply noise reduction using the noisereduce library with a noise clip estimate.
"""
	if audio.size == 0:
		return audio.astype(numpy.float32, copy=False)
	if audio.size < n_fft:
		return audio.astype(numpy.float32, copy=False)

	try:
		import noisereduce
	except ImportError as exc:
		raise RuntimeError("noisereduce is required for noise reduction") from exc

	noise_clip = _noise_clip_from_percentile(audio, sample_rate, frame_ms, percentile)
	if noise_clip.size == 0:
		noise_clip = audio[: min(audio.size, sample_rate // 2)]

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
	return reduced.astype(numpy.float32, copy=False)


def apply_spectral_subtraction (
	audio: numpy.ndarray,
	sample_rate: int,
	oversub: float = 0.7,
	floor: float = 0.06
) -> numpy.ndarray:
	"""
Spectral subtraction with gain smoothing to reduce musical noise.
"""
	if audio.size == 0:
		return audio.astype(numpy.float32, copy=False)

	frame_len = max(256, int(sample_rate * 0.02))
	n_fft = 1 << (frame_len - 1).bit_length()
	if audio.size < n_fft:
		return audio.astype(numpy.float32, copy=False)

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

	magnitude = numpy.abs(zxx)
	phase = zxx / numpy.maximum(magnitude, 1e-10)
	frame_energy = numpy.mean(magnitude * magnitude, axis=0)
	energy_threshold = numpy.percentile(frame_energy, 20.0)
	noise_frames = magnitude[:, frame_energy <= energy_threshold]
	if noise_frames.size == 0:
		noise_mag = numpy.median(magnitude, axis=1, keepdims=True)
	else:
		noise_mag = numpy.mean(noise_frames, axis=1, keepdims=True)

	subtracted = numpy.maximum(magnitude - noise_mag * oversub, noise_mag * floor)

	gain = subtracted / numpy.maximum(magnitude, 1e-10)
	smooth_kernel = numpy.ones((3, 5), dtype=numpy.float32)
	smooth_kernel /= smooth_kernel.sum()
	gain = scipy.signal.convolve2d(gain, smooth_kernel, mode="same", boundary="symm")
	gain = numpy.clip(gain, 0.0, 1.0)
	zxx_denoised = magnitude * gain * phase

	_, denoised = scipy.signal.istft(
		zxx_denoised,
		fs=sample_rate,
		window="hann",
		nperseg=n_fft,
		noverlap=n_fft - hop,
		input_onesided=True,
		boundary=True
	)
	return denoised[: audio.size].astype(numpy.float32, copy=False)
