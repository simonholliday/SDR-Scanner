"""Tests for spectral subtraction noise reduction."""

import numpy
import pytest

import sdr_scanner.dsp.noise_reduction


class TestFrameRMS:

	def test_ones_rms (self):
		"""RMS of all-ones signal should be 1.0."""
		audio = numpy.ones(1000, dtype=numpy.float32)
		rms = sdr_scanner.dsp.noise_reduction._frame_rms(audio, 256, 128)
		assert len(rms) > 0
		numpy.testing.assert_allclose(rms, 1.0, atol=1e-5)

	def test_short_signal (self):
		"""Signal shorter than frame_len should still return one frame."""
		audio = numpy.ones(50, dtype=numpy.float32) * 0.5
		rms = sdr_scanner.dsp.noise_reduction._frame_rms(audio, 256, 128)
		assert len(rms) == 1
		assert rms[0] == pytest.approx(0.5, abs=1e-5)

	def test_zeros_rms (self):
		audio = numpy.zeros(1000, dtype=numpy.float32)
		rms = sdr_scanner.dsp.noise_reduction._frame_rms(audio, 256, 128)
		numpy.testing.assert_allclose(rms, 0.0, atol=1e-7)


class TestNoiseClipFromPercentile:

	def test_noise_only (self):
		"""With uniform noise, most frames should be selected."""
		rng = numpy.random.default_rng(42)
		audio = rng.standard_normal(16000).astype(numpy.float32) * 0.01
		clip = sdr_scanner.dsp.noise_reduction._noise_clip_from_percentile(audio, 16000)
		# Should return a significant portion of the audio
		assert len(clip) > len(audio) * 0.1

	def test_signal_plus_silence (self):
		"""Noise clip should come from the silent portion."""
		audio = numpy.zeros(16000, dtype=numpy.float32)
		audio[8000:] = 0.5  # loud second half
		clip = sdr_scanner.dsp.noise_reduction._noise_clip_from_percentile(audio, 16000)
		# Clip should be from the quiet first half
		assert numpy.max(numpy.abs(clip)) < 0.1


class TestSpectralSubtraction:

	def test_noise_gets_quieter (self):
		"""Spectral subtraction should reduce the RMS of noise-only input."""
		rng = numpy.random.default_rng(42)
		noise = rng.standard_normal(16000).astype(numpy.float32) * 0.1
		rms_before = numpy.sqrt(numpy.mean(noise ** 2))
		denoised, nmag = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(
			noise, 16000
		)
		rms_after = numpy.sqrt(numpy.mean(denoised ** 2))
		assert rms_after < rms_before

	def test_tone_preserved (self):
		"""A strong tone in noise should still be present after subtraction."""
		sr = 16000
		t = numpy.arange(sr, dtype=numpy.float32) / sr
		tone = 0.5 * numpy.sin(2 * numpy.pi * 1000 * t)
		rng = numpy.random.default_rng(42)
		noisy = (tone + rng.standard_normal(sr).astype(numpy.float32) * 0.05).astype(numpy.float32)
		denoised, _ = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(noisy, sr)
		# FFT to check tone is still dominant
		spectrum = numpy.abs(numpy.fft.rfft(denoised))
		freqs = numpy.fft.rfftfreq(len(denoised), d=1.0 / sr)
		spectrum[0] = 0  # ignore DC
		peak_freq = freqs[numpy.argmax(spectrum)]
		assert abs(peak_freq - 1000) < 100

	def test_noise_mag_reuse (self):
		"""Second call with returned noise_mag should be consistent."""
		rng = numpy.random.default_rng(42)
		audio = rng.standard_normal(16000).astype(numpy.float32) * 0.1
		_, nmag = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		denoised2, nmag2 = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(
			audio, 16000, noise_mag=nmag
		)
		# noise_mag should be reused (same object)
		assert nmag2 is nmag

	def test_adaptive_noise_estimation (self):
		"""adaptive_noise_estimation parameter should not crash."""
		rng = numpy.random.default_rng(42)
		audio = rng.standard_normal(16000).astype(numpy.float32) * 0.1
		denoised, _ = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(
			audio, 16000, adaptive_noise_estimation=True
		)
		assert len(denoised) == len(audio)

	def test_empty_input (self):
		audio = numpy.array([], dtype=numpy.float32)
		out, nmag = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		assert len(out) == 0

	def test_short_input (self):
		"""Input shorter than FFT size should be returned unchanged."""
		audio = numpy.ones(10, dtype=numpy.float32) * 0.5
		out, _ = sdr_scanner.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		numpy.testing.assert_array_equal(out, audio)
