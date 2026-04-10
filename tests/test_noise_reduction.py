"""Tests for spectral subtraction noise reduction."""

import numpy
import pytest

import substation.dsp.noise_reduction


class TestFrameRMS:

	def test_ones_rms (self):
		"""RMS of all-ones signal should be 1.0."""
		audio = numpy.ones(1000, dtype=numpy.float32)
		rms = substation.dsp.noise_reduction._frame_rms(audio, 256, 128)
		assert len(rms) > 0
		numpy.testing.assert_allclose(rms, 1.0, atol=1e-5)

	def test_short_signal (self):
		"""Signal shorter than frame_len should still return one frame."""
		audio = numpy.ones(50, dtype=numpy.float32) * 0.5
		rms = substation.dsp.noise_reduction._frame_rms(audio, 256, 128)
		assert len(rms) == 1
		assert rms[0] == pytest.approx(0.5, abs=1e-5)

	def test_zeros_rms (self):
		audio = numpy.zeros(1000, dtype=numpy.float32)
		rms = substation.dsp.noise_reduction._frame_rms(audio, 256, 128)
		numpy.testing.assert_allclose(rms, 0.0, atol=1e-7)


class TestNoiseClipFromPercentile:

	def test_noise_only (self):
		"""With uniform noise, most frames should be selected."""
		rng = numpy.random.default_rng(42)
		audio = rng.standard_normal(16000).astype(numpy.float32) * 0.01
		clip = substation.dsp.noise_reduction._noise_clip_from_percentile(audio, 16000)
		# Should return a significant portion of the audio
		assert len(clip) > len(audio) * 0.1

	def test_signal_plus_silence (self):
		"""Noise clip should come from the silent portion."""
		audio = numpy.zeros(16000, dtype=numpy.float32)
		audio[8000:] = 0.5  # loud second half
		clip = substation.dsp.noise_reduction._noise_clip_from_percentile(audio, 16000)
		# Clip should be from the quiet first half
		assert numpy.max(numpy.abs(clip)) < 0.1


class TestSpectralSubtraction:

	def test_noise_gets_quieter (self):
		"""Spectral subtraction should reduce the RMS of noise-only input."""
		rng = numpy.random.default_rng(42)
		noise = rng.standard_normal(16000).astype(numpy.float32) * 0.1
		rms_before = numpy.sqrt(numpy.mean(noise ** 2))
		denoised, nmag = substation.dsp.noise_reduction.apply_spectral_subtraction(
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
		denoised, _ = substation.dsp.noise_reduction.apply_spectral_subtraction(noisy, sr)
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
		_, nmag = substation.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		denoised2, nmag2 = substation.dsp.noise_reduction.apply_spectral_subtraction(
			audio, 16000, noise_mag=nmag
		)
		# noise_mag should be reused (same object)
		assert nmag2 is nmag

	def test_adaptive_noise_estimation (self):
		"""adaptive_noise_estimation parameter should not crash."""
		rng = numpy.random.default_rng(42)
		audio = rng.standard_normal(16000).astype(numpy.float32) * 0.1
		denoised, _ = substation.dsp.noise_reduction.apply_spectral_subtraction(
			audio, 16000, adaptive_noise_estimation=True
		)
		assert len(denoised) == len(audio)

	def test_empty_input (self):
		audio = numpy.array([], dtype=numpy.float32)
		out, nmag = substation.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		assert len(out) == 0

	def test_short_input (self):
		"""Input shorter than FFT size should be returned unchanged."""
		audio = numpy.ones(10, dtype=numpy.float32) * 0.5
		out, _ = substation.dsp.noise_reduction.apply_spectral_subtraction(audio, 16000)
		numpy.testing.assert_array_equal(out, audio)


def _dbfs_to_linear (dbfs: float) -> float:
	"""Helper: convert a dBFS level to linear amplitude."""
	return 10.0 ** (dbfs / 20.0)


class TestDynamicsCurve:

	"""Tests for the apply_dynamics_curve function."""

	def test_passes_through_at_threshold (self):

		"""A sample at exactly threshold_dbfs should be unchanged (the curves are tangent to unity there)."""

		level = _dbfs_to_linear(-20.0)
		samples = numpy.array([level, -level], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-20.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		numpy.testing.assert_allclose(out, samples, atol=1e-5)

	def test_passes_through_at_full_scale (self):

		"""Samples at +/-1.0 (0 dBFS) should be unchanged — the boost hump is zero at the upper endpoint."""

		samples = numpy.array([1.0, -1.0], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-20.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		numpy.testing.assert_allclose(out, samples, atol=1e-5)

	def test_zeros_below_floor (self):

		"""Samples whose magnitude is at or below the floor should output as exactly 0.0."""

		below_floor = _dbfs_to_linear(-65.0)
		at_floor    = _dbfs_to_linear(-60.0)
		samples = numpy.array([below_floor, -below_floor, at_floor, -at_floor], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-20.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		numpy.testing.assert_array_equal(out, numpy.zeros_like(samples))

	def test_disabled_when_amounts_are_zero (self):

		"""With cut_db=0 and boost_db=0, samples in [floor, 0] dBFS pass through unchanged."""

		# Cover all three regions: cut zone, boost zone, and exactly at threshold
		levels = numpy.array([-50.0, -30.0, -20.0, -10.0, -3.0])
		samples = numpy.concatenate([_dbfs_to_linear(levels), -_dbfs_to_linear(levels)]).astype(numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-20.0, cut_db=0.0, boost_db=0.0, floor_dbfs=-60.0,
		)

		numpy.testing.assert_allclose(out, samples, atol=1e-6)

	def test_monotonically_increasing (self):

		"""Sweeping input from -1 to +1 should produce a monotonically non-decreasing output."""

		samples = numpy.linspace(-1.0, 1.0, 4096, dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-25.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		# Allow exact equality (silence region produces a long run of zeros)
		assert numpy.all(numpy.diff(out) >= -1e-7), "Output is not monotonically non-decreasing"

	def test_no_output_above_full_scale (self):

		"""No output sample should exceed +/-1.0 even with adventurous boost settings."""

		# Configuration that would, without the defensive clamp, push the boost
		# region midpoint above 0 dBFS: threshold -10 dBFS, boost 8 dB at midpoint -5 dBFS → +3 dBFS.
		samples = numpy.linspace(-0.99, 0.99, 2048, dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-10.0, cut_db=6.0, boost_db=8.0, floor_dbfs=-60.0,
		)

		assert numpy.max(numpy.abs(out)) <= 1.0 + 1e-6

	def test_cut_curve_skews_toward_threshold (self):

		"""cut_curve < 0.5 places the steepest gradient nearer the threshold,
		so the gain reduction at a level just below threshold should be larger
		with cut_curve=0.1 than with cut_curve=0.9."""

		# Pick a single test level just below the threshold
		level_db = -22.0
		sample = numpy.array([_dbfs_to_linear(level_db)], dtype=numpy.float32)

		out_steep_low = substation.dsp.noise_reduction.apply_dynamics_curve(
			sample, threshold_dbfs=-20.0, cut_db=6.0, boost_db=0.0, floor_dbfs=-60.0,
			cut_curve=0.1,
		)
		out_steep_high = substation.dsp.noise_reduction.apply_dynamics_curve(
			sample, threshold_dbfs=-20.0, cut_db=6.0, boost_db=0.0, floor_dbfs=-60.0,
			cut_curve=0.9,
		)

		# cut_curve=0.1 puts the steep part near the threshold, so by -22 dBFS
		# the curve has already accumulated more reduction than the cut_curve=0.9 case.
		assert out_steep_low[0] < out_steep_high[0], (
			f"Expected larger reduction with cut_curve=0.1 (steeper near threshold), "
			f"got {out_steep_low[0]:.6f} vs {out_steep_high[0]:.6f}"
		)

	def test_handles_zero_input (self):

		"""An array of all zeros should return all zeros without raising on log10(0)."""

		samples = numpy.zeros(128, dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-25.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		numpy.testing.assert_array_equal(out, samples)
		assert not numpy.any(numpy.isnan(out))

	def test_handles_empty_input (self):

		"""An empty input should return an empty output without errors."""

		samples = numpy.array([], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-25.0,
		)

		assert out.size == 0

	def test_preserves_shape_and_dtype (self):

		"""Output should match input shape and dtype."""

		samples = numpy.linspace(-0.5, 0.5, 1024, dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			samples, threshold_dbfs=-25.0, cut_db=6.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		assert out.shape == samples.shape
		assert out.dtype == samples.dtype

	def test_quiet_sample_is_cut (self):

		"""A sample halfway between floor and threshold should be cut by ~cut_db at the midpoint."""

		# With threshold=-20, floor=-60, cut_db=6, cut_curve=0.5:
		# Midpoint is at input -40 dBFS, expected reduction = 6 dB → output ~ -46 dBFS.
		input_db = -40.0
		sample = numpy.array([_dbfs_to_linear(input_db)], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			sample, threshold_dbfs=-20.0, cut_db=6.0, boost_db=0.0, floor_dbfs=-60.0,
		)

		out_db = 20.0 * numpy.log10(out[0])

		assert out_db == pytest.approx(input_db - 6.0, abs=0.1)

	def test_loud_sample_is_boosted (self):

		"""A sample halfway between threshold and 0 dBFS should be boosted by ~boost_db."""

		# With threshold=-20, boost_db=1.5, boost_curve=0.5:
		# Midpoint is at input -10 dBFS, expected boost = 1.5 dB → output ~ -8.5 dBFS.
		input_db = -10.0
		sample = numpy.array([_dbfs_to_linear(input_db)], dtype=numpy.float32)

		out = substation.dsp.noise_reduction.apply_dynamics_curve(
			sample, threshold_dbfs=-20.0, cut_db=0.0, boost_db=1.5, floor_dbfs=-60.0,
		)

		out_db = 20.0 * numpy.log10(out[0])

		assert out_db == pytest.approx(input_db + 1.5, abs=0.1)

	def test_validation_floor_not_below_threshold (self):

		"""floor_dbfs >= threshold_dbfs should raise ValueError."""

		samples = numpy.zeros(8, dtype=numpy.float32)

		with pytest.raises(ValueError, match="floor_dbfs"):
			substation.dsp.noise_reduction.apply_dynamics_curve(
				samples, threshold_dbfs=-20.0, floor_dbfs=-10.0,
			)

	def test_validation_threshold_not_at_or_above_zero (self):

		"""threshold_dbfs >= 0 should raise ValueError."""

		samples = numpy.zeros(8, dtype=numpy.float32)

		with pytest.raises(ValueError, match="threshold_dbfs"):
			substation.dsp.noise_reduction.apply_dynamics_curve(
				samples, threshold_dbfs=0.0,
			)

	def test_validation_negative_amounts_rejected (self):

		"""Negative cut_db or boost_db should raise ValueError."""

		samples = numpy.zeros(8, dtype=numpy.float32)

		with pytest.raises(ValueError, match="cut_db"):
			substation.dsp.noise_reduction.apply_dynamics_curve(
				samples, threshold_dbfs=-20.0, cut_db=-1.0,
			)

	def test_validation_curve_out_of_range_rejected (self):

		"""cut_curve or boost_curve outside [0, 1] should raise ValueError."""

		samples = numpy.zeros(8, dtype=numpy.float32)

		with pytest.raises(ValueError, match="curve"):
			substation.dsp.noise_reduction.apply_dynamics_curve(
				samples, threshold_dbfs=-20.0, cut_curve=1.5,
			)
