"""Tests for decimation, filtering, and fade curves."""

import logging

import numpy
import pytest

import substation.dsp.filters


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

class TestDecimateAudio:

	def test_integer_decimation_length (self):
		sr = 2_048_000
		audio_rate = 16_000
		n = sr  # 1 second
		signal = numpy.random.default_rng(0).standard_normal(n).astype(numpy.float32)
		out, _ = substation.dsp.filters.decimate_audio(signal, sr, audio_rate, {})
		expected_len = n // (sr // audio_rate)
		# Allow +-2 samples tolerance for filter transients
		assert abs(len(out) - expected_len) <= 2

	def test_decimation_output_dtype (self):
		signal = numpy.ones(4096, dtype=numpy.float32)
		out, _ = substation.dsp.filters.decimate_audio(signal, 64000, 16000, {})
		assert out.dtype == numpy.float32

	def test_state_continuity (self):
		"""Two consecutive blocks should produce similar output to one big block."""
		sr = 64000
		ar = 16000
		rng = numpy.random.default_rng(1)
		full = rng.standard_normal(sr).astype(numpy.float32)
		half = sr // 2

		# One big block
		out_full, _ = substation.dsp.filters.decimate_audio(full, sr, ar, {})

		# Two half blocks with state
		state = {}
		out_a, state = substation.dsp.filters.decimate_audio(full[:half], sr, ar, state)
		out_b, state = substation.dsp.filters.decimate_audio(full[half:], sr, ar, state)
		out_split = numpy.concatenate([out_a, out_b])

		# Lengths should match within 1 sample
		min_len = min(len(out_full), len(out_split))
		assert min_len > 0
		# After filter settling, they should be very close
		settle = min(50, min_len // 4)
		correlation = numpy.corrcoef(out_full[settle:min_len], out_split[settle:min_len])[0, 1]
		assert correlation > 0.95


class TestDecimateIQ:

	def test_iq_decimation (self):
		sr = 1_024_000
		target = 64000
		n = sr // 10  # 100ms
		signal = numpy.exp(2j * numpy.pi * 1000 * numpy.arange(n) / sr).astype(numpy.complex64)
		out, _ = substation.dsp.filters.decimate_iq(signal, sr, target, {})
		assert out.dtype == numpy.complex64
		assert len(out) > 0


class TestDecimateCommonBackstop:

	"""
	Regression tests for the defensive backstop in _decimate_common.

	The rational-resample path can be asked to design a polyphase filter
	of length ~20 * max(up, down), and for coprime sample/target rates
	that blows up — 2.5 MHz → 64103 Hz coprime would allocate ~50 million
	taps and lock the system up.  The backstop refuses any call where
	max(up, down) > _RESAMPLE_MAX_FACTOR, logs a clear error, and returns
	an empty array.  This test pokes the backstop directly so it's not
	dead code in CI.
	"""

	def test_refuses_pathological_ratio_and_returns_empty (self, caplog):

		"""
		sample_rate=2_500_000 → target_rate=64_103 is the exact pathological
		case that motivated the backstop: gcd = 1, so up=64_103, down=2_500_000,
		max(up, down) = 2_500_000 which is well past the 100_000 cap.
		"""

		signal = numpy.zeros(1024, dtype=numpy.complex64)

		with caplog.at_level(logging.ERROR, logger='substation.dsp.filters'):
			out, _ = substation.dsp.filters.decimate_iq(
				signal, 2_500_000, 64_103, {}
			)

		assert out.size == 0
		assert out.dtype == numpy.complex64

		error_messages = [
			r.getMessage() for r in caplog.records
			if 'Refusing pathological resample' in r.getMessage()
		]
		assert len(error_messages) == 1

	def test_preserves_real_dtype_on_refusal (self, caplog):

		"""
		When given a real (float32) input, the backstop should still
		return an empty array with matching dtype so downstream .astype
		calls don't unexpectedly widen.
		"""

		signal = numpy.zeros(1024, dtype=numpy.float32)

		with caplog.at_level(logging.ERROR, logger='substation.dsp.filters'):
			out, _ = substation.dsp.filters.decimate_audio(
				signal, 2_500_000, 64_103, {}
			)

		assert out.size == 0
		assert out.dtype == numpy.float32

	def test_normal_rational_resample_not_refused (self):

		"""
		A normal rational resample (2.4 MHz → 48 kHz) must still succeed —
		the backstop threshold should be well above typical config values.
		"""

		rng = numpy.random.default_rng(0)
		signal = rng.standard_normal(2400).astype(numpy.complex64)

		out, _ = substation.dsp.filters.decimate_iq(signal, 2_400_000, 48_000, {})

		# 2.4 MHz / 48 kHz = 50, exact integer divisor → integer path
		assert len(out) > 0
		assert out.dtype == numpy.complex64


# ---------------------------------------------------------------------------
# Fade curves
# ---------------------------------------------------------------------------

class TestApplyFade:

	def test_half_cosine_shape (self):
		audio = numpy.ones(1000, dtype=numpy.float32)
		faded = substation.dsp.filters.apply_fade(audio.copy(), 16000, 5.0, 5.0)
		# Starts at 0, ends at 0
		assert faded[0] == pytest.approx(0.0, abs=1e-6)
		assert faded[-1] == pytest.approx(0.0, abs=1e-6)
		# Middle is 1
		assert faded[len(faded) // 2] == pytest.approx(1.0, abs=0.01)

	def test_fade_in_only (self):
		audio = numpy.ones(500, dtype=numpy.float32)
		faded = substation.dsp.filters.apply_fade(audio.copy(), 16000, 10.0, None)
		assert faded[0] == pytest.approx(0.0, abs=1e-6)
		assert faded[-1] == pytest.approx(1.0)

	def test_fade_out_only (self):
		audio = numpy.ones(500, dtype=numpy.float32)
		faded = substation.dsp.filters.apply_fade(audio.copy(), 16000, None, 10.0)
		assert faded[0] == pytest.approx(1.0)
		assert faded[-1] == pytest.approx(0.0, abs=1e-6)

	def test_both_none_returns_unchanged (self):
		audio = numpy.ones(100, dtype=numpy.float32) * 0.7
		result = substation.dsp.filters.apply_fade(audio, 16000, None, None)
		numpy.testing.assert_array_equal(result, audio)

	def test_empty_audio (self):
		audio = numpy.array([], dtype=numpy.float32)
		result = substation.dsp.filters.apply_fade(audio, 16000, 5.0, 5.0)
		assert len(result) == 0

	def test_pad_constrained_fade_in (self):
		"""Fade should only affect the padding region."""
		audio = numpy.ones(1000, dtype=numpy.float32)
		pad = 50
		faded = substation.dsp.filters.apply_fade(
			audio.copy(), 16000, 15.0, None, pad_in_samples=pad
		)
		# Padding region faded
		assert faded[0] == pytest.approx(0.0, abs=1e-6)
		# Just past padding is untouched
		assert faded[pad] == pytest.approx(1.0, abs=0.01)

	def test_pad_constrained_fade_out (self):
		audio = numpy.ones(1000, dtype=numpy.float32)
		pad = 80
		faded = substation.dsp.filters.apply_fade(
			audio.copy(), 16000, None, 15.0, pad_out_samples=pad
		)
		assert faded[-1] == pytest.approx(0.0, abs=1e-6)
		# Just before padding region is untouched
		assert faded[-(pad + 1)] == pytest.approx(1.0, abs=0.01)

	def test_fade_longer_than_audio (self):
		"""Fade durations exceeding audio length are clamped."""
		audio = numpy.ones(10, dtype=numpy.float32)
		faded = substation.dsp.filters.apply_fade(audio.copy(), 16000, 1000.0, 1000.0)
		assert faded[0] == pytest.approx(0.0, abs=1e-6)
		assert len(faded) == 10

	def test_monotonic_fade_in (self):
		audio = numpy.ones(500, dtype=numpy.float32)
		faded = substation.dsp.filters.apply_fade(audio.copy(), 16000, 20.0, None)
		fade_len = int(16000 * 0.02)
		fade_region = faded[:fade_len]
		# Should be non-decreasing
		assert numpy.all(numpy.diff(fade_region) >= -1e-7)
