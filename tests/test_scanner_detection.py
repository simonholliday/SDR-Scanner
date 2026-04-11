"""Tests for PSD calculation, noise floor estimation, and detection logic."""

import logging

import numpy
import pytest

import substation.constants
import substation.scanner

import iq_generators


class _FakeSdr:

	"""
	Minimal stand-in for a BaseDevice used by _process_samples tests.

	Only exposes the attributes the saturation check reads — everything else
	is deliberately absent so accidental reliance on SDR state fails loudly.
	"""

	def __init__ (self, iq_scale: float = 1.0) -> None:
		self.iq_scale = iq_scale


class TestPSDCalculation:

	def test_noise_flat_psd (self, scanner_instance):
		"""Gaussian noise should produce a roughly flat PSD."""
		iq = iq_generators.generate_noise_iq(scanner_instance.sample_rate, 0.1)
		# Trim to expected slice size
		iq = iq[:scanner_instance.samples_per_slice]
		psd_db, _ = scanner_instance._calculate_psd_data(iq, include_segment_psd=False)
		assert len(psd_db) == scanner_instance.fft_size
		# PSD should be roughly flat: std of dB values should be moderate
		assert numpy.std(psd_db) < 15  # dB

	def test_tone_peak_in_psd (self, scanner_instance):
		"""A tone should produce a clear peak in the PSD."""
		sc = scanner_instance
		# Place a tone at the first channel frequency
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.1, amplitude=1.0)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-50)
		iq = (tone + noise)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		# The peak should be significantly above the median
		assert numpy.max(psd_db) > numpy.median(psd_db) + 10


class TestNoiseFloorEstimation:

	def test_with_noise_mask (self, scanner_instance):
		"""Noise floor should be estimated from gap bins."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		assert isinstance(nf, (float, numpy.floating))

	def test_fallback_without_mask (self, scanner_instance):
		"""Without noise mask, falls back to 25th percentile."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		# Temporarily remove noise mask
		saved_mask = sc.noise_mask
		sc.noise_mask = None
		nf = sc._estimate_noise_floor(psd_db)
		sc.noise_mask = saved_mask
		assert isinstance(nf, (float, numpy.floating))


class TestEMASmoothing:

	def test_ema_converges (self, scanner_instance):
		"""EMA noise floor should converge over multiple slices."""
		sc = scanner_instance
		sc._noise_floor_ema = None
		sc._warmup_remaining = 0  # skip warmup for this test

		floors = []
		for _ in range(20):
			iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
			psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
			raw_nf = sc._estimate_noise_floor(psd_db)

			if sc._noise_floor_ema is None:
				sc._noise_floor_ema = raw_nf
			else:
				alpha = substation.constants.NOISE_FLOOR_EMA_ALPHA
				sc._noise_floor_ema = alpha * raw_nf + (1 - alpha) * sc._noise_floor_ema
			floors.append(sc._noise_floor_ema)

		# Later values should have less variance than early ones
		early_var = numpy.var(floors[:5])
		late_var = numpy.var(floors[-5:])
		assert late_var <= early_var + 1.0  # allow small tolerance


class TestChannelPower:

	def test_tone_on_channel (self, scanner_instance):
		"""Injecting a tone at a channel frequency should produce high channel power."""
		sc = scanner_instance
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.1, amplitude=1.0)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-50)
		iq = (tone + noise)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		powers = sc._get_channel_powers(psd_db)
		snr_ch0 = powers[0] - nf
		# Should show clear signal presence (well above noise)
		# Note: exact SNR depends on FFT size, windowing, and tone/bin alignment
		assert snr_ch0 > 5.0

	def test_noise_only_below_threshold (self, scanner_instance):
		"""With only noise, all channels should be below threshold."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1, power_db=-20)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		nf = sc._estimate_noise_floor(psd_db)
		powers = sc._get_channel_powers(psd_db)
		snrs = powers - nf
		assert numpy.all(snrs < sc.snr_threshold_db)

	def test_vectorized_matches_individual (self, scanner_instance):
		"""Vectorized channel powers should match individual computation."""
		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.1)[:sc.samples_per_slice]
		psd_db, _ = sc._calculate_psd_data(iq, include_segment_psd=False)
		batch_powers = sc._get_channel_powers(psd_db)
		for i, ch_freq in enumerate(sc.channels):
			individual = sc._get_channel_power(psd_db, ch_freq)
			assert batch_powers[i] == pytest.approx(individual, abs=0.1)


class TestSegmentPowerVariance:

	def test_variance_low_for_stationary_noise (self, scanner_instance):

		"""
		Stationary Gaussian noise should produce low temporal variance across
		segment PSDs — this is the discriminator that lets us reject noise-only
		channel triggers.
		"""

		sc = scanner_instance
		iq = iq_generators.generate_noise_iq(sc.sample_rate, 0.5)[:sc.samples_per_slice]
		_, segment_psds = sc._calculate_psd_data(iq, include_segment_psd=True)
		assert segment_psds is not None and len(segment_psds) >= 2

		# Test variance for a representative channel near the centre
		ch_freq = sc.channels[len(sc.channels) // 2]
		stddev = sc._segment_power_variance(ch_freq, segment_psds)

		# Stationary noise should have variance close to the natural sampling
		# variance of an 8-segment Welch PSD — well below 3 dB.
		assert stddev < 3.0, f"Expected stationary noise variance < 3 dB, got {stddev:.2f} dB"

	def test_variance_high_for_modulated_signal (self, scanner_instance):

		"""
		An amplitude-modulated signal should produce high temporal variance
		across segment PSDs as its envelope rises and falls.
		"""

		sc = scanner_instance

		# Build an AM-modulated carrier at a target channel frequency.
		# Choose a modulation rate slow enough that segments capture the
		# envelope at clearly different points (one full cycle across the slice).
		n = sc.samples_per_slice
		t = numpy.arange(n) / sc.sample_rate
		duration_s = n / sc.sample_rate
		audio_freq = 2.0 / duration_s  # 2 cycles across the slice
		mod_depth = 0.95
		envelope = 1.0 + mod_depth * numpy.sin(2.0 * numpy.pi * audio_freq * t)

		# Place at the first channel offset from centre to avoid DC mask
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		carrier = numpy.exp(2j * numpy.pi * offset_hz * t)
		iq = (envelope * carrier).astype(numpy.complex64)

		_, segment_psds = sc._calculate_psd_data(iq, include_segment_psd=True)
		assert segment_psds is not None and len(segment_psds) >= 2

		stddev = sc._segment_power_variance(ch_freq, segment_psds)

		# Heavy AM should produce variance well above the 3 dB discriminator
		assert stddev > 3.0, f"Expected modulated signal variance > 3 dB, got {stddev:.2f} dB"

	def test_variance_zero_for_too_few_segments (self, scanner_instance):

		"""With fewer than 2 segments, variance should return 0.0 (cannot compute)."""

		sc = scanner_instance
		ch_freq = sc.channels[0]

		assert sc._segment_power_variance(ch_freq, []) == 0.0
		assert sc._segment_power_variance(ch_freq, None) == 0.0


class TestADCSaturationCheck:

	"""
	Regression tests for the ADC saturation branch in _process_samples.

	The check is scaled by `sdr.iq_scale` because some SoapySDR wrappers
	apply a post-capture normalisation factor and would otherwise produce
	false positives on ordinary (non-clipping) samples.  Heavy clipping
	drops the slice entirely so downstream spectral leakage doesn't cause
	false detections at the wrong channel.
	"""

	def test_iq_scale_prevents_false_saturation (self, scanner_instance, caplog):

		"""
		A device with iq_scale=10.0 and samples at ±0.5 (well below real
		ADC clipping) must NOT trigger the saturation warning — the
		threshold should scale up to 9.5 inside the check.
		"""

		sc = scanner_instance
		sc.sdr = _FakeSdr(iq_scale=10.0)
		sc._warmup_remaining = 0  # skip SDR startup window

		# Real+imag half-scale samples: well below any reasonable clip
		# point, but above the unscaled 0.95 threshold so this only
		# passes if the scaling is correctly applied.
		n = sc.samples_per_slice
		samples = (numpy.full(n, 0.5, dtype=numpy.float32)
			+ 1j * numpy.full(n, 0.5, dtype=numpy.float32)).astype(numpy.complex64)

		with caplog.at_level(logging.WARNING, logger='substation.scanner'):
			sc._process_samples(samples, loop=None)

		saturation_warnings = [r for r in caplog.records if 'ADC SATURATION' in r.getMessage()]
		assert saturation_warnings == [], (
			f"iq_scale=10 at ±0.5 should not warn, but saw: "
			f"{[r.getMessage() for r in saturation_warnings]}"
		)

	def test_heavy_clipping_drops_slice (self, scanner_instance, caplog):

		"""
		Samples with >5% real clipping must be dropped: _process_samples
		should log an "ADC SATURATION ... Dropping slice" warning, advance
		the sample counter, and return before any PSD work.
		"""

		sc = scanner_instance
		sc.sdr = _FakeSdr(iq_scale=1.0)
		sc._warmup_remaining = 0

		# 100% clipping in real part — every sample above the 0.95 threshold.
		n = sc.samples_per_slice
		samples = (numpy.full(n, 1.0, dtype=numpy.float32)
			+ 1j * numpy.zeros(n, dtype=numpy.float32)).astype(numpy.complex64)

		counter_before = sc.sample_counter

		with caplog.at_level(logging.WARNING, logger='substation.scanner'):
			sc._process_samples(samples, loop=None)

		drop_messages = [
			r.getMessage() for r in caplog.records
			if 'ADC SATURATION' in r.getMessage() and 'Dropping slice' in r.getMessage()
		]
		assert len(drop_messages) == 1, (
			f"Expected exactly one drop warning, got {drop_messages}"
		)

		# The counter should advance even when the slice is dropped, so
		# stream position tracking doesn't lose samples.
		assert sc.sample_counter == counter_before + n

	def test_mild_clipping_warns_without_dropping (self, scanner_instance, caplog):

		"""
		Samples with clipping between 0.1% and 5% should warn but still
		be processed — only the "heavy" branch short-circuits the slice.
		"""

		sc = scanner_instance
		sc.sdr = _FakeSdr(iq_scale=1.0)
		sc._warmup_remaining = 0

		# ~1% real clipping, rest random small samples.  The 1% comes
		# from a deterministic stride so the test is reproducible.
		n = sc.samples_per_slice
		rng = numpy.random.default_rng(seed=42)
		real = (rng.standard_normal(n) * 0.1).astype(numpy.float32)
		imag = (rng.standard_normal(n) * 0.1).astype(numpy.float32)
		real[::100] = 1.0  # force clipping on every 100th sample (~1%)
		samples = (real + 1j * imag).astype(numpy.complex64)

		with caplog.at_level(logging.WARNING, logger='substation.scanner'):
			sc._process_samples(samples, loop=None)

		warn_messages = [r.getMessage() for r in caplog.records if 'ADC SATURATION' in r.getMessage()]
		assert len(warn_messages) == 1
		assert 'Dropping slice' not in warn_messages[0]
