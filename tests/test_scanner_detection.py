"""Tests for PSD calculation, noise floor estimation, and detection logic."""

import numpy
import pytest

import substation.constants
import substation.scanner

import iq_generators


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
