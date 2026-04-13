"""Tests for channel frequency calculation, IQ extraction, and device overrides."""

import fractions

import numpy
import pytest

import substation.config
import substation.devices
import substation.scanner

import iq_generators


class TestCalculateChannels:

	def test_pmr446_channels (self, scanner_instance):
		"""PMR446: 8 channels from 446.00625 to 446.09375 at 12.5 kHz spacing."""
		channels = scanner_instance.channels
		# Our test config has freq_start=446.00625e6, freq_end=446.09375e6, spacing=12500
		expected_n = 8
		assert len(channels) == expected_n
		# Check first and last
		assert channels[0] == pytest.approx(446.00625e6)
		assert channels[-1] == pytest.approx(446.09375e6)

	def test_two_channels (self, minimal_config_dict):
		"""A band spanning exactly one spacing should have 2 channels."""
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 446.00625e6 + 12500.0
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		assert len(sc.channels) == 2
		assert sc.channels[0] == pytest.approx(446.00625e6)
		assert sc.channels[1] == pytest.approx(446.00625e6 + 12500.0)

	def test_no_float_drift (self, minimal_config_dict):
		"""Even with many channels, the last frequency should match freq_end exactly."""
		# 100 channels
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 100e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 101.2375e6
		minimal_config_dict["bands"]["test_nfm"]["channel_spacing"] = 12500.0
		minimal_config_dict["bands"]["test_nfm"]["sample_rate"] = 2.4e6
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		assert len(sc.channels) == 100
		assert sc.channels[-1] == pytest.approx(101.2375e6, rel=1e-10)

	def test_excluded_channels (self, minimal_config_dict):
		"""Excluded indices should be removed from the channel list."""
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [1, 3]
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		# Original 8 channels minus 2 excluded = 6
		assert len(sc.channels) == 6


class TestChannelExtraction:

	def test_extract_tone_on_channel (self, scanner_instance):
		"""A tone at a channel frequency should be present in the extracted IQ."""
		sc = scanner_instance
		sc._precompute_fft_params()
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, 0.05, amplitude=0.5)
		noise = iq_generators.generate_noise_iq(sc.sample_rate, 0.05, power_db=-40)
		iq = (tone + noise)[:sc.samples_per_slice].astype(numpy.complex64)
		extracted = sc._extract_channel_iq(iq, ch_freq)
		# Extracted signal should have significant energy (the tone was shifted to baseband)
		power = numpy.mean(numpy.abs(extracted) ** 2)
		assert power > 0.01

	def test_phase_continuity (self, scanner_instance):
		"""Extracting the same channel across two blocks should have no phase jump."""
		sc = scanner_instance
		ch_freq = sc.channels[0]
		offset_hz = ch_freq - sc.center_freq
		n = sc.samples_per_slice * 2
		tone = iq_generators.generate_tone_iq(offset_hz, sc.sample_rate, n / sc.sample_rate, amplitude=0.5)
		block1 = tone[:sc.samples_per_slice].astype(numpy.complex64)
		block2 = tone[sc.samples_per_slice:n].astype(numpy.complex64)

		ext1 = sc._extract_channel_iq(block1, ch_freq, sample_offset=0)
		sc.sample_counter += sc.samples_per_slice
		ext2 = sc._extract_channel_iq(block2, ch_freq, sample_offset=0)

		if len(ext1) > 0 and len(ext2) > 0:
			# Phase at boundary should be continuous
			phase_jump = numpy.abs(numpy.angle(ext2[0]) - numpy.angle(ext1[-1]))
			# Wrap to [-pi, pi]
			phase_jump = min(phase_jump, 2 * numpy.pi - phase_jump)
			assert phase_jump < 0.5  # radians — small jump


class TestDcOffset:

	def test_center_shifted_when_channel_on_dc (self, minimal_config_dict):
		"""Center frequency shifts by half a channel when a channel falls on DC."""
		# 9 channels (odd) → midpoint at 100.05 MHz lands exactly on channel 4
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 100e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 100e6 + 12500 * 8
		minimal_config_dict["bands"]["test_nfm"]["channel_spacing"] = 12500
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr"
		)
		midpoint = (100e6 + 100e6 + 12500 * 8) / 2  # 100050000
		# Center should have been shifted by half a channel spacing
		assert sc.center_freq == pytest.approx(midpoint + 6250)

	def test_center_unchanged_when_dc_clear (self, scanner_instance):
		"""Center frequency unchanged when DC falls between channels."""
		sc = scanner_instance
		midpoint = (sc.freq_start + sc.freq_end) / 2
		# PMR's midpoint (446.1 MHz) falls between channels — no shift
		assert sc.center_freq == pytest.approx(midpoint)


class TestNormalizeDeviceFamily:

	def test_rtlsdr_aliases (self):
		assert substation.devices.normalize_device_family("rtlsdr") == "rtlsdr"
		assert substation.devices.normalize_device_family("rtl") == "rtlsdr"
		assert substation.devices.normalize_device_family("RTL-SDR") == "rtlsdr"

	def test_hackrf_aliases (self):
		assert substation.devices.normalize_device_family("hackrf") == "hackrf"
		assert substation.devices.normalize_device_family("HackRF-One") == "hackrf"

	def test_airspy_aliases (self):
		assert substation.devices.normalize_device_family("airspy") == "airspy"
		assert substation.devices.normalize_device_family("airspy-r2") == "airspy"
		assert substation.devices.normalize_device_family("AirSpyR2") == "airspy"

	def test_airspyhf_aliases (self):
		assert substation.devices.normalize_device_family("airspyhf") == "airspyhf"
		assert substation.devices.normalize_device_family("airspy-hf") == "airspyhf"
		assert substation.devices.normalize_device_family("airspyhf+") == "airspyhf"

	def test_soapy_driver (self):
		assert substation.devices.normalize_device_family("soapy:lime") == "lime"
		assert substation.devices.normalize_device_family("soapy:airspy") == "airspy"

	def test_unknown_passthrough (self):
		assert substation.devices.normalize_device_family("bladerf") == "bladerf"


class TestDeviceOverrideApplied:

	def test_override_applied_on_matching_device (self, minimal_config_dict):
		"""Device override should replace base values when device matches."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6, "snr_threshold_db": 8.0},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="airspy",
		)
		assert sc.sample_rate == 2.5e6
		assert sc.snr_threshold_db == 8.0

	def test_override_not_applied_on_different_device (self, minimal_config_dict):
		"""Device override should not be applied when device doesn't match."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="rtlsdr",
		)
		assert sc.sample_rate == 1.024e6

	def test_override_preserves_base_fields (self, minimal_config_dict):
		"""Fields not in the override should keep their base values."""
		minimal_config_dict["bands"]["test_nfm"]["device_overrides"] = {
			"airspy": {"sample_rate": 2.5e6},
		}
		config = substation.config.validate_config(minimal_config_dict)
		sc = substation.scanner.RadioScanner(
			config=config, band_name="test_nfm", device_type="airspy",
		)
		# snr_threshold_db should be unchanged from base config
		assert sc.snr_threshold_db == 12.0
