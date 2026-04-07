"""Tests for configuration loading and validation."""

import fractions
import logging

import pydantic
import pytest
import yaml

import sdr_scanner.config
import sdr_scanner.constants


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYamlLoading:

	def test_load_valid_yaml (self, tmp_path, minimal_config_dict):
		cfg_path = tmp_path / "config.yaml"
		cfg_path.write_text(yaml.dump(minimal_config_dict))
		config = sdr_scanner.config.load_config(str(cfg_path))
		assert "test_nfm" in config.bands

	def test_load_missing_file (self):
		with pytest.raises(FileNotFoundError):
			sdr_scanner.config.load_config("/nonexistent/config.yaml")

	def test_fraction_tag (self, tmp_path):
		content = "value: !fraction 25000/3\n"
		path = tmp_path / "frac.yaml"
		path.write_text(content)
		data = yaml.load(path.read_text(), Loader=sdr_scanner.config._SdrYamlLoader)
		assert data["value"] == fractions.Fraction(25000, 3)

	def test_empty_yaml_raises (self, tmp_path):
		path = tmp_path / "empty.yaml"
		path.write_text("")
		with pytest.raises(ValueError):
			sdr_scanner.config.load_config(str(path))


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------

class TestBandValidation:

	def test_valid_band (self, minimal_config_dict):
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		band = config.bands["test_nfm"]
		assert band.freq_start < band.freq_end

	def test_freq_start_ge_freq_end_raises (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["freq_start"] = 500e6
		minimal_config_dict["bands"]["test_nfm"]["freq_end"] = 400e6
		with pytest.raises(pydantic.ValidationError):
			sdr_scanner.config.validate_config(minimal_config_dict)

	def test_snr_threshold_below_hysteresis_raises (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["snr_threshold_db"] = 2.0
		with pytest.raises(pydantic.ValidationError):
			sdr_scanner.config.validate_config(minimal_config_dict)

	def test_sample_rate_below_band_span_accepted (self, minimal_config_dict):
		# Band span vs sample_rate is checked at scanner init, not config validation.
		# Config should accept bands wider than their sample_rate.
		minimal_config_dict["bands"]["test_nfm"]["sample_rate"] = 50000.0
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sample_rate == 50000.0

	def test_channel_width_defaults (self, app_config):
		band = app_config.bands["test_nfm"]
		expected = band.channel_spacing * sdr_scanner.constants.CHANNEL_WIDTH_FRACTION
		assert band.channel_width == pytest.approx(expected)

	def test_modulation_uppercase (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["modulation"] = "nfm"
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].modulation == "NFM"

	def test_gain_auto (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["sdr_gain_db"] = "auto"
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_gain_db == "auto"

	def test_gain_numeric (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["sdr_gain_db"] = 42
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].sdr_gain_db == 42.0

	def test_extra_field_rejected (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["bogus_field"] = True
		with pytest.raises(pydantic.ValidationError):
			sdr_scanner.config.validate_config(minimal_config_dict)


class TestRecordingValidation:

	def test_buffer_size_zero_raises (self, minimal_config_dict):
		minimal_config_dict["recording"] = {"buffer_size_seconds": 0}
		with pytest.raises(pydantic.ValidationError):
			sdr_scanner.config.validate_config(minimal_config_dict)


# ---------------------------------------------------------------------------
# Band defaults inheritance
# ---------------------------------------------------------------------------

class TestBandDefaults:

	def test_type_inherits_defaults (self, minimal_config_dict):
		minimal_config_dict["band_defaults"] = {
			"TEST_TYPE": {"snr_threshold_db": 8.0, "sdr_gain_db": 25}
		}
		minimal_config_dict["bands"]["test_nfm"]["type"] = "TEST_TYPE"
		# Band's own snr_threshold_db (12) should override the default (8)
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].snr_threshold_db == 12.0
		# sdr_gain_db from default (25) should be used if not set in band
		# (but our band has it set to 30, so it stays 30)
		assert config.bands["test_nfm"].sdr_gain_db == 30

	def test_unknown_type_warns (self, minimal_config_dict, caplog):
		minimal_config_dict["band_defaults"] = {"KNOWN_TYPE": {"sdr_gain_db": 10}}
		minimal_config_dict["bands"]["test_nfm"]["type"] = "UNKNOWN_TYPE"
		with caplog.at_level(logging.WARNING):
			sdr_scanner.config.validate_config(minimal_config_dict)
		assert "UNKNOWN_TYPE" in caplog.text


class TestExcludeChannelIndices:

	def test_valid_indices (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [0, 2]
		config = sdr_scanner.config.validate_config(minimal_config_dict)
		assert config.bands["test_nfm"].exclude_channel_indices == [0, 2]

	def test_negative_index_raises (self, minimal_config_dict):
		minimal_config_dict["bands"]["test_nfm"]["exclude_channel_indices"] = [-1]
		with pytest.raises(pydantic.ValidationError):
			sdr_scanner.config.validate_config(minimal_config_dict)
