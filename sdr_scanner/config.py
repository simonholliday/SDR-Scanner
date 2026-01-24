"""
Configuration loading and validation for SDR Scanner
"""

from __future__ import annotations

import typing

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import sdr_scanner.constants


def _normalize_label(value: typing.Any) -> typing.Any:
	if value is None:
		return value
	if isinstance(value, str):
		return value.strip().upper()
	return value


def _normalize_gain(value: typing.Any) -> typing.Any:
	if value is None:
		return 'auto'
	if isinstance(value, str):
		text = value.strip().lower()
		if text == 'auto':
			return 'auto'
		try:
			return float(text)
		except ValueError as exc:
			raise ValueError("sdr_gain_db must be a number or 'auto'") from exc
	return float(value)

class ScannerConfig(BaseModel):
	"""Scanner-level configuration."""

	model_config = ConfigDict(extra='forbid')

	sdr_device_sample_size: int = Field(gt=0)
	band_time_slice_ms: int = Field(gt=0)
	sample_queue_maxsize: int = Field(default=30, gt=0)
	calibration_frequency_hz: float | None = Field(default=None, gt=0)


class RecordingConfig(BaseModel):
	"""Recording configuration."""

	model_config = ConfigDict(extra='forbid')

	enabled: bool = False
	buffer_size_seconds: float = Field(default=30.0, ge=0.0)
	disk_flush_interval_seconds: float = Field(default=5.0, gt=0.0)
	audio_sample_rate: int = Field(default=16000, gt=0)
	audio_output_dir: str = './audio'
	fade_in_ms: float | None = Field(default=None, ge=0.0)
	fade_out_ms: float | None = Field(default=None, ge=0.0)
	soft_limit_drive: float = Field(default=2.0, gt=0.0)


class BandTypeConfig(BaseModel):
	"""Default settings for a band type (e.g., DMR, TETRA)."""

	model_config = ConfigDict(extra='forbid')

	channel_spacing: float | None = Field(default=None, gt=0)
	sample_rate: float | None = Field(default=None, gt=0)
	channel_width: float | None = Field(default=None, gt=0)
	modulation: str | None = None
	snr_threshold_db: float | None = Field(default=None)
	sdr_gain_db: float | str | None = 'auto'

	@field_validator('modulation', mode='before')
	@classmethod
	def _validate_label(cls, value: typing.Any) -> typing.Any:
		return _normalize_label(value)

	@field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain(cls, value: typing.Any) -> typing.Any:
		return _normalize_gain(value)


class BandConfig(BaseModel):
	"""Per-band configuration."""

	model_config = ConfigDict(extra='forbid')

	freq_start: float = Field(gt=0)
	freq_end: float = Field(gt=0)
	channel_spacing: float = Field(gt=0)
	sample_rate: float = Field(gt=0)
	channel_width: float | None = Field(default=None, gt=0)
	type: str | None = None
	modulation: str | None = None
	exclude_channel_indices: list[int] = Field(default_factory=list)
	snr_threshold_db: float = Field(default=12.0)
	sdr_gain_db: float | str | None = 'auto'

	@field_validator('modulation', 'type', mode='before')
	@classmethod
	def _validate_label(cls, value: typing.Any) -> typing.Any:
		return _normalize_label(value)

	@field_validator('exclude_channel_indices', mode='before')
	@classmethod
	def _validate_exclusions(cls, value: typing.Any) -> list[int]:
		if value is None:
			return []
		if not isinstance(value, list):
			raise ValueError("exclude_channel_indices must be a list of integers")
		indices: list[int] = []
		for item in value:
			idx = int(item)
			if idx < 0:
				raise ValueError("exclude_channel_indices must be >= 0")
			indices.append(idx)
		return indices

	@field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain(cls, value: typing.Any) -> typing.Any:
		return _normalize_gain(value)

	@model_validator(mode='after')
	def _validate_band(self) -> 'BandConfig':
		if self.freq_start >= self.freq_end:
			raise ValueError('freq_start must be less than freq_end')
		if self.channel_width is None:
			self.channel_width = self.channel_spacing * sdr_scanner.constants.CHANNEL_WIDTH_FRACTION
		if self.snr_threshold_db <= sdr_scanner.constants.HYSTERESIS_DB:
			raise ValueError(
				f"snr_threshold_db must be > {sdr_scanner.constants.HYSTERESIS_DB} dB to allow OFF hysteresis"
			)

		return self


class AppConfig(BaseModel):
	"""Top-level application configuration."""

	model_config = ConfigDict(extra='forbid')

	scanner: ScannerConfig
	recording: RecordingConfig = Field(default_factory=RecordingConfig)
	band_defaults: dict[str, BandTypeConfig] = Field(default_factory=dict)
	bands: dict[str, BandConfig]

	@model_validator(mode='after')
	def _validate_bands(self) -> 'AppConfig':
		if not self.bands:
			raise ValueError('bands must contain at least one band')
		return self


def _load_raw_config(config_path: str) -> dict:
	"""
	Load raw configuration data from YAML.
	"""
	with open(config_path, 'r') as f:
		data = yaml.safe_load(f)
	if data is None:
		raise ValueError(f"Config file is empty: {config_path}")
	if not isinstance(data, dict):
		raise ValueError('Config root must be a mapping')
	return data


def _apply_band_defaults(data: dict) -> dict:
	"""
	Merge per-type defaults into bands (band values override defaults).
	"""
	defaults_raw = data.get('band_defaults')
	bands_raw = data.get('bands')
	if not isinstance(defaults_raw, dict) or not isinstance(bands_raw, dict):
		return data

	normalized_types: dict[str, dict] = {}
	for type_name, type_defaults in defaults_raw.items():
		if not isinstance(type_name, str) or not isinstance(type_defaults, dict):
			continue
		normalized_types[type_name.strip().upper()] = type_defaults

	merged_bands: dict[str, dict | typing.Any] = {}
	for band_name, band_config in bands_raw.items():
		if not isinstance(band_config, dict):
			merged_bands[band_name] = band_config
			continue

		band_type = band_config.get('type')
		if isinstance(band_type, str):
			type_key = band_type.strip().upper()
			type_defaults = normalized_types.get(type_key)
			if isinstance(type_defaults, dict):
				merged = dict(type_defaults)
				merged.update(band_config)
				merged_bands[band_name] = merged
				continue

		merged_bands[band_name] = band_config

	merged = dict(data)
	merged['band_defaults'] = normalized_types
	merged['bands'] = merged_bands
	return merged


def load_config(config_path: str) -> AppConfig:
	"""
	Load and validate configuration from YAML file.
	"""
	data = _load_raw_config(config_path)
	data = _apply_band_defaults(data)
	return AppConfig.model_validate(data)


def validate_config(config: dict | AppConfig) -> AppConfig:
	"""
	Validate configuration data and return a typed AppConfig.
	"""
	if isinstance(config, AppConfig):
		return config
	return AppConfig.model_validate(_apply_band_defaults(config))


def get_band_config(config: dict | AppConfig, band_name: str) -> BandConfig:
	"""
	Get configuration for a specific band.
	"""
	typed_config = validate_config(config)
	if band_name not in typed_config.bands:
		available_bands = ', '.join(typed_config.bands.keys())
		raise KeyError(f"Band '{band_name}' not found in configuration. Available bands: {available_bands}")

	return typed_config.bands[band_name]
