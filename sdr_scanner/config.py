"""
Configuration loading and validation for SDR Scanner.

This module handles loading configuration from YAML files and validates it using
Pydantic models. The configuration system supports:
- Type validation with helpful error messages
- Default values for optional parameters
- Band type templates (e.g., "DMR", "TETRA") that can be applied to multiple bands
- Per-band overrides of type defaults

Pydantic provides automatic type checking, validation, and clear error messages
when configuration is invalid, which is much better than runtime errors or silent
failures.
"""

from __future__ import annotations

import fractions
import typing
import pydantic
import yaml

import sdr_scanner.constants


def _fraction_constructor (loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> fractions.Fraction:
	"""
	YAML constructor for the !fraction tag.

	Converts a string like "25000/3" into a fractions.Fraction object.
	This allows representing precise radio frequencies that cannot be
	perfectly represented as floating-point numbers.
	"""
	value = loader.construct_scalar(node)
	return fractions.Fraction(value)


# Register the !fraction tag with the YAML loader.
yaml.SafeLoader.add_constructor('!fraction', _fraction_constructor)


def _normalize_label (value: typing.Any) -> typing.Any:
	"""
	Normalize string labels to uppercase for case-insensitive matching.

	Used for modulation types ("NFM", "AM") and band types ("DMR", "TETRA")
	so users can write them in any case in the config file.

	Args:
		value: Input value (string, None, or other type)

	Returns:
		Uppercase string if input is string, otherwise original value
	"""
	if value is None:
		return value

	if isinstance(value, str):
		return value.strip().upper()

	return value


def _normalize_gain (value: typing.Any) -> typing.Any:
	"""
	Normalize gain values to either 'auto' or a float.

	Gain can be specified as:
	- None → 'auto'
	- "auto" (any case) → 'auto'
	- Number → float (e.g., 20.0 for 20 dB)

	This allows flexible configuration while ensuring consistent internal representation.

	Args:
		value: Input gain value (None, string, or number)

	Returns:
		Either 'auto' string or float value

	Raises:
		ValueError: If string value is not 'auto' and can't be parsed as number
	"""
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

class ScannerConfig(pydantic.BaseModel):
	"""
	Scanner-level configuration (applies to all bands).

	These parameters control the core scanning engine behavior: how samples
	are acquired from the SDR, how they're buffered, and optional frequency
	calibration.

	Attributes:
		sdr_device_sample_size: Number of samples to read from SDR in each chunk.
			Larger = more efficient USB transfers, smaller = lower latency.
			Must be power of 2 for RTL-SDR (e.g., 16384, 32768, 65536).

		band_time_slice_ms: How often to analyze the spectrum (in milliseconds).
			100ms is typical: fast enough to detect brief transmissions but
			not so fast that processing can't keep up. Longer = less CPU, but
			slower detection.

		sample_queue_maxsize: Maximum number of sample blocks to buffer.
			If processing falls behind, oldest samples are dropped when queue fills.
			Larger queue = more tolerance for processing spikes, but more memory.

		calibration_frequency_hz: Optional frequency of a known strong signal
			for automatic PPM calibration (e.g., 93.7e6 for a local FM station).
			If None, calibration is skipped. Only works with RTL-SDR and HackRF.
	"""

	# Reject unknown fields (catch typos in config file)
	model_config = pydantic.ConfigDict(extra='forbid')

	sdr_device_sample_size: int = pydantic.Field(gt=0)
	band_time_slice_ms: int = pydantic.Field(gt=0)
	sample_queue_maxsize: int = pydantic.Field(default=30, gt=0)
	calibration_frequency_hz: float | None = pydantic.Field(default=None, gt=0)


class RecordingConfig(pydantic.BaseModel):
	"""
	Recording configuration (applies to all recorded bands).

	Controls how audio is recorded, buffered, and saved to WAV files.

	Attributes:
		buffer_size_seconds: Maximum audio buffer size per channel (in seconds).
			If disk writes fall behind, oldest audio is dropped. Larger = more
			tolerance for slow disks, but more memory usage. 30s is generous.

		disk_flush_interval_seconds: How often to write buffered audio to disk.
			More frequent = less buffering delay and memory, but more I/O overhead.
			5 seconds is a good balance.

		audio_sample_rate: Output audio sample rate in Hz.
			16000 (16 kHz) is sufficient for voice and uses half the disk space
			of 32 kHz. Higher rates preserve more high-frequency content.

		audio_output_dir: Directory where WAV files are saved.
			Files are organized as: output_dir/YYYY-MM-DD/band_name/filename.wav

		fade_in_ms: Fade-in duration in milliseconds when recording starts.
			Prevents clicks from sudden audio onset. None = no fade. 50-100ms typical.

		fade_out_ms: Fade-out duration in milliseconds when recording ends.
			Prevents clicks from sudden cutoff. None = no fade. 50-100ms typical.

		soft_limit_drive: Soft limiter aggressiveness (1.0 to 4.0).
			Higher = more compression of loud signals. 2.0 is moderate limiting.
			Prevents clipping while maintaining some dynamic range.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	buffer_size_seconds: float = pydantic.Field(default=30.0, ge=0.0)
	disk_flush_interval_seconds: float = pydantic.Field(default=5.0, gt=0.0)
	audio_sample_rate: int = pydantic.Field(default=16000, gt=0)
	audio_output_dir: str = './audio'
	fade_in_ms: float | None = pydantic.Field(default=None, ge=0.0)
	fade_out_ms: float | None = pydantic.Field(default=None, ge=0.0)
	soft_limit_drive: float = pydantic.Field(default=2.0, gt=0.0)
	noise_reduction_enabled: bool = pydantic.Field(default=True)


class BandTypeConfig(pydantic.BaseModel):
	"""
	Template for common band types (e.g., DMR, TETRA, PMR).

	Allows defining default settings for a type of radio service, then applying
	those defaults to multiple bands. For example, all DMR bands can share the
	same channel spacing (12.5 kHz), modulation (4FSK), etc.

	Bands can override any of these defaults by specifying the parameter explicitly.

	All fields are optional here (bands must specify required fields or inherit them).

	Attributes:
		channel_spacing: Spacing between channel center frequencies in Hz.
			E.g., 12.5 kHz for PMR/DMR, 25 kHz for marine VHF.

		sample_rate: SDR sample rate in Hz. Must be high enough to cover
			the entire band plus margins. E.g., 2 MHz for PMR446 (188 kHz band).

		channel_width: Occupied bandwidth per channel in Hz.
			If None, defaults to channel_spacing * 0.84 (leaves guard bands).

		modulation: Modulation scheme (e.g., "NFM", "AM", "WFM").
			Determines which demodulator is used for recording.

		recording_enabled: Whether to record audio from detected transmissions.
			False = detection only (no audio files created).

		snr_threshold_db: Minimum SNR to consider a channel active (in dB).
			Lower = more sensitive (detects weak signals) but more false positives.

		sdr_gain_db: SDR gain setting in dB, or 'auto' for automatic gain control.
			Higher gain = more sensitive but also amplifies noise and can cause clipping.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	channel_spacing: float | None = pydantic.Field(default=None, gt=0)
	sample_rate: float | None = pydantic.Field(default=None, gt=0)
	channel_width: float | None = pydantic.Field(default=None, gt=0)
	modulation: str | None = None
	recording_enabled: bool = False
	snr_threshold_db: float | None = pydantic.Field(default=None)
	sdr_gain_db: float | str | None = 'auto'

	@pydantic.field_validator('modulation', mode='before')
	@classmethod
	def _validate_label (cls, value: typing.Any) -> typing.Any:
		"""Normalize modulation labels to uppercase for case-insensitive matching."""
		return _normalize_label(value)

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float."""
		return _normalize_gain(value)


class BandConfig(pydantic.BaseModel):
	"""
	Configuration for a specific band to scan.

	Each band defines a frequency range, channel spacing, and scanning parameters.
	The scanner will monitor all channels in this band simultaneously.

	Attributes:
		freq_start: Start of frequency range in Hz (e.g., 446.00625e6 for PMR446).

		freq_end: End of frequency range in Hz (e.g., 446.19375e6 for PMR446).
			Must be greater than freq_start.

		channel_spacing: Spacing between channel centers in Hz (e.g., 12500 for 12.5 kHz).
			Channels are generated from freq_start to freq_end with this spacing.

		sample_rate: SDR sample rate in Hz. Must be high enough to cover
			(freq_end - freq_start + channel_width + margins). E.g., 2 MHz for PMR446.

		channel_width: Occupied bandwidth per channel in Hz.
			If None, defaults to channel_spacing * 0.84 (leaves 16% guard bands).

		type: Optional band type for inheriting defaults (e.g., "DMR", "TETRA").
			If specified, inherits default values from band_defaults section.

		modulation: Modulation scheme for demodulation (e.g., "NFM", "AM").
			Required if recording_enabled is True.

		recording_enabled: Whether to record audio from active channels.
			If False, only detection is performed (no audio files).

		exclude_channel_indices: List of channel indices to skip (0-indexed).
			Useful for excluding known interference or out-of-band channels.
			E.g., [0, 1] excludes the first two channels.

		snr_threshold_db: Minimum SNR to detect a channel as active (in dB).
			Default 12 dB is conservative. Lower values (e.g., 8-10 dB) detect
			weaker signals but may have more false positives from noise.

		sdr_gain_db: SDR gain in dB, or 'auto' for AGC.
			'auto' is convenient but may not be optimal. Manual gain (e.g., 20-40 dB
			for RTL-SDR) often works better for specific scenarios.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	freq_start: float = pydantic.Field(gt=0)
	freq_end: float = pydantic.Field(gt=0)
	channel_spacing: float = pydantic.Field(gt=0)
	sample_rate: float = pydantic.Field(gt=0)
	channel_width: float | None = pydantic.Field(default=None, gt=0)
	type: str | None = None
	modulation: str | None = None
	recording_enabled: bool = False
	exclude_channel_indices: list[int] = pydantic.Field(default_factory=list)
	snr_threshold_db: float = pydantic.Field(default=12.0)
	sdr_gain_db: float | str | None = 'auto'

	@pydantic.field_validator('modulation', 'type', mode='before')
	@classmethod
	def _validate_label (cls, value: typing.Any) -> typing.Any:
		"""Normalize labels to uppercase for case-insensitive matching."""
		return _normalize_label(value)

	@pydantic.field_validator('exclude_channel_indices', mode='before')
	@classmethod
	def _validate_exclusions (cls, value: typing.Any) -> list[int]:
		"""
		Validate and normalize channel exclusion list.

		Converts None to empty list, validates that all values are non-negative integers.
		"""
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

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float."""
		return _normalize_gain(value)

	@pydantic.model_validator(mode='after')
	def _validate_band (self) -> 'BandConfig':
		"""
		Cross-field validation after all fields are parsed.

		Validates:
		- freq_end > freq_start
		- Sets default channel_width if not specified
		- Ensures snr_threshold_db is high enough for hysteresis to work
		"""
		if self.freq_start >= self.freq_end:
			raise ValueError('freq_start must be less than freq_end')

		# Default channel width to 84% of spacing (leaves guard bands)
		if self.channel_width is None:
			self.channel_width = self.channel_spacing * sdr_scanner.constants.CHANNEL_WIDTH_FRACTION

		# Ensure SNR threshold is high enough for hysteresis
		# OFF threshold = ON threshold - HYSTERESIS_DB, so ON must be > HYSTERESIS_DB
		if self.snr_threshold_db <= sdr_scanner.constants.HYSTERESIS_DB:
			raise ValueError(
				f"snr_threshold_db must be > {sdr_scanner.constants.HYSTERESIS_DB} dB to allow OFF hysteresis"
			)

		return self


class AppConfig(pydantic.BaseModel):
	"""
	Top-level application configuration.

	This is the root of the configuration hierarchy, containing global settings
	(scanner and recording) and band-specific settings.

	Configuration file structure:
		scanner: {...}           # Global scanner settings
		recording: {...}         # Global recording settings
		band_defaults:           # Optional templates for band types
			DMR: {...}
			TETRA: {...}
		bands:                   # Actual bands to scan
			pmr: {...}
			airband: {...}

	Attributes:
		scanner: Global scanner configuration (required).

		recording: Global recording configuration (optional, uses defaults if not specified).

		band_defaults: Optional templates for band types. Bands can reference these
			via the 'type' field to inherit default values.

		bands: Dictionary of bands to scan, keyed by band name.
			At least one band is required.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	scanner: ScannerConfig
	recording: RecordingConfig = pydantic.Field(default_factory=RecordingConfig)
	band_defaults: dict[str, BandTypeConfig] = pydantic.Field(default_factory=dict)
	bands: dict[str, BandConfig]

	@pydantic.model_validator(mode='after')
	def _validate_bands (self) -> 'AppConfig':
		"""Ensure at least one band is configured."""
		if not self.bands:
			raise ValueError('bands must contain at least one band')

		return self


def _load_raw_config (config_path: str) -> dict:

	"""
	Load raw configuration data from YAML file.

	Performs basic validation (file exists, contains valid YAML, root is a dict)
	but doesn't validate the structure or types yet (that's done by Pydantic).

	Args:
		config_path: Path to YAML configuration file

	Returns:
		Raw configuration as a dictionary

	Raises:
		FileNotFoundError: If config file doesn't exist
		ValueError: If file is empty or root element is not a dict
		yaml.YAMLError: If file contains invalid YAML syntax
	"""

	with open(config_path, 'r') as f:
		data = yaml.safe_load(f)

	if data is None:
		raise ValueError(f"Config file is empty: {config_path}")

	if not isinstance(data, dict):
		raise ValueError('Config root must be a mapping')

	return data


def _apply_band_defaults (data: dict) -> dict:

	"""
	Apply band type defaults to individual bands.

	This implements the template/inheritance system: if a band specifies
	type: "DMR", it inherits all defaults from band_defaults.DMR, with
	explicit band values taking precedence.

	Example:
		band_defaults:
			DMR:
				channel_spacing: 12500
				modulation: "NFM"
		bands:
			dmr_band1:
				type: "DMR"          # Inherits channel_spacing and modulation
				freq_start: 446e6
				freq_end: 447e6

	The merging is shallow (no recursive merging of nested dicts).

	Args:
		data: Raw configuration dictionary

	Returns:
		Configuration with band defaults merged into individual bands
	"""

	defaults_raw = data.get('band_defaults')
	bands_raw = data.get('bands')

	# Skip if no defaults or bands defined
	if not isinstance(defaults_raw, dict) or not isinstance(bands_raw, dict):
		return data

	# Normalize type names to uppercase for case-insensitive matching
	normalized_types: dict[str, dict] = {}
	for type_name, type_defaults in defaults_raw.items():
		if not isinstance(type_name, str) or not isinstance(type_defaults, dict):
			continue

		normalized_types[type_name.strip().upper()] = type_defaults

	# Merge defaults into each band that specifies a type
	merged_bands: dict[str, dict | typing.Any] = {}
	for band_name, band_config in bands_raw.items():
		if not isinstance(band_config, dict):
			merged_bands[band_name] = band_config
			continue

		band_type = band_config.get('type')

		# If band has a type, merge in the defaults for that type
		if isinstance(band_type, str):
			type_key = band_type.strip().upper()
			type_defaults = normalized_types.get(type_key)

			if isinstance(type_defaults, dict):
				# Defaults first, then band config (band values override)
				merged = dict(type_defaults)
				merged.update(band_config)
				merged_bands[band_name] = merged
				continue

		# No type or type not found: use band config as-is
		merged_bands[band_name] = band_config

	# Return modified config with merged bands
	merged = dict(data)
	merged['band_defaults'] = normalized_types
	merged['bands'] = merged_bands

	return merged


def load_config (config_path: str) -> AppConfig:

	"""
	Load and validate configuration from YAML file.

	This is the main entry point for loading configuration. It:
	1. Loads raw YAML from file
	2. Applies band type defaults
	3. Validates using Pydantic (type checking, range validation, etc.)

	Args:
		config_path: Path to YAML configuration file

	Returns:
		Validated AppConfig object

	Raises:
		FileNotFoundError: If config file doesn't exist
		ValueError: If config structure is invalid
		pydantic.ValidationError: If config values are invalid
	"""

	data = _load_raw_config(config_path)
	data = _apply_band_defaults(data)

	return AppConfig.model_validate(data)


def validate_config (config: dict | AppConfig) -> AppConfig:

	"""
	Validate configuration data and return a typed AppConfig.

	Can accept either a raw dict (from YAML) or an existing AppConfig.
	Useful for testing or programmatic configuration.

	Args:
		config: Configuration as dict or AppConfig

	Returns:
		Validated AppConfig object

	Raises:
		pydantic.ValidationError: If config values are invalid
	"""

	if isinstance(config, AppConfig):
		return config

	return AppConfig.model_validate(_apply_band_defaults(config))


def get_band_config (config: dict | AppConfig, band_name: str) -> BandConfig:

	"""
	Extract configuration for a specific band.

	Validates the config if needed, then returns the configuration for
	the requested band.

	Args:
		config: Configuration as dict or AppConfig
		band_name: Name of the band (key in config.bands)

	Returns:
		Configuration for the specified band

	Raises:
		KeyError: If band name not found in configuration
		pydantic.ValidationError: If config is invalid
	"""

	typed_config = validate_config(config)

	if band_name not in typed_config.bands:
		available_bands = ', '.join(typed_config.bands.keys())
		raise KeyError(f"Band '{band_name}' not found in configuration. Available bands: {available_bands}")

	return typed_config.bands[band_name]
