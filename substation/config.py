"""
Configuration loading and validation for Substation.

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

import fractions
import logging
import pathlib
import typing

import pydantic
import yaml

import substation.constants

logger = logging.getLogger(__name__)


def _fraction_constructor (loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> fractions.Fraction:
	"""
	YAML constructor for the !fraction tag.

	Converts a string like "25000/3" into a fractions.Fraction object.
	This allows representing precise radio frequencies that cannot be
	perfectly represented as floating-point numbers.
	"""

	value = loader.construct_scalar(node)
	return fractions.Fraction(value)


# Custom loader subclass so the !fraction tag doesn't mutate the global SafeLoader.
class _YamlLoader (yaml.SafeLoader):
	pass

_YamlLoader.add_constructor('!fraction', _fraction_constructor)


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
	
	# Optional: threshold in seconds to consider a channel "stuck" (e.g., constant transmitter).
	# If exceeded, a warning will be logged to console. Set to null or remove to disable.
	stuck_channel_threshold_seconds: float | None = pydantic.Field(default=None, gt=0)


class DynamicsCurveConfig(pydantic.BaseModel):

	"""
	Parameters for the experimental dynamics-curve noise reduction stage.

	apply_dynamics_curve is a per-sample dual-region expander: a smoothstep
	S-curve cuts the level of quiet samples below the threshold (downward
	expansion, suppresses noise), and a sin² hump gently boosts samples
	above the threshold (upward expansion, gives voice presence).

	See substation.dsp.noise_reduction.apply_dynamics_curve for the full
	mathematical specification.

	Attributes:
		threshold_dbfs: Dividing line between cut and boost regions, in dBFS.
			Must be strictly less than 0.  Typical values: -20 to -35 dBFS.

		cut_db: Gain reduction at the midpoint of the cut S-curve.  The
			maximum reduction (at the floor) is twice this value.  Set to
			0.0 to disable the cut region.  Default: 6.0.

		boost_db: Maximum gain boost at the peak of the boost hump.  Set
			to 0.0 to disable the boost region.  Default: 1.5.

		floor_dbfs: Lower threshold below which output is hard-zeroed.
			Must be strictly less than threshold_dbfs.  Default: -60.0.

		cut_curve: Position of the steepest gradient within the cut
			S-curve, in [0, 1].  0.5 = symmetric.  Lower values shift the
			steepest part toward the threshold; higher values shift it
			toward the floor.

		boost_curve: Same concept for the boost hump.  0.5 = symmetric.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	threshold_dbfs: float = pydantic.Field(default=-25.0, lt=0.0)
	cut_db: float = pydantic.Field(default=6.0, ge=0.0)
	boost_db: float = pydantic.Field(default=1.5, ge=0.0)
	floor_dbfs: float = pydantic.Field(default=-60.0, lt=0.0)
	cut_curve: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)
	boost_curve: float = pydantic.Field(default=0.5, ge=0.0, le=1.0)

	@pydantic.model_validator(mode='after')
	def _validate_levels (self) -> 'DynamicsCurveConfig':

		"""
		Cross-field validation for the level parameters.

		Enforces that the floor sits strictly below the threshold so the
		cut region has a non-zero width.  Logs a warning (does not raise)
		if the boost configuration could push the output above 0 dBFS —
		that case is also caught by a defensive clamp inside the function,
		but a startup warning gives the user a chance to dial it back
		before listening to the result.
		"""

		if self.floor_dbfs >= self.threshold_dbfs:
			raise ValueError(
				f"floor_dbfs ({self.floor_dbfs}) must be strictly below "
				f"threshold_dbfs ({self.threshold_dbfs})"
			)

		# Maximum boost output occurs near the midpoint of the boost
		# region, where the input level is roughly threshold_dbfs / 2 and
		# the boost is up to boost_db.  If that sum exceeds 0 dBFS the
		# defensive clamp inside apply_dynamics_curve will engage.
		boost_peak_dbfs = (self.threshold_dbfs / 2.0) + self.boost_db

		if boost_peak_dbfs > 0.0:
			logger.warning(
				f"dynamics_curve: boost_db={self.boost_db} dB combined with "
				f"threshold_dbfs={self.threshold_dbfs} would push the boost "
				f"region above 0 dBFS (~{boost_peak_dbfs:+.1f} dBFS at the peak); "
				f"the defensive clamp will engage and the curve will not match "
				f"the configured shape.  Reduce boost_db or lower threshold_dbfs."
			)

		return self


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

		audio_format: Output audio file format: 'wav' or 'flac'.
			WAV (default) is uncompressed and embeds broadcast metadata (BEXT)
			with sample-accurate timestamps, allowing audio editors to place
			recordings on a timeline at their real capture time.
			FLAC is lossless compressed (~39% smaller) but cannot carry BEXT
			timeline metadata — date and time are stored as text tags only.

		audio_output_dir: Directory where audio files are saved.
			Files are organized as: output_dir/YYYY-MM-DD/band_name/filename.{wav,flac}

		fade_in_ms: Fade-in duration in milliseconds when recording starts.
			Prevents clicks from sudden audio onset. None = no fade. 50-100ms typical.

		fade_out_ms: Fade-out duration in milliseconds when recording ends.
			Prevents clicks from sudden cutoff. None = no fade. 50-100ms typical.

		soft_limit_drive: Soft limiter aggressiveness (1.0 to 4.0).
			Higher = more compression of loud signals. 2.0 is moderate limiting.
			Prevents clipping while maintaining some dynamic range.

		noise_reduction_enabled: Toggle spectral-subtraction noise reduction
			on recorded audio.  Default: True.

		recording_hold_time_ms: Duration in ms to continue recording after
			signal drops below threshold (RF hold time).  Default: 500.

		discard_empty_enabled: Automatically discard noise-only recordings
			using spectral flatness.  Applies at turn-ON (Gate 2) and
			turn-OFF (Gate 3b).  Default: True.

		min_recording_seconds: Discard recordings shorter than this duration.
			Catches brief transients (radar, ignition).  Set to 0 to
			disable.  Default: 0.5.

		audio_silence_timeout_ms: Stop recording when demodulated audio has
			been silent for this duration, even if the RF carrier persists
			(common on AM airband).  Set to 0 to disable.  Default: 3000.

		dynamics_curve_enabled: Whether to apply the experimental dynamics-curve
			noise reduction stage to recorded audio.  Disabled by default; see
			DynamicsCurveConfig and apply_dynamics_curve for details.

		dynamics_curve: Parameters for the dynamics-curve stage.  Only consulted
			when dynamics_curve_enabled is True.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	buffer_size_seconds: float = pydantic.Field(default=30.0, gt=0.0)
	disk_flush_interval_seconds: float = pydantic.Field(default=5.0, gt=0.0)
	audio_sample_rate: int = pydantic.Field(default=16000, gt=0)
	audio_format: typing.Literal['wav', 'flac'] = 'wav'
	audio_output_dir: str = './audio'
	fade_in_ms: float | None = pydantic.Field(default=None, ge=0.0)
	fade_out_ms: float | None = pydantic.Field(default=None, ge=0.0)
	soft_limit_drive: float = pydantic.Field(default=2.0, gt=0.0)
	noise_reduction_enabled: bool = pydantic.Field(default=True)
	recording_hold_time_ms: float = pydantic.Field(default=500.0, ge=0.0)
	discard_empty_enabled: bool = pydantic.Field(default=True)
	min_recording_seconds: float = pydantic.Field(default=0.5, ge=0.0)
	audio_silence_timeout_ms: float = pydantic.Field(default=3000.0, ge=0.0)
	trim_carrier_transients: bool = pydantic.Field(default=False)
	dynamics_curve_enabled: bool = pydantic.Field(default=False)
	dynamics_curve: DynamicsCurveConfig = pydantic.Field(default_factory=DynamicsCurveConfig)


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


class DeviceOverrideConfig(pydantic.BaseModel):
	"""Device-specific overrides for a band.

	All fields are optional — only specified fields override the band's
	base values when the matching device type is selected at runtime.

	Attributes:
		sample_rate: Device-specific SDR sample rate (must cover band span).
		sdr_gain_db: Gain in dB or 'auto'.
		sdr_gain_elements: Per-element gain mapping (e.g., LNA, MIX, VGA).
		sdr_device_settings: Device-specific settings (e.g., bias tee).
		snr_threshold_db: Device-specific SNR threshold.
		activation_variance_db: Device-specific variance threshold.
	"""

	model_config = pydantic.ConfigDict(extra='forbid')

	sample_rate: float | None = pydantic.Field(default=None, gt=0)
	sdr_gain_db: float | str | None = None
	sdr_gain_elements: dict[str, float] | None = None
	sdr_device_settings: dict[str, str] | None = None
	snr_threshold_db: float | None = None
	activation_variance_db: float | None = pydantic.Field(default=None, ge=0)

	@pydantic.field_validator('sdr_gain_db', mode='before')
	@classmethod
	def _validate_gain (cls, value: typing.Any) -> typing.Any:
		"""Normalize gain to 'auto' or float, preserving None as 'not overridden'."""
		if value is None:
			return None
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

		sdr_gain_elements: Optional per-element gain mapping for devices with
			multiple gain stages (e.g., AirSpy R2 has LNA, Mixer, VGA).
			Element names are device-specific — check the log output at
			startup for available elements and their ranges.
			Mutually exclusive with sdr_gain_db — if both are set,
			sdr_gain_elements takes priority.

		sdr_device_settings: Optional device-specific settings passed through
			SoapySDR's writeSetting() API. Used for features like bias tee
			control, external clock configuration, or device calibration.
			Keys and values are device-specific strings.

		hysteresis_db: Margin in dB between the ON and OFF thresholds.
			A channel turns ON when SNR > snr_threshold_db, and OFF when
			SNR < (snr_threshold_db - hysteresis_db).  This prevents rapid
			toggling when SNR hovers near threshold.  Default 3.0 dB is
			good for strong signals; use a lower value (e.g. 1.5) when
			scanning weak signals with a low snr_threshold_db.

		activation_variance_db: Minimum power variance (dB) across segment PSDs
			required for a channel to be considered active.  Suppresses
			channel triggers caused by stationary noise that crosses the SNR
			threshold but contains no real signal.  Voice and data signals
			show 5-15+ dB variance over a typical detection slice; stationary
			noise shows under 2 dB.
			Applies to all bands regardless of recording state.
			Defaults to substation.constants.ACTIVATION_VARIANCE_DB.
			Set to 0 to disable the check entirely.
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
	hysteresis_db: float = pydantic.Field(default=3.0, ge=0)
	sdr_gain_db: float | str | None = 'auto'
	sdr_gain_elements: dict[str, float] | None = None
	sdr_device_settings: dict[str, str] | None = None
	activation_variance_db: float | None = pydantic.Field(default=None, ge=0)
	device_overrides: dict[str, DeviceOverrideConfig] | None = None

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
		- Warns if sdr_gain_elements overrides sdr_gain_db
		"""
		if self.freq_start >= self.freq_end:
			raise ValueError('freq_start must be less than freq_end')

		# Default channel width to 84% of spacing (leaves guard bands)
		if self.channel_width is None:
			self.channel_width = self.channel_spacing * substation.constants.CHANNEL_WIDTH_FRACTION

		# Warn if SNR threshold is at or below hysteresis margin.
		# OFF threshold = snr_threshold_db - hysteresis_db, which can go
		# negative (means "turn off at noise floor") — valid for weak signals.
		if self.snr_threshold_db <= self.hysteresis_db:
			logger.warning(
				f"snr_threshold_db ({self.snr_threshold_db}) <= hysteresis_db ({self.hysteresis_db}): "
				f"OFF threshold will be {self.snr_threshold_db - self.hysteresis_db:.1f} dB "
				f"(channel turns off at noise floor level)"
			)

		# Warn if per-element gain overrides sdr_gain_db
		if self.sdr_gain_elements is not None and self.sdr_gain_db is not None:
			logger.info(
				"sdr_gain_elements is set — sdr_gain_db will be ignored. "
				"Per-element gain takes priority for fine-tuned stage control."
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


def _deep_merge (
	base: dict[str, typing.Any],
	override: dict[str, typing.Any],
) -> dict[str, typing.Any]:

	"""Recursively merge *override* onto *base*, returning a new dict.

	For each key in override: if both values are dicts, recurse; otherwise the
	override value wins (including explicit None / YAML null).  Keys present in
	base but absent from override are preserved unchanged.  Neither input is
	mutated.
	"""

	result = dict(base)

	for key, override_value in override.items():
		base_value = result.get(key)
		if isinstance(base_value, dict) and isinstance(override_value, dict):
			result[key] = _deep_merge(base_value, override_value)
		elif isinstance(base_value, dict) and override_value is None:
			pass
		else:
			result[key] = override_value

	return result


def _locate_default_config () -> pathlib.Path:

	"""Return the path to the bundled config.yaml.default.

	Raises FileNotFoundError if the file is missing (broken installation).
	"""

	default = pathlib.Path(__file__).parent.parent / "config.yaml.default"

	if not default.exists():
		raise FileNotFoundError(
			f"Bundled config.yaml.default not found at {default}. "
			"The package installation may be corrupted."
		)

	return default


def _resolve_user_config_path (
	explicit: pathlib.Path | None,
) -> pathlib.Path | None:

	"""Return the user's config override path, or None if no user config exists.

	Priority: explicit path argument → ./config.yaml in CWD → None.
	When an explicit path is provided it must exist; no CWD fallback is tried.
	"""

	if explicit is not None:
		if explicit.exists():
			return explicit

		raise FileNotFoundError(f"Config file not found: {explicit}")

	cwd_config = pathlib.Path.cwd() / "config.yaml"
	if cwd_config.exists():
		return cwd_config

	return None


def _load_raw_config (config_path: pathlib.Path) -> dict:

	"""Load raw configuration data from YAML file.

	Performs basic validation (file exists, contains valid YAML, root is a dict)
	but doesn't validate the structure or types yet (that's done by Pydantic).
	"""

	with open(config_path, 'r') as f:
		data = yaml.load(f, Loader=_YamlLoader)

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
		if isinstance(band_type, str) and band_type.strip().upper() not in normalized_types:
			available = ', '.join(sorted(normalized_types.keys())) or '(none)'
			logger.warning(
				f"Band '{band_name}' specifies type '{band_type}' which is not defined in band_defaults. "
				f"Available types: {available}. No defaults will be inherited."
			)
		merged_bands[band_name] = band_config

	# Return modified config with merged bands
	merged = dict(data)
	merged['band_defaults'] = normalized_types
	merged['bands'] = merged_bands

	return merged


def load_config (path: pathlib.Path | None = None) -> AppConfig:

	"""Load configuration, merging config.yaml.default with config.yaml.

	Always loads config.yaml.default as the base.  If a user config.yaml exists
	(or an explicit path is given), it is deep-merged on top so user settings
	override defaults while unspecified keys inherit default values.
	"""

	default_path = _locate_default_config()
	base = _load_raw_config(default_path)

	user_path = _resolve_user_config_path(path)

	if user_path is not None and user_path.resolve() == default_path.resolve():
		user_path = None

	if user_path is not None:
		user = _load_raw_config(user_path)
		raw = _deep_merge(base, user)
		logger.debug("Loaded config from %s + %s", default_path.name, user_path.name)
	else:
		raw = base
		logger.debug("Loaded config from %s (no user overrides)", default_path.name)

	data = _apply_band_defaults(raw)

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
