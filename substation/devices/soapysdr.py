"""
SoapySDR device implementation for universal SDR hardware support.

Wraps the SoapySDR API to support any device with a SoapySDR driver module,
including AirSpy R2, AirSpy HF+ Discovery, and potentially any other
SoapySDR-compatible hardware.

SoapySDR uses a synchronous readStream() API, so this implementation runs
a dedicated reader thread internally to match the async callback interface
expected by BaseDevice.

Requirements:
- SoapySDR core and Python bindings (system package: python3-soapysdr)
- Device-specific SoapySDR modules (e.g., soapysdr-module-airspy)
- The Python venv must use --system-site-packages to access SoapySDR
"""

import logging
import threading
import time
import typing

import numpy
import numpy.typing

import substation.devices.base

logger = logging.getLogger(__name__)


class SoapySdrDevice (substation.devices.base.BaseDevice):

	"""
	Universal SDR device wrapper using SoapySDR.

	Supports any hardware with a SoapySDR driver module installed.
	Provides the async callback interface by running an internal
	reader thread around SoapySDR's synchronous readStream() API.
	"""

	def __init__ (self, driver: str, device_index: int = 0) -> None:

		"""
		Initialize a SoapySDR device.

		Enumerates devices matching the driver name, opens the device at the
		given index, and logs available capabilities (gain elements, sample
		rates, antennas, stream formats, and device-specific settings).

		Args:
			driver: SoapySDR driver name (e.g., 'airspy', 'airspyhf', 'rtlsdr')
			device_index: Device index for multi-device setups (default: 0)
		"""

		import SoapySDR

		self._soapy = SoapySDR
		self._driver = driver
		self._device_index = device_index

		# Enumerate matching devices and open by index

		results = SoapySDR.Device.enumerate({'driver': driver})

		if not results:
			raise RuntimeError(f"No SoapySDR devices found for driver '{driver}'")

		if device_index >= len(results):
			raise RuntimeError(
				f"Device index {device_index} out of range "
				f"({len(results)} '{driver}' device(s) found)"
			)

		self._device = SoapySDR.Device(results[device_index])

		# Cache state (avoids USB round-trips for repeated reads)
		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain: float | str | None = None
		self._gain_elements: dict[str, float] | None = None
		self._device_settings: dict[str, str] | None = None

		# Stream management
		self._stream: typing.Any = None
		self._stream_format: str | None = None

		# IQ normalisation scale factor.  Some SoapySDR modules (notably
		# SoapyAirspyHF) deliver CF32 samples at a much smaller scale than
		# the expected [-1, 1] range.  This factor is computed from a
		# calibration block at the start of streaming and applied to every
		# subsequent sample block.
		self._iq_scale: float = 1.0

		# Reader thread control
		self._reader_thread: threading.Thread | None = None
		self._stop_event = threading.Event()
		self._rx_buffer: numpy.typing.NDArray[numpy.complex64] = numpy.array([], dtype=numpy.complex64)

		# Log device capabilities for user configuration
		self._log_device_capabilities()

	def _log_device_capabilities (self) -> None:

		"""
		Log available device capabilities.

		Hardware identity (serial, driver name) stays at INFO — that's
		the one line users reliably want in their startup log to confirm
		which device was picked.  The rest of the capability dump (gain
		ranges, sample rates, antennas, stream formats) is demoted to
		DEBUG: the *active* values for each of these are logged at INFO
		elsewhere (`Sample rate:` in the scanner init block, `Gain
		element 'LNA' set to ...` after gain assignment, `Stream format:
		...` at stream start), so repeating the full capability table
		on every run is noise.  Enable DEBUG when configuring a new
		device for the first time to see the full list.
		"""

		hw_info = self._device.getHardwareInfo()

		if hw_info:
			logger.info(f"SoapySDR [{self._driver}]: {dict(hw_info)}")

		# Gain elements and ranges — useful when tuning a new device but
		# redundant once the per-element values are set and logged later.

		gain_elements = self._device.listGains(self._soapy.SOAPY_SDR_RX, 0)

		if gain_elements:
			parts = []
			for element in gain_elements:
				gain_range = self._device.getGainRange(self._soapy.SOAPY_SDR_RX, 0, element)
				parts.append(f"{element}: {gain_range.minimum():.0f}–{gain_range.maximum():.0f} dB")
			logger.debug(f"Gain elements: {', '.join(parts)}")

		# Overall gain range

		overall_range = self._device.getGainRange(self._soapy.SOAPY_SDR_RX, 0)
		logger.debug(f"Overall gain range: {overall_range.minimum():.0f}–{overall_range.maximum():.0f} dB")

		# AGC capability — note this is the device's advertised capability,
		# not whether the Substation code path will actually use it (the
		# AirSpy R2 reports True here but the auto-gain fallback logs its
		# real behaviour at INFO at the gain-setter site).

		has_agc = self._device.hasGainMode(self._soapy.SOAPY_SDR_RX, 0)
		logger.debug(f"Automatic gain control: {'supported' if has_agc else 'not supported'}")

		# Sample rates — the *chosen* rate is already logged at INFO by
		# the scanner before this method runs.

		sample_rates = self._device.listSampleRates(self._soapy.SOAPY_SDR_RX, 0)

		if sample_rates:
			rates_str = ', '.join(f"{r / 1e6:.3f} MHz" for r in sample_rates)
			logger.debug(f"Supported sample rates: {rates_str}")

		# Antennas

		antennas = self._device.listAntennas(self._soapy.SOAPY_SDR_RX, 0)

		if antennas:
			logger.debug(f"Antennas: {', '.join(antennas)}")

		# Stream format negotiation context.  The *chosen* format is
		# logged at INFO by _negotiate_stream_format() at stream start, so
		# the capability enumeration here is DEBUG-only.
		# getNativeStreamFormat requires a mutable fullScale output parameter
		# in the SoapySDR Python bindings; some older bindings have a
		# different signature and raise TypeError or AttributeError.

		try:
			native_fmt = self._device.getNativeStreamFormat(self._soapy.SOAPY_SDR_RX, 0, [0.0])
			supported_fmts = self._device.getStreamFormats(self._soapy.SOAPY_SDR_RX, 0)
			logger.debug(f"Native format: {native_fmt}, supported: {', '.join(supported_fmts)}")
		except (TypeError, AttributeError):
			supported_fmts = self._device.getStreamFormats(self._soapy.SOAPY_SDR_RX, 0)
			logger.debug(f"Supported formats: {', '.join(supported_fmts)}")

		# Device-specific settings (bias tee, clock source, etc.)

		settings_info = self._device.getSettingInfo()

		if settings_info:
			for info in settings_info:
				logger.debug(f"Device setting: {info.key} — {info.description} (default: {info.value})")

	@property
	def sample_rate (self) -> float | None:
		"""Get the current sample rate in Hz."""
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:

		"""
		Set the sample rate in Hz.

		After setting, reads back the actual rate the device applied.
		Some devices only support discrete rates and will round to the
		nearest supported value. If the actual rate differs from the
		requested rate, a warning is logged — this matters because the
		scanner uses the sample rate for all frequency calculations.
		"""

		# Check if the requested rate matches a supported rate.
		# Some devices (e.g., AirSpy R2) only support discrete rates and
		# may silently round to the nearest one without reporting the change.

		supported = self._device.listSampleRates(self._soapy.SOAPY_SDR_RX, 0)

		if supported:

			# Find the closest supported rate
			closest = min(supported, key=lambda r: abs(r - value))

			if abs(closest - value) > 1.0:
				logger.warning(
					f"Requested sample rate {value/1e6:.3f} MHz is not supported by this device. "
					f"Using nearest supported rate: {closest/1e6:.3f} MHz. "
					f"Supported: {', '.join(f'{r/1e6:.3f}' for r in supported)} MHz."
				)
				value = closest

		self._device.setSampleRate(self._soapy.SOAPY_SDR_RX, 0, value)
		self._sample_rate = value

	@property
	def center_freq (self) -> float | None:
		"""Get the current center frequency in Hz."""
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		"""Set the center frequency in Hz."""
		self._device.setFrequency(self._soapy.SOAPY_SDR_RX, 0, value)
		self._center_freq = value

	@property
	def gain (self) -> float | str | None:
		"""Get the current gain setting (dB, 'auto', or None)."""
		return self._gain

	@gain.setter
	def gain (self, value: float | str | None) -> None:

		"""
		Set the gain (dB, 'auto', or None).

		When 'auto' or None, enables AGC if supported by the hardware.
		When numeric, sets the overall gain and lets SoapySDR distribute
		it across the device's gain stages.
		"""

		if value == 'auto' or value is None:

			# AirSpy R2 special case.  SoapyAirspy reports hasGainMode(True)
			# but the underlying AirSpy R2 hardware does not provide a true
			# closed-loop AGC: enabling AGC mode leaves gain at 0 dB and
			# does not respond to incoming signal levels.  Real-world
			# symptom is "zero recordings overnight" because the receiver
			# is effectively muted.  Fall back to the README-recommended
			# manual per-element gain (LNA-first noise figure principle).
			# To override, set sdr_gain_db or sdr_gain_elements explicitly.
			if self._driver == 'airspy':
				logger.info(
					"AirSpy R2 'auto' gain requested, but the SoapyAirspy module's "
					"AGC mode is not effective.  Falling back to manual gain "
					"(LNA=10, MIX=5, VGA=12, 27 dB total).  Override with "
					"sdr_gain_db or sdr_gain_elements in config.yaml for finer control."
				)
				self._device.setGainMode(self._soapy.SOAPY_SDR_RX, 0, False)
				self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, 'LNA', 10.0)
				self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, 'MIX', 5.0)
				self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, 'VGA', 12.0)
				self._gain_elements = {'LNA': 10.0, 'MIX': 5.0, 'VGA': 12.0}
				self._gain = None
				return

			if self._device.hasGainMode(self._soapy.SOAPY_SDR_RX, 0):
				self._device.setGainMode(self._soapy.SOAPY_SDR_RX, 0, True)
				self._gain = 'auto'
			else:
				logger.warning(
					f"SoapySDR [{self._driver}] does not support AGC. "
					f"Setting overall gain to midpoint of available range. "
					f"Set sdr_gain_db or sdr_gain_elements in config.yaml for manual control."
				)
				gain_range = self._device.getGainRange(self._soapy.SOAPY_SDR_RX, 0)
				midpoint = (gain_range.minimum() + gain_range.maximum()) / 2.0
				self._device.setGainMode(self._soapy.SOAPY_SDR_RX, 0, False)
				self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, midpoint)
				self._gain = midpoint
		else:

			self._device.setGainMode(self._soapy.SOAPY_SDR_RX, 0, False)
			self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, float(value))
			self._gain = float(value)

	@property
	def gain_elements (self) -> dict[str, float] | None:
		"""Get the current per-element gain settings, or None if not set individually."""
		return self._gain_elements

	@gain_elements.setter
	def gain_elements (self, value: dict[str, float]) -> None:

		"""
		Set per-element gain for fine-tuned control of the device's gain stages.

		This allows optimising the noise figure by setting each gain stage
		independently (e.g., LNA, Mixer, VGA on AirSpy R2).

		Args:
			value: Mapping of gain element name to dB value.
				Element names are device-specific — check the log output
				at startup for available elements and their ranges.

		Raises:
			ValueError: If an element name is not recognised by the device.
		"""

		available = self._device.listGains(self._soapy.SOAPY_SDR_RX, 0)

		# Disable AGC before setting individual stages
		self._device.setGainMode(self._soapy.SOAPY_SDR_RX, 0, False)

		for element, db_value in value.items():

			if element not in available:
				raise ValueError(
					f"Unknown gain element '{element}' for driver '{self._driver}'. "
					f"Available elements: {', '.join(available)}"
				)

			self._device.setGain(self._soapy.SOAPY_SDR_RX, 0, element, float(db_value))
			logger.info(f"Gain element '{element}' set to {db_value:.1f} dB")

		self._gain_elements = dict(value)
		self._gain = None

	@property
	def device_settings (self) -> dict[str, str] | None:
		"""Get the currently applied device-specific settings."""
		return self._device_settings

	@device_settings.setter
	def device_settings (self, value: dict[str, str]) -> None:

		"""
		Apply device-specific settings via SoapySDR's settings API.

		This provides access to features like bias tee control, external
		clock configuration, and device-specific calibration options.

		Args:
			value: Mapping of setting key to string value.
				Available settings are logged at DEBUG level on device init.
		"""

		for key, setting_value in value.items():
			self._device.writeSetting(key, str(setting_value))
			logger.info(f"Device setting '{key}' = '{setting_value}'")

		self._device_settings = dict(value)

	def _negotiate_stream_format (self) -> str:

		"""
		Choose the best available stream format for maximum sample resolution.

		Prefers CF32 (complex float32) for full resolution from oversampled
		ADCs (e.g., AirSpy R2 achieves 16-bit effective from 12-bit ADC).
		Falls back to CS16 (complex int16) if CF32 is not available.

		Returns:
			SoapySDR format string (e.g., SOAPY_SDR_CF32 or SOAPY_SDR_CS16)
		"""

		supported = self._device.getStreamFormats(self._soapy.SOAPY_SDR_RX, 0)

		if self._soapy.SOAPY_SDR_CF32 in supported:
			logger.info(f"Stream format: CF32 (complex float32)")
			return self._soapy.SOAPY_SDR_CF32

		if self._soapy.SOAPY_SDR_CS16 in supported:
			logger.info(f"Stream format: CS16 (complex int16, will convert to float32)")
			return self._soapy.SOAPY_SDR_CS16

		# Last resort — use whatever the device offers natively
		native_fmt = self._device.getNativeStreamFormat(self._soapy.SOAPY_SDR_RX, 0, [0.0])
		logger.warning(f"Neither CF32 nor CS16 available. Using native format: {native_fmt}")
		return native_fmt

	def _calibrate_iq_scale (self, stream: typing.Any, stream_format: str) -> float:

		"""
		Measure the IQ sample scale from initial blocks and return a
		normalisation factor.

		Some SoapySDR modules deliver CF32 samples at a much smaller
		scale than the expected [-1, 1] range (e.g., AirSpy HF+ peaks
		around 0.001–0.005).  This method reads warmup blocks and
		measures the median RMS across blocks to determine the typical
		noise floor level.

		Using median RMS rather than peak amplitude makes the measurement
		robust against transient strong signals that may be present
		during calibration — a strong signal shifts the peak drastically
		but barely moves the median.

		The target is to bring the noise floor RMS to ~0.01 (typical for
		RTL-SDR), which leaves ample headroom for signals above the noise.

		Returns 1.0 if samples are already in a sensible range.
		"""

		is_cs16 = (stream_format == self._soapy.SOAPY_SDR_CS16)

		if is_cs16:
			buf = numpy.empty(65536 * 2, dtype=numpy.int16)
		else:
			buf = numpy.empty(65536, dtype=numpy.complex64)

		rms_values = []

		for _ in range(20):

			sr = self._device.readStream(stream, [buf], 65536, timeoutUs=500000)

			if sr.ret > 0:

				# Unify both code paths onto complex64 so the downstream RMS
				# calculation has a single, well-typed input.  asarray with
				# an explicit dtype is a no-op view when `buf` is already
				# complex64, and a conversion otherwise.
				samples: numpy.typing.NDArray[numpy.complex64]
				if is_cs16:
					samples = self._convert_cs16_to_complex64(buf, sr.ret)
				else:
					samples = numpy.asarray(buf[:sr.ret], dtype=numpy.complex64)

				block_rms = float(numpy.sqrt(numpy.mean(numpy.abs(samples) ** 2)))
				rms_values.append(block_rms)

				if len(rms_values) >= 10:
					break

		if not rms_values:
			logger.warning("IQ calibration: no samples received, using scale factor 1.0")
			return 1.0

		median_rms = float(numpy.median(rms_values))

		# Log the gain context so the normalisation is traceable.
		# This is a logging convenience — failures are surfaced at DEBUG
		# level so they're discoverable but don't break calibration.
		try:
			overall_gain = self._device.getGain(self._soapy.SOAPY_SDR_RX, 0)
			logger.info(f"IQ calibration: device overall gain {overall_gain:.1f} dB, median RMS {median_rms:.6f}")
		except Exception as exc:
			logger.debug(f"IQ calibration: could not read overall gain ({exc}); median RMS {median_rms:.6f}")

		if median_rms < 1e-10:
			logger.warning("IQ calibration: signal too weak to measure, using scale factor 1.0")
			return 1.0

		if median_rms > 0.001:
			# Samples are large enough for the demodulator to work correctly.
			# No normalisation needed — avoids overdriving the audio output.
			# RTL-SDR typically produces RMS ~0.01-0.05; values above 0.001
			# are well within float32 precision and demodulator range.
			# Demoted to DEBUG because the preceding "IQ calibration ..."
			# line already shows the median RMS, and no-op is the default
			# case — only the non-trivial scale path below is load-bearing
			# for interpreting downstream amplitudes (e.g. the ADC
			# saturation check).
			logger.debug(f"IQ sample scale: no normalisation needed (median RMS {median_rms:.6f})")
			return 1.0

		# Samples are very small (e.g., AirSpy HF+ with attenuation).
		# Scale so that the median noise floor RMS maps to ~0.01, which is
		# typical for RTL-SDR and produces good demodulated audio levels.
		# This branch stays at INFO because a non-trivial scale factor
		# changes the meaning of every subsequent sample amplitude.
		target_rms = 0.01
		scale = target_rms / median_rms
		logger.info(f"IQ sample scale: median RMS {median_rms:.6f} — applying {scale:.1f}x normalisation")
		return scale

	def _convert_cs16_to_complex64 (self, buf: numpy.ndarray, count: int) -> numpy.typing.NDArray[numpy.complex64]:

		"""
		Convert CS16 (interleaved int16 I/Q) samples to complex64.

		Preserves the full 16-bit dynamic range by normalising to [-1, 1].

		Args:
			buf: Raw buffer from readStream (int16 interleaved I/Q)
			count: Number of complex samples read

		Returns:
			Complex64 array of normalised IQ samples
		"""

		raw = buf[:count * 2]
		result = numpy.empty(count, dtype=numpy.complex64)
		result.real = raw[0::2].astype(numpy.float32) / 32768.0
		result.imag = raw[1::2].astype(numpy.float32) / 32768.0

		return result

	def _buffer_samples (self, samples: numpy.typing.NDArray[numpy.complex64], chunk_size: int, callback: typing.Callable) -> None:

		"""
		Rechunk variable-size SoapySDR blocks into fixed-size chunks for
		the scanner.  Delegates to the shared rechunk_samples helper so
		all device backends use identical boundary logic.
		"""

		self._rx_buffer = substation.devices.base.rechunk_samples(
			self._rx_buffer, samples, chunk_size, callback
		)

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:

		"""
		Start asynchronous sample streaming via an internal reader thread.

		Sets up a SoapySDR RX stream, negotiates the best format for sample
		resolution, and spawns a daemon thread that continuously reads
		samples and invokes the callback with fixed-size chunks.

		Args:
			callback: Function to call with samples (signature: callback(samples, context))
			num_samples: Number of samples to deliver per callback invocation
		"""

		# Negotiate stream format for best resolution
		self._stream_format = self._negotiate_stream_format()

		# Setup and activate the RX stream
		self._stream = self._device.setupStream(self._soapy.SOAPY_SDR_RX, self._stream_format)
		self._device.activateStream(self._stream)

		# Calibrate IQ scale from initial samples.  Some devices (e.g.,
		# AirSpy HF+) deliver CF32 at a much smaller scale than [-1, 1].
		# See _calibrate_iq_scale for the median-RMS measurement strategy
		# and the rationale for picking it over peak detection.
		self._iq_scale = self._calibrate_iq_scale(self._stream, self._stream_format)

		# Expose the calibration scale on the BaseDevice attribute so
		# upstream code (notably the ADC saturation check in scanner.py)
		# can interpret post-normalisation amplitudes correctly.
		self.iq_scale = self._iq_scale

		self._stop_event.clear()
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		# Capture state for the reader thread closure.  All device-related
		# state is captured up front so the closure does not depend on
		# attribute lookups through self while the thread is running.
		device = self._device
		stream = self._stream
		stream_format = self._stream_format
		is_cs16 = (stream_format == self._soapy.SOAPY_SDR_CS16)
		iq_scale = self._iq_scale

		def _reader_loop () -> None:

			"""
			Continuously read from SoapySDR stream and invoke callback.

			Handles SoapySDR error codes:
			- TIMEOUT: retry silently (normal during quiet periods)
			- OVERFLOW: log warning, continue (samples were lost but stream is fine)
			- Other negatives: log error, stop thread
			"""

			# Determine read buffer size — use MTU if larger than requested
			mtu = device.getStreamMTU(stream)
			read_size = max(mtu, num_samples)

			# Allocate read buffer according to negotiated format
			if is_cs16:
				buf = numpy.empty(read_size * 2, dtype=numpy.int16)
			else:
				buf = numpy.empty(read_size, dtype=numpy.complex64)

			while not self._stop_event.is_set():

				try:
					sr = device.readStream(stream, [buf], read_size, timeoutUs=500000)
				except Exception as exc:
					logger.error(f"SoapySDR readStream exception: {exc}")
					break

				if sr.ret > 0:

					# Convert format if needed, otherwise just slice and copy
					if is_cs16:
						samples = self._convert_cs16_to_complex64(buf, sr.ret)
					else:
						samples = buf[:sr.ret].copy()

					# Apply IQ normalisation (no-op when scale is 1.0)
					if iq_scale != 1.0:
						samples *= iq_scale

					self._buffer_samples(samples, num_samples, callback)

				elif sr.ret == self._soapy.SOAPY_SDR_TIMEOUT:
					# Normal during quiet periods — just retry
					continue

				elif sr.ret == self._soapy.SOAPY_SDR_OVERFLOW:
					logger.warning("SoapySDR: overflow detected (samples lost)")
					continue

				else:
					# Other negative return codes are fatal errors
					logger.error(f"SoapySDR readStream error: {sr.ret}")
					break

			# Signal the processing loop that streaming has ended
			logger.debug("SoapySDR reader thread exiting")

		self._reader_thread = threading.Thread(target=_reader_loop, daemon=True, name='soapy-reader')
		self._reader_thread.start()

	def read_samples (self, num_samples: int) -> numpy.typing.NDArray[numpy.complex64]:

		"""
		Synchronous sample read for calibration.

		Sets up a temporary stream, reads the requested number of samples,
		then tears down the stream. Used by the scanner's frequency
		calibration routine.

		The read is bounded by a wall-clock deadline so a stalled or
		disconnected device cannot hang startup indefinitely.

		Args:
			num_samples: Number of complex IQ samples to read

		Returns:
			Array of complex64 IQ samples

		Raises:
			RuntimeError: If the read does not complete within the deadline,
				or if SoapySDR returns a fatal error code.
		"""

		# Cap the calibration read at 10 seconds.  This is intentionally
		# generous — slow USB devices can take a few seconds to start
		# delivering samples — but bounded so a hung device fails fast
		# instead of blocking startup forever.
		deadline = time.monotonic() + 10.0

		fmt = self._negotiate_stream_format()
		is_cs16 = (fmt == self._soapy.SOAPY_SDR_CS16)

		stream = self._device.setupStream(self._soapy.SOAPY_SDR_RX, fmt)
		self._device.activateStream(stream)

		try:

			collected = numpy.empty(num_samples, dtype=numpy.complex64)
			offset = 0

			if is_cs16:
				buf = numpy.empty(num_samples * 2, dtype=numpy.int16)
			else:
				buf = numpy.empty(num_samples, dtype=numpy.complex64)

			while offset < num_samples:

				if time.monotonic() > deadline:
					raise RuntimeError(
						f"SoapySDR calibration read timed out after 10 s "
						f"({offset}/{num_samples} samples received)"
					)

				remaining = num_samples - offset
				sr = self._device.readStream(stream, [buf], remaining, timeoutUs=1000000)

				if sr.ret > 0:

					# Same unification trick as _calibrate_iq_scale: annotate
					# once as complex64 and let asarray do a no-op view when
					# buf is already complex64, a real conversion otherwise.
					chunk: numpy.typing.NDArray[numpy.complex64]
					if is_cs16:
						chunk = self._convert_cs16_to_complex64(buf, sr.ret)
					else:
						chunk = numpy.asarray(buf[:sr.ret], dtype=numpy.complex64).copy()

					collected[offset:offset + sr.ret] = chunk
					offset += sr.ret

				elif sr.ret == self._soapy.SOAPY_SDR_TIMEOUT:
					continue

				elif sr.ret < 0:
					raise RuntimeError(f"SoapySDR readStream error during calibration: {sr.ret}")

			return collected

		finally:
			self._device.deactivateStream(stream)
			self._device.closeStream(stream)

	def cancel_read_async (self) -> None:

		"""Cancel asynchronous sample reading and join the reader thread."""

		self._stop_event.set()

		if self._reader_thread and self._reader_thread.is_alive():
			self._reader_thread.join(timeout=3.0)

		self._reader_thread = None

	def close (self) -> None:

		"""
		Close the device and release all resources.

		Stops any active streaming, deactivates and closes the stream,
		and releases the SoapySDR device handle.
		"""

		try:
			self.cancel_read_async()
		except Exception as exc:
			logger.debug(f"Error cancelling async read during close: {exc}")

		if self._stream is not None:

			try:
				self._device.deactivateStream(self._stream)
				self._device.closeStream(self._stream)
			except Exception as exc:
				logger.warning(f"Error closing SoapySDR stream: {exc}")

			self._stream = None

		self._device = None
