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
import typing

import numpy
import numpy.typing

import sdr_scanner.devices.base

logger = logging.getLogger(__name__)


class SoapySdrDevice (sdr_scanner.devices.base.BaseDevice):

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

		# Stream management
		self._stream: typing.Any = None
		self._stream_format: str | None = None

		# Reader thread control
		self._reader_thread: threading.Thread | None = None
		self._stop_event = threading.Event()
		self._rx_buffer: numpy.typing.NDArray[numpy.complex64] = numpy.array([], dtype=numpy.complex64)

		# Log device capabilities for user configuration
		self._log_device_capabilities()

	def _log_device_capabilities (self) -> None:

		"""
		Log available device capabilities at INFO/DEBUG level.

		This information helps users tune their configuration for best
		reception quality — gain element names and ranges, supported sample
		rates, antennas, stream formats, and device-specific settings.
		"""

		hw_info = self._device.getHardwareInfo()

		if hw_info:
			logger.info(f"SoapySDR [{self._driver}]: {dict(hw_info)}")

		# Gain elements and ranges — critical for optimal noise figure

		gain_elements = self._device.listGains(self._soapy.SOAPY_SDR_RX, 0)

		if gain_elements:
			parts = []
			for element in gain_elements:
				gain_range = self._device.getGainRange(self._soapy.SOAPY_SDR_RX, 0, element)
				parts.append(f"{element}: {gain_range.minimum():.0f}–{gain_range.maximum():.0f} dB")
			logger.info(f"Gain elements: {', '.join(parts)}")

		# Overall gain range

		overall_range = self._device.getGainRange(self._soapy.SOAPY_SDR_RX, 0)
		logger.info(f"Overall gain range: {overall_range.minimum():.0f}–{overall_range.maximum():.0f} dB")

		# AGC support

		has_agc = self._device.hasGainMode(self._soapy.SOAPY_SDR_RX, 0)
		logger.info(f"Automatic gain control: {'supported' if has_agc else 'not supported'}")

		# Sample rates

		sample_rates = self._device.listSampleRates(self._soapy.SOAPY_SDR_RX, 0)

		if sample_rates:
			rates_str = ', '.join(f"{r / 1e6:.3f} MHz" for r in sample_rates)
			logger.info(f"Supported sample rates: {rates_str}")

		# Antennas

		antennas = self._device.listAntennas(self._soapy.SOAPY_SDR_RX, 0)

		if antennas:
			logger.info(f"Antennas: {', '.join(antennas)}")

		# Stream formats (for debugging resolution issues)

		native_fmt, native_fullscale = self._device.getNativeStreamFormat(self._soapy.SOAPY_SDR_RX, 0)
		supported_fmts = self._device.getStreamFormats(self._soapy.SOAPY_SDR_RX, 0)
		logger.info(f"Native format: {native_fmt} (full scale: {native_fullscale:.0f}), supported: {', '.join(supported_fmts)}")

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
		"""Set the sample rate in Hz."""
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
		return self._gain_elements if hasattr(self, '_gain_elements') else None

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
		return self._device_settings if hasattr(self, '_device_settings') else None

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
		native_fmt, _ = self._device.getNativeStreamFormat(self._soapy.SOAPY_SDR_RX, 0)
		logger.warning(f"Neither CF32 nor CS16 available. Using native format: {native_fmt}")
		return native_fmt

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
		Accumulate samples and emit fixed-size chunks to the callback.

		SoapySDR's readStream may deliver variable-size blocks. The scanner
		expects fixed-size blocks (sdr_device_sample_size), so this method
		buffers and rechunks.

		This method:
		1. Concatenates new samples with any leftover from previous call
		2. Emits as many full chunks as possible
		3. Saves any remaining samples for next call
		"""

		# If chunk_size is 0 or negative, just pass through directly
		if chunk_size <= 0:
			callback(samples, None)
			return

		# Combine leftover from previous call with new samples
		combined = numpy.concatenate((self._rx_buffer, samples)) if self._rx_buffer.size > 0 else samples

		# Calculate how many full chunks we can emit
		num_chunks = combined.size // chunk_size

		# Emit each full chunk
		for i in range(num_chunks):
			start, end = i * chunk_size, (i + 1) * chunk_size
			callback(combined[start:end], None)

		# Save any remaining samples for next call
		leftover = combined.size % chunk_size
		self._rx_buffer = combined[-leftover:] if leftover > 0 else numpy.array([], dtype=numpy.complex64)

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

		self._stop_event.clear()
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		# Capture state for the reader thread closure
		stream = self._stream
		stream_format = self._stream_format
		is_cs16 = (stream_format == self._soapy.SOAPY_SDR_CS16)

		def _reader_loop () -> None:

			"""
			Continuously read from SoapySDR stream and invoke callback.

			Handles SoapySDR error codes:
			- TIMEOUT: retry silently (normal during quiet periods)
			- OVERFLOW: log warning, continue (samples were lost but stream is fine)
			- Other negatives: log error, stop thread
			"""

			# Determine read buffer size — use MTU if larger than requested
			mtu = self._device.getStreamMTU(stream)
			read_size = max(mtu, num_samples)

			# Allocate read buffer according to negotiated format
			if is_cs16:
				buf = numpy.empty(read_size * 2, dtype=numpy.int16)
			else:
				buf = numpy.empty(read_size, dtype=numpy.complex64)

			while not self._stop_event.is_set():

				try:
					sr = self._device.readStream(stream, [buf], read_size, timeoutUs=500000)
				except Exception as exc:
					logger.error(f"SoapySDR readStream exception: {exc}")
					break

				if sr.ret > 0:

					# Convert format if needed, otherwise just slice and copy
					if is_cs16:
						samples = self._convert_cs16_to_complex64(buf, sr.ret)
					else:
						samples = buf[:sr.ret].copy()

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

		Args:
			num_samples: Number of complex IQ samples to read

		Returns:
			Array of complex64 IQ samples
		"""

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
				remaining = num_samples - offset
				sr = self._device.readStream(stream, [buf], remaining, timeoutUs=1000000)

				if sr.ret > 0:

					if is_cs16:
						chunk = self._convert_cs16_to_complex64(buf, sr.ret)
					else:
						chunk = buf[:sr.ret].copy()

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
		except Exception:
			pass

		if self._stream is not None:

			try:
				self._device.deactivateStream(self._stream)
				self._device.closeStream(self._stream)
			except Exception as exc:
				logger.warning(f"Error closing SoapySDR stream: {exc}")

			self._stream = None

		self._device = None
