"""
HackRF device implementation with multi-binding support.

HackRF One is a wideband SDR transceiver with more capabilities than RTL-SDR:
- Frequency range: 1 MHz - 6 GHz (full coverage)
- Sample rate: 2-20 MHz
- 8-bit ADC/DAC resolution
- Transmit and receive capable
- USB 2.0 interface

Unlike RTL-SDR which has one standard Python library, HackRF has multiple
competing Python bindings with different APIs. This implementation automatically
detects and adapts to whichever binding is installed:
- python_hackrf (most common)
- hackrf
- pyhackrf

The adapter layer maps different function names and calling conventions
to a unified internal interface.
"""

import importlib
import logging
import numpy
import numpy.typing
import typing

import sdr_scanner.devices.base

logger = logging.getLogger(__name__)


class HackRfDevice(sdr_scanner.devices.base.BaseDevice):
	"""
	Wrapper for HackRF devices with automatic binding detection.

	Unifies multiple possible Python bindings into a single interface.
	Different HackRF libraries use different function names and signatures,
	so we detect what's available and create an adapter layer.
	"""

	def __init__ (self, device_index: int = 0) -> None:
		"""
		Initialize HackRF device with auto-detection of Python bindings.

		Args:
			device_index: Index of the HackRF device to use (default: 0)
		"""
		# Try to import any available HackRF binding
		self._module = self._import_hackrf_module()
		self._device_index = device_index
		self._device: typing.Any = None
		self._initialized_library = False

		# Buffer for accumulating samples into fixed-size chunks
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)
		self._rx_wrapper: typing.Callable | None = None

		# Cache hardware state (HackRF doesn't provide getters for these)
		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain_db: float | None = None

		# Detect available functions and map them to our internal names
		self._setup_bindings()
		# Initialize library and open the device
		self._open_device()

	def _import_hackrf_module (self) -> typing.Any:
		"""
		Try to import any available HackRF Python binding.

		Tries multiple module paths in order of preference:
		1. python_hackrf: Most complete and maintained
		2. hackrf: Alternative binding
		3. pyhackrf: Older binding

		Returns the first successfully imported module.
		"""

		for module_name in ('python_hackrf.pylibhackrf.pyhackrf', 'hackrf', 'pyhackrf'):
			try:
				return importlib.import_module(module_name)
			except ImportError:
				continue

		raise RuntimeError("HackRF bindings not found. Please install 'python_hackrf'.")

	def _setup_bindings (self) -> None:
		"""
		Detect and map module-specific functions to a unified internal interface.

		Different HackRF bindings use different naming conventions:
		- python_hackrf uses: pyhackrf_set_sample_rate, pyhackrf_set_freq, etc.
		- Other bindings use: set_sample_rate, set_freq, etc.

		This method searches for each function by trying multiple names,
		storing the first match in our internal function dictionary.
		"""

		self._funcs = {}
		m = self._module

		# Map our internal function names to possible binding-specific names
		# Listed in order of preference (most common first)
		targets = {
			'set_sample_rate': ['pyhackrf_set_sample_rate', 'set_sample_rate'],
			'set_freq': ['pyhackrf_set_freq', 'set_freq', 'set_frequency'],
			'set_vga_gain': ['pyhackrf_set_vga_gain', 'set_vga_gain', 'set_gain'],
			'set_lna_gain': ['pyhackrf_set_lna_gain', 'set_lna_gain'],
			'start_rx': ['pyhackrf_start_rx', 'start_rx', 'start_rx_streaming'],
			'stop_rx': ['pyhackrf_stop_rx', 'stop_rx', 'stop_rx_streaming'],
			'close': ['pyhackrf_close', 'close']
		}

		# For each internal function name, try to find it in the module
		for key, names in targets.items():
			for name in names:
				func = getattr(m, name, None)

				if func:
					self._funcs[key] = func
					break

	def _call_safe (self, key: str, *args: typing.Any) -> None:
		"""
		Call a mapped function, handling both module-level and method signatures.

		Different HackRF bindings have different calling patterns:
		1. Module functions: func(device, *args) - e.g., pyhackrf_set_freq(device, freq)
		2. Device methods: device.func(*args) - e.g., device.set_freq(freq)
		3. Module functions without device: func(*args) - less common

		This method tries pattern #1 first, falls back to #2 or #3 if that fails.
		"""

		func = self._funcs.get(key)

		if not func:
			return

		try:
			# Try: module_function(device, *args)
			func(self._device, *args)
		except (TypeError, AttributeError):
			# Try: device.method(*args) or module_function(*args)
			method = getattr(self._device, key, None)

			if method:
				method(*args)
			else:
				func(*args)

	def _open_device (self) -> None:
		"""
		Initialize HackRF library and open device by index.

		Different bindings have different initialization sequences:
		1. python_hackrf: Call pyhackrf_init(), enumerate devices, open by serial
		2. Other bindings: Simply instantiate HackRF() object

		This method detects which pattern to use based on available functions.
		"""

		# Some bindings require explicit library initialization
		if hasattr(self._module, 'pyhackrf_init'):
			self._module.pyhackrf_init()
			self._initialized_library = True

		# Bindings with device enumeration support
		if hasattr(self._module, 'pyhackrf_device_list'):
			# Get list of all connected HackRF devices
			device_list = self._module.pyhackrf_device_list()

			if device_list.device_count == 0:
				raise RuntimeError("No HackRF devices found.")

			if self._device_index >= device_list.device_count:
				raise RuntimeError(f"Index {self._device_index} out of range ({device_list.device_count} found).")

			# Open specific device by serial number (more reliable than index)
			serial = device_list.serial_numbers[self._device_index]
			self._device = self._module.pyhackrf_open_by_serial(serial)

		# Simpler bindings: just instantiate the HackRF class
		elif hasattr(self._module, 'HackRF'):
			self._device = self._module.HackRF()

		if self._device is None:
			raise RuntimeError("Failed to open HackRF device.")

	@property
	def sample_rate (self) -> float | None:
		return self._sample_rate

	@sample_rate.setter
	def sample_rate (self, value: float) -> None:
		self._sample_rate = value
		self._call_safe('set_sample_rate', value)

	@property
	def center_freq (self) -> float | None:
		return self._center_freq

	@center_freq.setter
	def center_freq (self, value: float) -> None:
		self._center_freq = value
		self._call_safe('set_freq', int(value))

	@property
	def gain (self) -> float | None:
		"""Get the current gain setting (cached, HackRF doesn't provide getters)."""
		return self._gain_db

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		"""
		Set the receive gain.

		HackRF has two gain stages:
		- LNA (Low Noise Amplifier): 0-40 dB in 8 dB steps
		- VGA (Variable Gain Amplifier): 0-62 dB in 2 dB steps

		For simplicity, we set both to the same value. This isn't optimal
		(ideally you'd maximize LNA first, then use VGA for fine tuning),
		but it works for most applications.
		"""
		self._gain_db = None if value == 'auto' else float(value) if value is not None else None

		if self._gain_db is not None:
			gain_val = int(self._gain_db)
			# Set both gain stages to the same value
			self._call_safe('set_vga_gain', gain_val)
			self._call_safe('set_lna_gain', gain_val)

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:
		"""
		Start asynchronous sample streaming.

		HackRF delivers samples in variable-size blocks (typically 262144 bytes).
		We need to:
		1. Convert raw int8 samples to complex64
		2. Buffer and rechunk to the requested num_samples size
		3. Call the user callback with fixed-size blocks

		Different bindings pass samples differently, so the wrapper handles
		multiple calling patterns.
		"""

		# Clear any leftover samples from previous streaming session
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		def wrapper (*args: typing.Any) -> int:
			"""
			Callback invoked by HackRF library for each sample block.

			Different bindings pass arguments differently:
			- Some: wrapper(device, buffer) - 2 args
			- Some: wrapper(buffer) - 1 arg

			Returns 0 to continue streaming, non-zero to stop.
			"""
			try:
				# Extract buffer from args (handle both calling patterns)
				buffer_obj = args[1] if len(args) >= 2 else args[0]
				# Convert raw int8 IQ to complex64 normalized samples
				samples = self._convert_samples(buffer_obj)
				# Rechunk to requested size and call user callback
				self._buffer_samples(samples, num_samples, callback)
				return 0  # Continue streaming
			except Exception as e:
				logger.error(f"HackRF Callback Error: {e}")
				return 0  # Continue despite error (don't crash streaming)

		self._rx_wrapper = wrapper

		# Different bindings register callbacks differently
		if hasattr(self._device, 'set_rx_callback'):
			# Pattern 1: device.set_rx_callback(func), then start_rx()
			self._device.set_rx_callback(wrapper)
			self._call_safe('start_rx')
		else:
			# Pattern 2: start_rx(callback)
			self._call_safe('start_rx', wrapper)

	def cancel_read_async (self) -> None:
		self._call_safe('stop_rx')

	def close (self) -> None:
		self._call_safe('close')

		if self._initialized_library and hasattr(self._module, 'pyhackrf_exit'):
			self._module.pyhackrf_exit()
			self._initialized_library = False

	def _convert_samples (self, data: typing.Any) -> numpy.typing.NDArray[numpy.complex64]:
		"""
		Convert raw HackRF int8 samples to complex64 normalized to [-1, 1].

		HackRF sends samples as interleaved int8 pairs: [I0, Q0, I1, Q1, I2, Q2, ...]
		Each value is in the range [-128, 127].

		Conversion process:
		1. Parse as int8 array
		2. Reshape to (N, 2) for [I, Q] pairs
		3. Normalize to [-1, 1] range by dividing by 128
		4. Combine into complex numbers: I + jQ
		"""

		# Some bindings already provide complex samples
		if isinstance(data, numpy.ndarray) and numpy.iscomplexobj(data):
			return data.astype(numpy.complex64)

		# Convert buffer to numpy int8 array
		raw = numpy.frombuffer(data, dtype=numpy.int8) if not isinstance(data, numpy.ndarray) else data

		# Samples must come in I/Q pairs (even number of bytes)
		# Drop trailing byte if odd length (shouldn't happen but be defensive)
		if raw.size % 2 != 0:
			raw = raw[:-1]

		# Reshape from [I0, Q0, I1, Q1, ...] to [[I0, Q0], [I1, Q1], ...]
		iq = raw.astype(numpy.float32).reshape(-1, 2)

		# Normalize from [-128, 127] to approximately [-1, 1] and create complex samples
		# Formula: (I / 128) + j*(Q / 128)
		return ((iq[:, 0] / 128.0) + 1j * (iq[:, 1] / 128.0)).astype(numpy.complex64)

	def _buffer_samples (self, samples: numpy.typing.NDArray[numpy.complex64], chunk_size: int, callback: typing.Callable) -> None:
		"""
		Accumulate samples and emit fixed-size chunks to the callback.

		HackRF delivers variable-size blocks (e.g., 131072 samples), but the
		scanner expects fixed-size blocks (e.g., sdr_device_sample_size).

		This method:
		1. Concatenates new samples with any leftover from previous call
		2. Emits as many full chunks as possible
		3. Saves any remaining samples for next call

		Example: If chunk_size=50000 and we get 131072 samples:
		- Call 1: emit 50000, emit 50000, save 31072
		- Call 2: get 131072, combine with 31072 (162144 total)
		          emit 50000, emit 50000, emit 50000, save 12144
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
