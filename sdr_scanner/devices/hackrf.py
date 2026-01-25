"""
HackRF device implementation
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
	Wrapper for HackRF devices.
	Unifies multiple possible Python bindings into a single interface.
	"""

	def __init__ (self, device_index: int = 0) -> None:
		self._module = self._import_hackrf_module()
		self._device_index = device_index
		self._device: typing.Any = None
		self._initialized_library = False
		self._rx_buffer = numpy.array([], dtype=numpy.complex64)
		self._rx_wrapper: typing.Callable | None = None

		# State cache
		self._sample_rate: float | None = None
		self._center_freq: float | None = None
		self._gain_db: float | None = None

		# Bind methods based on detected module capabilities
		self._setup_bindings()
		self._open_device()

	def _import_hackrf_module (self) -> typing.Any:

		"""Try to import any available HackRF binding."""

		for module_name in ('python_hackrf.pylibhackrf.pyhackrf', 'hackrf', 'pyhackrf'):
			try:
				return importlib.import_module(module_name)
			except ImportError:
				continue

		raise RuntimeError("HackRF bindings not found. Please install 'python_hackrf'.")

	def _setup_bindings (self) -> None:

		"""Detect and map module-specific functions to a unified internal interface."""

		self._funcs = {}
		m = self._module

		targets = {
			'set_sample_rate': ['pyhackrf_set_sample_rate', 'set_sample_rate'],
			'set_freq': ['pyhackrf_set_freq', 'set_freq', 'set_frequency'],
			'set_vga_gain': ['pyhackrf_set_vga_gain', 'set_vga_gain', 'set_gain'],
			'set_lna_gain': ['pyhackrf_set_lna_gain', 'set_lna_gain'],
			'start_rx': ['pyhackrf_start_rx', 'start_rx', 'start_rx_streaming'],
			'stop_rx': ['pyhackrf_stop_rx', 'stop_rx', 'stop_rx_streaming'],
			'close': ['pyhackrf_close', 'close']
		}

		for key, names in targets.items():
			for name in names:
				func = getattr(m, name, None)

				if func:
					self._funcs[key] = func
					break

	def _call_safe (self, key: str, *args: typing.Any) -> None:

		"""Call a mapped function, handling both (device, *args) and (*args) signatures."""

		func = self._funcs.get(key)

		if not func:
			return

		try:
			func(self._device, *args)
		except (TypeError, AttributeError):
			method = getattr(self._device, key, None)

			if method:
				method(*args)
			else:
				func(*args)

	def _open_device (self) -> None:

		"""Initialize library and open device by index."""

		if hasattr(self._module, 'pyhackrf_init'):
			self._module.pyhackrf_init()
			self._initialized_library = True

		if hasattr(self._module, 'pyhackrf_device_list'):
			device_list = self._module.pyhackrf_device_list()

			if device_list.device_count == 0:
				raise RuntimeError("No HackRF devices found.")

			if self._device_index >= device_list.device_count:
				raise RuntimeError(f"Index {self._device_index} out of range ({device_list.device_count} found).")

			serial = device_list.serial_numbers[self._device_index]
			self._device = self._module.pyhackrf_open_by_serial(serial)

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
		return self._gain_db

	@gain.setter
	def gain (self, value: float | str | None) -> None:
		self._gain_db = None if value == 'auto' else float(value) if value is not None else None

		if self._gain_db is not None:
			gain_val = int(self._gain_db)
			self._call_safe('set_vga_gain', gain_val)
			self._call_safe('set_lna_gain', gain_val)

	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:

		"""Start asynchronous streaming."""

		self._rx_buffer = numpy.array([], dtype=numpy.complex64)

		def wrapper (*args: typing.Any) -> int:
			try:
				buffer_obj = args[1] if len(args) >= 2 else args[0]
				samples = self._convert_samples(buffer_obj)
				self._buffer_samples(samples, num_samples, callback)
				return 0
			except Exception as e:
				logger.error(f"HackRF Callback Error: {e}")
				return 0

		self._rx_wrapper = wrapper

		if hasattr(self._device, 'set_rx_callback'):
			self._device.set_rx_callback(wrapper)
			self._call_safe('start_rx')
		else:
			self._call_safe('start_rx', wrapper)

	def cancel_read_async (self) -> None:
		self._call_safe('stop_rx')

	def close (self) -> None:
		self._call_safe('close')

		if self._initialized_library and hasattr(self._module, 'pyhackrf_exit'):
			self._module.pyhackrf_exit()
			self._initialized_library = False

	def _convert_samples (self, data: typing.Any) -> numpy.typing.NDArray[numpy.complex64]:

		"""Convert raw HackRF int8 samples to complex64 normalized to [-1, 1]."""

		if isinstance(data, numpy.ndarray) and numpy.iscomplexobj(data):
			return data.astype(numpy.complex64)

		raw = numpy.frombuffer(data, dtype=numpy.int8) if not isinstance(data, numpy.ndarray) else data

		if raw.size % 2 != 0:
			raw = raw[:-1]

		iq = raw.astype(numpy.float32).reshape(-1, 2)

		return ((iq[:, 0] / 128.0) + 1j * (iq[:, 1] / 128.0)).astype(numpy.complex64)

	def _buffer_samples (self, samples: numpy.typing.NDArray[numpy.complex64], chunk_size: int, callback: typing.Callable) -> None:

		"""Accumulate samples and emit fixed-size chunks to the callback."""

		if chunk_size <= 0:
			callback(samples, None)
			return

		combined = numpy.concatenate((self._rx_buffer, samples)) if self._rx_buffer.size > 0 else samples
		num_chunks = combined.size // chunk_size

		for i in range(num_chunks):
			start, end = i * chunk_size, (i + 1) * chunk_size
			callback(combined[start:end], None)

		leftover = combined.size % chunk_size
		self._rx_buffer = combined[-leftover:] if leftover > 0 else numpy.array([], dtype=numpy.complex64)
