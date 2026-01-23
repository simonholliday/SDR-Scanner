"""
RTL-SDR device implementation
"""

import rtlsdr
import typing


from .base import BaseDevice


class RtlSdrDevice(BaseDevice):
	"""
	Wrapper for RTL-SDR devices

	Provides a unified interface for RTL-SDR hardware.
	"""

	def __init__(self, device_index: int = 0) -> None:
		"""
		Initialize RTL-SDR device

		Args:
			device_index: Index of the RTL-SDR device to use (default: 0)
		"""
		self._device_index = device_index
		self._device = rtlsdr.RtlSdr(device_index)

	@property
	def sample_rate(self) -> float:
		"""Get the current sample rate in Hz"""
		return self._device.sample_rate

	@sample_rate.setter
	def sample_rate(self, value: float) -> None:
		"""Set the sample rate in Hz"""
		self._device.sample_rate = value

	@property
	def center_freq(self) -> float:
		"""Get the current center frequency in Hz"""
		return self._device.center_freq

	@center_freq.setter
	def center_freq(self, value: float) -> None:
		"""Set the center frequency in Hz"""
		self._device.center_freq = value

	@property
	def gain(self) -> float | str | None:
		"""Get the current gain setting (dB, 'auto', or None)"""
		return self._device.gain

	@gain.setter
	def gain(self, value: float | str | None) -> None:
		"""Set the gain (dB, 'auto', or None)"""
		self._device.gain = value

	@property
	def freq_correction(self) -> int:
		"""Get the current frequency correction in PPM"""
		return self._device.freq_correction

	@freq_correction.setter
	def freq_correction(self, value: int) -> None:
		"""Set the frequency correction in PPM"""
		self._device.freq_correction = value

	@property
	def serial(self) -> str | None:
		"""Get the device serial number if available."""
		serial = None
		try:
			serials = rtlsdr.RtlSdr.get_device_serial_addresses()
			if 0 <= self._device_index < len(serials):
				serial = serials[self._device_index]
		except Exception:
			serial = None

		if isinstance(serial, bytes):
			serial = serial.decode('ascii', errors='replace')

		if isinstance(serial, str):
			serial = serial.strip()
			return serial if serial else None

		return None

	def read_samples(self, num_samples: int) -> typing.Any:
		"""
		Read samples synchronously

		Args:
			num_samples: Number of samples to read

		Returns:
			Complex IQ samples as numpy array
		"""
		return self._device.read_samples(num_samples)

	def read_samples_async(self, callback: typing.Callable, num_samples: int) -> None:
		"""
		Start asynchronous sample reading

		Args:
			callback: Function to call with samples (signature: callback(samples, context))
			num_samples: Number of samples to read per callback
		"""
		self._device.read_samples_async(callback, num_samples)

	def cancel_read_async(self) -> None:
		"""Cancel asynchronous sample reading"""
		self._device.cancel_read_async()

	def close(self) -> None:
		"""Close the device and release resources"""
		self._device.close()
