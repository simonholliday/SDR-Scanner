"""
Abstract base class for SDR device implementations.

Defines a unified interface for different SDR hardware (RTL-SDR, HackRF, etc.).
All device implementations must provide methods for:
- Setting sample rate and center frequency
- Configuring gain (manual or automatic)
- Reading IQ samples asynchronously via callbacks
- Resource cleanup

This abstraction allows the scanner to work with different hardware
without knowing the specific device type.
"""

import abc
import typing


class BaseDevice (abc.ABC):
	"""
	Abstract base class defining the interface for SDR devices.

	This class uses Python's ABC (Abstract Base Class) pattern to enforce
	a consistent interface across different SDR hardware implementations.
	Subclasses must implement all abstract methods and properties.

	The interface is designed around asynchronous sample reading:
	- Set hardware parameters (frequency, sample rate, gain)
	- Start async streaming with a callback function
	- Hardware continuously calls the callback with IQ sample blocks
	- Cancel streaming when done
	- Clean up resources

	All device implementations must provide these properties and methods.
	"""

	@property
	@abc.abstractmethod
	def sample_rate (self) -> float | None:
		"""Get the current sample rate in Hz"""
		pass

	@sample_rate.setter
	@abc.abstractmethod
	def sample_rate (self, value: float) -> None:
		"""Set the sample rate in Hz"""
		pass

	@property
	@abc.abstractmethod
	def center_freq (self) -> float | None:
		"""Get the current center frequency in Hz"""
		pass

	@center_freq.setter
	@abc.abstractmethod
	def center_freq (self, value: float) -> None:
		"""Set the center frequency in Hz"""
		pass

	@property
	@abc.abstractmethod
	def gain (self) -> float | str | None:
		"""Get the current gain setting (dB, 'auto', or None)"""
		pass

	@gain.setter
	@abc.abstractmethod
	def gain (self, value: float | str | None) -> None:
		"""Set the gain (dB, 'auto', or None)"""
		pass

	@abc.abstractmethod
	def read_samples_async (self, callback: typing.Callable, num_samples: int) -> None:
		"""
		Start asynchronous sample reading

		Args:
			callback: Function to call with samples (signature: callback(samples, context))
			num_samples: Number of samples to read per callback
		"""
		pass

	@abc.abstractmethod
	def cancel_read_async (self) -> None:
		"""Cancel asynchronous sample reading"""
		pass

	@abc.abstractmethod
	def close (self) -> None:
		"""Close the device and release resources"""
		pass
