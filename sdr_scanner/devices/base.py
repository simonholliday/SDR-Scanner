"""
Abstract base class for SDR device implementations
"""

import typing
import abc


class BaseDevice(abc.ABC):
	"""
	Abstract base class defining the interface for SDR devices

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
