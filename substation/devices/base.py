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

Also provides shared utilities used by multiple device wrappers.
"""

import abc
import typing

import numpy
import numpy.typing


def rechunk_samples (
	rx_buffer: numpy.typing.NDArray[numpy.complex64],
	samples: numpy.typing.NDArray[numpy.complex64],
	chunk_size: int,
	callback: typing.Callable,
) -> numpy.typing.NDArray[numpy.complex64]:

	"""
	Accumulate samples and emit fixed-size chunks to a callback.

	SDR backends like HackRF and SoapySDR deliver variable-size blocks,
	but the scanner expects fixed-size blocks (sdr_device_sample_size).
	This helper carries leftover samples between calls so the boundary
	logic is identical for every backend.

	1. Concatenate any leftover from a previous call with the new samples
	2. Emit as many full chunks as possible to the callback
	3. Return any remaining samples to be carried into the next call

	When chunk_size is zero or negative the samples are forwarded directly
	to the callback without rechunking, and the leftover buffer is reset.

	Args:
		rx_buffer: Leftover samples from the previous call (may be empty)
		samples: Newly received samples to be rechunked
		chunk_size: Target chunk size; pass 0 or less to disable rechunking
		callback: Function invoked as callback(chunk, None) for each chunk

	Returns:
		The new leftover buffer to be passed back on the next call.
	"""

	if chunk_size <= 0:
		callback(samples, None)
		return numpy.array([], dtype=numpy.complex64)

	combined = numpy.concatenate((rx_buffer, samples)) if rx_buffer.size > 0 else samples

	num_chunks = combined.size // chunk_size

	for i in range(num_chunks):
		start, end = i * chunk_size, (i + 1) * chunk_size
		callback(combined[start:end], None)

	leftover = combined.size % chunk_size

	if leftover > 0:
		return combined[-leftover:]

	return numpy.array([], dtype=numpy.complex64)


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

	# Multiplicative factor that the wrapper applies to raw IQ samples
	# before delivering them to the scanner.  Defaults to 1.0 (no scaling).
	# The SoapySDR wrapper sets this to whatever calibration factor it
	# computed at startup, so the scanner can interpret amplitudes
	# correctly — for example, the ADC saturation check needs to know
	# whether a 0.95-amplitude post-wrapper sample corresponds to a
	# real 0.95-amplitude raw sample (iq_scale==1.0) or to a much
	# smaller raw sample that has been amplified during normalisation.
	iq_scale: float = 1.0

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
