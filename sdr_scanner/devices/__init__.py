"""
Device abstraction layer for SDR hardware.

Provides a unified interface for different SDR devices through the
BaseDevice abstract class. Specific implementations handle the details
of RTL-SDR, HackRF, and potentially other hardware.

The factory function create_device() simplifies device instantiation
by accepting a device type string and returning the appropriate instance.
"""

from __future__ import annotations

import typing

import sdr_scanner.devices.base


def create_device (device_type: str, device_index: int = 0) -> sdr_scanner.devices.base.BaseDevice:
	"""
	Factory function to create SDR device instances.

	This function provides a simple way to instantiate SDR devices without
	needing to know the specific class names. It accepts various common
	name variations for each device type.

	Supported devices:
	- RTL-SDR (aliases: 'rtl', 'rtlsdr', 'rtl-sdr')
	  Low-cost receiver, 24 MHz - 1.7 GHz, up to 2.4 MHz sample rate

	- HackRF (aliases: 'hackrf', 'hackrf-one', 'hackrfone')
	  Wideband transceiver, 1 MHz - 6 GHz, 2-20 MHz sample rate

	Args:
		device_type: Type of device (case-insensitive, accepts aliases)
		device_index: Device index for multi-device setups (default: 0)

	Returns:
		Device instance implementing the BaseDevice interface

	Raises:
		ValueError: If device_type is not recognized
		RuntimeError: If device cannot be opened (hardware not found, etc.)

	Example:
		device = create_device('rtlsdr')
		device.sample_rate = 2048000
		device.center_freq = 446000000
		device.gain = 'auto'
	"""

	device_type_lower = device_type.lower()

	# Lazy imports: only load the binding for the requested device type.
	if device_type_lower in ('rtl', 'rtlsdr', 'rtl-sdr'):
		import sdr_scanner.devices.rtlsdr
		return sdr_scanner.devices.rtlsdr.RtlSdrDevice(device_index)

	if device_type_lower in ('hackrf', 'hackrf-one', 'hackrfone'):
		import sdr_scanner.devices.hackrf
		return sdr_scanner.devices.hackrf.HackRfDevice(device_index)

	raise ValueError(f"Unsupported device_type: {device_type}")
