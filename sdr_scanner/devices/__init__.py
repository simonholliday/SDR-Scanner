"""
Device abstraction layer for SDR hardware
"""

import typing


def create_device (device_type: str, device_index: int = 0) -> typing.Any:

	"""
	Factory function to create SDR device instances

	Args:
		device_type: Type of device ('rtlsdr', 'rtl', 'rtl-sdr', 'hackrf', 'hackrf-one', 'hackrfone')
		device_index: Device index for multi-device setups (default: 0)

	Returns:
		Device instance implementing the device interface

	Raises:
		ValueError: If device_type is not supported
	"""

	import sdr_scanner.devices.rtlsdr
	import sdr_scanner.devices.hackrf

	device_type_lower = device_type.lower()

	if device_type_lower in ('rtl', 'rtlsdr', 'rtl-sdr'):
		return sdr_scanner.devices.rtlsdr.RtlSdrDevice(device_index)

	if device_type_lower in ('hackrf', 'hackrf-one', 'hackrfone'):
		return sdr_scanner.devices.hackrf.HackRfDevice(device_index)

	raise ValueError(f"Unsupported device_type: {device_type}")
