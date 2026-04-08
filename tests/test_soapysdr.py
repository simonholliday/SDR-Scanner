"""Tests for the SoapySDR device wrapper."""

import threading
import types
import unittest.mock

import numpy
import pytest


def _make_mock_soapy_module () -> types.ModuleType:

	"""
	Create a mock SoapySDR module with the constants and classes
	needed by SoapySdrDevice.
	"""

	mod = types.ModuleType('SoapySDR')

	# Stream direction and format constants
	mod.SOAPY_SDR_RX = 0
	mod.SOAPY_SDR_CF32 = 'CF32'
	mod.SOAPY_SDR_CS16 = 'CS16'

	# Error code constants
	mod.SOAPY_SDR_TIMEOUT = -1
	mod.SOAPY_SDR_OVERFLOW = -4

	return mod


def _make_mock_device () -> unittest.mock.MagicMock:

	"""Create a mock SoapySDR.Device with sensible defaults."""

	device = unittest.mock.MagicMock()

	# Hardware info
	device.getHardwareInfo.return_value = {'driver': 'airspy', 'label': 'AIRSPY'}

	# Gain element info
	gain_range = unittest.mock.MagicMock()
	gain_range.minimum.return_value = 0.0
	gain_range.maximum.return_value = 40.0

	device.listGains.return_value = ['LNA', 'MIX', 'VGA']
	device.getGainRange.return_value = gain_range
	device.hasGainMode.return_value = True

	# Sample rates and antennas
	device.listSampleRates.return_value = [2.5e6, 10e6]
	device.listAntennas.return_value = ['RX']
	device.getNativeStreamFormat.return_value = ('CF32', 4096.0)
	device.getStreamFormats.return_value = ['CF32', 'CS16']
	device.getSettingInfo.return_value = []

	# Stream setup
	device.setupStream.return_value = 'mock_stream'
	device.getStreamMTU.return_value = 65536

	# Enumerate returns a list with one device
	device.enumerate = unittest.mock.MagicMock(return_value=[{'driver': 'airspy'}])

	return device


def _create_soapy_device (driver: str = 'airspy', device_index: int = 0) -> tuple:

	"""
	Create a SoapySdrDevice with a fully mocked SoapySDR backend.

	Returns:
		(soapy_device, mock_device, mock_soapy_module)
	"""

	import sys

	mock_soapy = _make_mock_soapy_module()
	mock_device = _make_mock_device()

	# SoapySDR.Device is both a class (constructor) and has enumerate
	mock_device_cls = unittest.mock.MagicMock(return_value=mock_device)
	mock_device_cls.enumerate = unittest.mock.MagicMock(return_value=[{'driver': driver}])
	mock_soapy.Device = mock_device_cls

	with unittest.mock.patch.dict(sys.modules, {'SoapySDR': mock_soapy}):

		import sdr_scanner.devices.soapysdr
		sdr_device = sdr_scanner.devices.soapysdr.SoapySdrDevice(driver, device_index)

	return sdr_device, mock_device, mock_soapy


class TestSoapySdrInit:

	def test_no_devices_raises (self):

		"""Enumerate returning empty list should raise RuntimeError."""

		import sys

		mock_soapy = _make_mock_soapy_module()
		mock_device_cls = unittest.mock.MagicMock()
		mock_device_cls.enumerate = unittest.mock.MagicMock(return_value=[])
		mock_soapy.Device = mock_device_cls

		with unittest.mock.patch.dict(sys.modules, {'SoapySDR': mock_soapy}):

			import sdr_scanner.devices.soapysdr
			with pytest.raises(RuntimeError, match="No SoapySDR devices found"):
				sdr_scanner.devices.soapysdr.SoapySdrDevice('airspy')

	def test_index_out_of_range_raises (self):

		"""Device index beyond enumerated count should raise RuntimeError."""

		import sys

		mock_soapy = _make_mock_soapy_module()
		mock_device_cls = unittest.mock.MagicMock()
		mock_device_cls.enumerate = unittest.mock.MagicMock(return_value=[{'driver': 'airspy'}])
		mock_soapy.Device = mock_device_cls

		with unittest.mock.patch.dict(sys.modules, {'SoapySDR': mock_soapy}):

			import sdr_scanner.devices.soapysdr
			with pytest.raises(RuntimeError, match="out of range"):
				sdr_scanner.devices.soapysdr.SoapySdrDevice('airspy', device_index=5)

	def test_capabilities_logged (self):

		"""Init should log device capabilities without error."""

		sdr_device, mock_device, _ = _create_soapy_device()

		# Verify capability queries were called
		mock_device.getHardwareInfo.assert_called_once()
		mock_device.listGains.assert_called()
		mock_device.listSampleRates.assert_called()
		mock_device.listAntennas.assert_called()
		mock_device.getNativeStreamFormat.assert_called()
		mock_device.getStreamFormats.assert_called()


class TestSoapySdrProperties:

	def test_sample_rate (self):

		"""Setting sample_rate should call setSampleRate on the device."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.sample_rate = 10e6
		mock_device.setSampleRate.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, 10e6)
		assert sdr_device.sample_rate == 10e6

	def test_center_freq (self):

		"""Setting center_freq should call setFrequency on the device."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.center_freq = 446e6
		mock_device.setFrequency.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, 446e6)
		assert sdr_device.center_freq == 446e6

	def test_gain_auto (self):

		"""Setting gain to 'auto' should enable AGC."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.gain = 'auto'
		mock_device.setGainMode.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, True)
		assert sdr_device.gain == 'auto'

	def test_gain_numeric (self):

		"""Setting gain to a number should disable AGC and set overall gain."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.gain = 30.0
		mock_device.setGainMode.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, False)
		mock_device.setGain.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, 30.0)
		assert sdr_device.gain == 30.0

	def test_gain_auto_no_agc_fallback (self):

		"""If device doesn't support AGC, should fall back to midpoint gain."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()
		mock_device.hasGainMode.return_value = False

		sdr_device.gain = 'auto'

		# Should set overall gain to midpoint (0 + 40) / 2 = 20
		mock_device.setGain.assert_called_with(mock_soapy.SOAPY_SDR_RX, 0, 20.0)
		assert sdr_device.gain == 20.0


class TestSoapySdrGainElements:

	def test_per_element_gain (self):

		"""Setting gain_elements should set each element individually."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.gain_elements = {'LNA': 10.0, 'MIX': 5.0, 'VGA': 12.0}

		calls = mock_device.setGain.call_args_list
		expected = [
			unittest.mock.call(mock_soapy.SOAPY_SDR_RX, 0, 'LNA', 10.0),
			unittest.mock.call(mock_soapy.SOAPY_SDR_RX, 0, 'MIX', 5.0),
			unittest.mock.call(mock_soapy.SOAPY_SDR_RX, 0, 'VGA', 12.0),
		]
		assert calls == expected

	def test_unknown_element_raises (self):

		"""Unknown gain element name should raise ValueError."""

		sdr_device, mock_device, _ = _create_soapy_device()

		with pytest.raises(ValueError, match="Unknown gain element 'BOGUS'"):
			sdr_device.gain_elements = {'BOGUS': 10.0}

	def test_agc_disabled_before_per_element (self):

		"""AGC should be disabled before setting per-element gains."""

		sdr_device, mock_device, mock_soapy = _create_soapy_device()

		sdr_device.gain_elements = {'LNA': 10.0}

		# setGainMode(False) should come before setGain per-element
		all_calls = mock_device.method_calls
		agc_off_idx = next(i for i, c in enumerate(all_calls) if c[0] == 'setGainMode' and c[1] == (mock_soapy.SOAPY_SDR_RX, 0, False))
		set_gain_idx = next(i for i, c in enumerate(all_calls) if c[0] == 'setGain' and len(c[1]) == 4)
		assert agc_off_idx < set_gain_idx


class TestSoapySdrDeviceSettings:

	def test_device_settings (self):

		"""Setting device_settings should call writeSetting for each pair."""

		sdr_device, mock_device, _ = _create_soapy_device()

		sdr_device.device_settings = {'biastee': 'true', 'clock_source': 'external'}

		calls = mock_device.writeSetting.call_args_list
		assert unittest.mock.call('biastee', 'true') in calls
		assert unittest.mock.call('clock_source', 'external') in calls


class TestSoapySdrStreamFormat:

	def test_prefers_cf32 (self):

		"""Should choose CF32 when available."""

		sdr_device, mock_device, _ = _create_soapy_device()

		fmt = sdr_device._negotiate_stream_format()
		assert fmt == 'CF32'

	def test_falls_back_to_cs16 (self):

		"""Should fall back to CS16 when CF32 is not available."""

		sdr_device, mock_device, _ = _create_soapy_device()
		mock_device.getStreamFormats.return_value = ['CS16']

		fmt = sdr_device._negotiate_stream_format()
		assert fmt == 'CS16'


class TestSoapySdrCS16Conversion:

	def test_cs16_to_complex64 (self):

		"""CS16 int16 I/Q pairs should convert to normalised complex64."""

		sdr_device, _, _ = _create_soapy_device()

		# Two complex samples: (32767, -32768) and (0, 16384)
		raw = numpy.array([32767, -32768, 0, 16384], dtype=numpy.int16)
		result = sdr_device._convert_cs16_to_complex64(raw, 2)

		assert result.dtype == numpy.complex64
		assert len(result) == 2
		assert result[0].real == pytest.approx(32767.0 / 32768.0, abs=1e-4)
		assert result[0].imag == pytest.approx(-1.0, abs=1e-4)
		assert result[1].real == pytest.approx(0.0, abs=1e-4)
		assert result[1].imag == pytest.approx(16384.0 / 32768.0, abs=1e-4)


class TestSoapySdrBufferSamples:

	def test_rechunking (self):

		"""Buffer should accumulate and emit fixed-size chunks."""

		sdr_device, _, _ = _create_soapy_device()
		sdr_device._rx_buffer = numpy.array([], dtype=numpy.complex64)

		emitted = []
		callback = lambda samples, ctx: emitted.append(samples.copy())

		# Send 150 samples with chunk size 100
		samples = numpy.ones(150, dtype=numpy.complex64)
		sdr_device._buffer_samples(samples, 100, callback)

		# Should emit 1 chunk of 100, leaving 50 in buffer
		assert len(emitted) == 1
		assert len(emitted[0]) == 100
		assert sdr_device._rx_buffer.size == 50

	def test_leftover_emitted_next_call (self):

		"""Leftover from previous call should be combined with next batch."""

		sdr_device, _, _ = _create_soapy_device()
		sdr_device._rx_buffer = numpy.array([], dtype=numpy.complex64)

		emitted = []
		callback = lambda samples, ctx: emitted.append(len(samples))

		# Call 1: 80 samples, chunk size 100 — nothing emitted yet
		sdr_device._buffer_samples(numpy.ones(80, dtype=numpy.complex64), 100, callback)
		assert len(emitted) == 0

		# Call 2: 80 more (160 total) — should emit 1 chunk, leave 60
		sdr_device._buffer_samples(numpy.ones(80, dtype=numpy.complex64), 100, callback)
		assert len(emitted) == 1
		assert emitted[0] == 100
		assert sdr_device._rx_buffer.size == 60


class TestSoapySdrStreaming:

	def test_read_samples_async_starts_thread (self):

		"""read_samples_async should start a daemon reader thread."""

		sdr_device, mock_device, _ = _create_soapy_device()

		# Make readStream return TIMEOUT so the thread doesn't loop forever
		sr_result = unittest.mock.MagicMock()
		sr_result.ret = -1  # SOAPY_SDR_TIMEOUT
		mock_device.readStream.return_value = sr_result

		callback = unittest.mock.MagicMock()
		sdr_device.read_samples_async(callback, 1024)

		assert sdr_device._reader_thread is not None
		assert sdr_device._reader_thread.daemon is True
		assert sdr_device._reader_thread.is_alive()

		# Clean up
		sdr_device.cancel_read_async()

	def test_cancel_stops_thread (self):

		"""cancel_read_async should stop the reader thread."""

		sdr_device, mock_device, _ = _create_soapy_device()

		sr_result = unittest.mock.MagicMock()
		sr_result.ret = -1  # SOAPY_SDR_TIMEOUT
		mock_device.readStream.return_value = sr_result

		sdr_device.read_samples_async(unittest.mock.MagicMock(), 1024)
		sdr_device.cancel_read_async()

		assert sdr_device._reader_thread is None

	def test_close_cleans_up_stream (self):

		"""close() should deactivate and close the stream."""

		sdr_device, mock_device, _ = _create_soapy_device()

		# Setup a stream first
		sdr_device._stream = 'mock_stream'
		sdr_device.close()

		mock_device.deactivateStream.assert_called_with('mock_stream')
		mock_device.closeStream.assert_called_with('mock_stream')
		assert sdr_device._stream is None


class TestSoapySdrSyncRead:

	def test_read_samples_returns_correct_count (self):

		"""Synchronous read should return exactly the requested number of samples."""

		sdr_device, mock_device, _ = _create_soapy_device()

		# readStream returns 512 samples per call
		def fake_read (stream, bufs, count, timeoutUs=0):
			result = unittest.mock.MagicMock()
			n = min(count, 512)
			bufs[0][:n] = numpy.ones(n, dtype=numpy.complex64) * 0.5
			result.ret = n
			return result

		mock_device.readStream.side_effect = fake_read

		samples = sdr_device.read_samples(1024)

		assert len(samples) == 1024
		assert samples.dtype == numpy.complex64

		# Stream should be cleaned up
		mock_device.deactivateStream.assert_called()
		mock_device.closeStream.assert_called()
