"""
Tests for substation.osc_sender.

The whole module is skipped cleanly when python-osc isn't installed,
so a minimal install can still run `pytest` without failure.  Tests
patch SimpleUDPClient.send_message to capture calls without ever
opening a real UDP socket.
"""

import logging
import pathlib
import unittest.mock

import numpy
import pytest


# Skip the entire module if python-osc isn't available.  Matches how
# test_soapysdr.py handles its optional dependency.
pytest.importorskip('pythonosc')

import pythonosc.udp_client  # noqa: E402

import substation.osc_sender  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_send_message ():

	"""
	Patch SimpleUDPClient.send_message so tests can inspect outgoing
	messages without touching the network.  Yields the mock.
	"""

	with unittest.mock.patch.object(
		pythonosc.udp_client.SimpleUDPClient, 'send_message'
	) as mock_send:
		yield mock_send


# ---------------------------------------------------------------------------
# on_state_change
# ---------------------------------------------------------------------------

class TestOnStateChange:

	def test_active_sends_one (self, patched_send_message):

		"""An active channel change sends /radio/state with is_active=1 and no-tone sentinels."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 3, True, 15.2)

		assert patched_send_message.call_count == 1
		address, args = patched_send_message.call_args[0]
		assert address == '/radio/state'
		assert args == ['pmr', 3, 1, 15.2, 0.0, 0]

	def test_inactive_sends_zero (self, patched_send_message):

		"""An inactive channel change sends is_active=0."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 7, False, 3.4)

		address, args = patched_send_message.call_args[0]
		assert address == '/radio/state'
		assert args == ['pmr', 7, 0, 3.4, 0.0, 0]

	def test_ctcss_included_when_detected (self, patched_send_message):

		"""A detected CTCSS tone rides along on /radio/state as the 5th arg."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 3, True, 15.2, ctcss_hz=136.5, dcs_code=None)

		_, args = patched_send_message.call_args[0]
		assert args == ['pmr', 3, 1, 15.2, 136.5, 0]

	def test_dcs_included_when_detected (self, patched_send_message):

		"""A detected DCS code rides along on /radio/state as the 6th arg."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 3, True, 15.2, ctcss_hz=None, dcs_code=0o023)

		_, args = patched_send_message.call_args[0]
		assert args == ['pmr', 3, 1, 15.2, 0.0, 0o023]

	def test_numpy_snr_coerced_to_float (self, patched_send_message):

		"""
		A numpy.float64 SNR value must be coerced to plain Python float
		so pythonosc doesn't receive a subclass it can't tag correctly.
		Defence against future numpy creep in the scanner's dispatch path.
		"""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 0, True, numpy.float64(12.5))

		_, args = patched_send_message.call_args[0]
		# After float() coercion the value is a plain float (not numpy.float64)
		assert type(args[3]) is float
		assert args[3] == 12.5


# ---------------------------------------------------------------------------
# on_recording_saved
# ---------------------------------------------------------------------------

class TestOnRecordingSaved:

	def test_default_sends_only_radio_recording (self, patched_send_message):

		"""
		With no sampler_host, only /radio/recording is sent — and the
		internal _sampler_client is None so the sampler code path is
		definitely skipped.
		"""

		sender = substation.osc_sender.OscEventSender()
		assert sender._sampler_client is None

		sender.on_recording_saved('pmr', 3, '/tmp/foo.wav')

		assert patched_send_message.call_count == 1
		address, args = patched_send_message.call_args[0]
		assert address == '/radio/recording'
		assert args == ['pmr', 3, '/tmp/foo.wav', 0.0, 0]

	def test_recording_tone_fields_included (self, patched_send_message):

		"""Detected tone on recording_saved rides along as trailing args."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_recording_saved('pmr', 3, '/tmp/foo.wav', ctcss_hz=136.5, dcs_code=None)

		address, args = patched_send_message.call_args[0]
		assert address == '/radio/recording'
		assert args == ['pmr', 3, '/tmp/foo.wav', 136.5, 0]

	def test_with_sampler_sends_both_messages (self, patched_send_message):

		"""
		With sampler_host set, /radio/recording goes to the sequencer
		client and /sample/import goes to the sampler client — two
		distinct send_message calls with the same file path.
		"""

		sender = substation.osc_sender.OscEventSender(sampler_host='127.0.0.1')
		assert sender._sampler_client is not None

		sender.on_recording_saved('pmr', 3, '/tmp/foo.wav')

		assert patched_send_message.call_count == 2

		# Order is: sequencer first, then sampler.
		first_address, first_args = patched_send_message.call_args_list[0][0]
		second_address, second_args = patched_send_message.call_args_list[1][0]

		assert first_address == '/radio/recording'
		assert first_args == ['pmr', 3, '/tmp/foo.wav', 0.0, 0]

		assert second_address == '/sample/import'
		assert second_args == ['/tmp/foo.wav']

	def test_pathlib_path_coerced_to_str (self, patched_send_message):

		"""
		Forward-compat guard: if the scanner ever starts passing a
		pathlib.Path instead of a str, the OSC argument must still be
		a plain str (pythonosc doesn't know how to encode Path).
		"""

		sender = substation.osc_sender.OscEventSender(sampler_host='127.0.0.1')
		sender.on_recording_saved('pmr', 3, pathlib.Path('/tmp/foo.wav'))

		# /radio/recording: path is 3rd arg (before the two tone fields).
		# /sample/import: path is the only arg.
		radio_args = patched_send_message.call_args_list[0][0][1]
		assert type(radio_args[2]) is str and radio_args[2] == '/tmp/foo.wav'

		sampler_args = patched_send_message.call_args_list[1][0][1]
		assert type(sampler_args[0]) is str and sampler_args[0] == '/tmp/foo.wav'


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

class TestAttach:

	def test_registers_both_callbacks (self):

		"""
		attach() should register event handlers via scanner.on().
		"""

		sender = substation.osc_sender.OscEventSender()
		mock_scanner = unittest.mock.Mock()

		sender.attach(mock_scanner)

		# attach() should call scanner.on() twice: once for channel_state, once for recording_saved
		assert mock_scanner.on.call_count == 2
		event_names = [call.args[0] for call in mock_scanner.on.call_args_list]
		assert 'channel_state' in event_names
		assert 'recording_saved' in event_names


class TestEventAdapters:

	def test_state_adapter_forwards_tone_kwargs (self, patched_send_message):

		"""_on_state_event pulls ctcss_hz / dcs_code from kwargs and forwards them."""

		sender = substation.osc_sender.OscEventSender()
		sender._on_state_event(
			band='pmr', index=3, freq=446e6, is_active=True, snr_db=15.2,
			ctcss_hz=136.5, dcs_code=None,
		)

		_, args = patched_send_message.call_args[0]
		assert args == ['pmr', 3, 1, 15.2, 136.5, 0]

	def test_recording_adapter_forwards_tone_kwargs (self, patched_send_message):

		"""_on_recording_event pulls ctcss_hz / dcs_code from kwargs and forwards them."""

		sender = substation.osc_sender.OscEventSender()
		sender._on_recording_event(
			band='pmr', index=3, freq=446e6, file_path='/tmp/foo.wav',
			ctcss_hz=None, dcs_code=0o023,
		)

		_, args = patched_send_message.call_args[0]
		assert args == ['pmr', 3, '/tmp/foo.wav', 0.0, 0o023]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

	def test_oserror_is_logged_not_raised (self, caplog):

		"""
		A transient UDP / socket failure (OSError) must be swallowed with
		a warning — a raising callback would noise up the scanner's event
		loop and potentially leak state.
		"""

		sender = substation.osc_sender.OscEventSender()

		with unittest.mock.patch.object(
			sender._client, 'send_message', side_effect=OSError('network down')
		):
			with caplog.at_level(logging.WARNING, logger='substation.osc_sender'):
				# Must not raise
				sender.on_state_change('pmr', 3, True, 15.2)

		# Exactly one warning about /radio/state
		warnings = [r for r in caplog.records if '/radio/state' in r.getMessage()]
		assert len(warnings) == 1
		assert 'network down' in warnings[0].getMessage()

	def test_sampler_failure_does_not_block_recording_message (self, caplog):

		"""
		If the sampler send fails, the /radio/recording send should still
		have happened (sequencer is primary, sampler is opportunistic).
		"""

		sender = substation.osc_sender.OscEventSender(sampler_host='127.0.0.1')

		# Sequencer send succeeds; sampler send fails
		with unittest.mock.patch.object(
			sender._client, 'send_message'
		) as seq_mock, unittest.mock.patch.object(
			sender._sampler_client, 'send_message', side_effect=OSError('sampler unreachable')
		):
			with caplog.at_level(logging.WARNING, logger='substation.osc_sender'):
				sender.on_recording_saved('pmr', 3, '/tmp/foo.wav')

		# Sequencer message still went out
		seq_mock.assert_called_once_with(
			'/radio/recording', ['pmr', 3, '/tmp/foo.wav', 0.0, 0]
		)

		# Sampler failure logged
		sampler_warnings = [r for r in caplog.records if '/sample/import' in r.getMessage()]
		assert len(sampler_warnings) == 1
