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

		"""An active channel change sends /radio/state with is_active=1."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 3, True, 15.2)

		assert patched_send_message.call_count == 1
		address, args = patched_send_message.call_args[0]
		assert address == '/radio/state'
		assert args == ['pmr', 3, 1, 15.2]

	def test_inactive_sends_zero (self, patched_send_message):

		"""An inactive channel change sends is_active=0."""

		sender = substation.osc_sender.OscEventSender()
		sender.on_state_change('pmr', 7, False, 3.4)

		address, args = patched_send_message.call_args[0]
		assert address == '/radio/state'
		assert args == ['pmr', 7, 0, 3.4]

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
		assert args == ['pmr', 3, '/tmp/foo.wav']

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
		assert first_args == ['pmr', 3, '/tmp/foo.wav']

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

		for call in patched_send_message.call_args_list:
			address, args = call[0]
			# The last positional arg is always the path (both addresses)
			path_arg = args[-1]
			assert type(path_arg) is str
			assert path_arg == '/tmp/foo.wav'


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

class TestAttach:

	def test_registers_both_callbacks (self):

		"""
		attach() should call add_state_callback and add_recording_callback
		on the scanner, each with the sender's bound method.
		"""

		sender = substation.osc_sender.OscEventSender()
		mock_scanner = unittest.mock.Mock()

		sender.attach(mock_scanner)

		mock_scanner.add_state_callback.assert_called_once_with(sender.on_state_change)
		mock_scanner.add_recording_callback.assert_called_once_with(sender.on_recording_saved)


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
			'/radio/recording', ['pmr', 3, '/tmp/foo.wav']
		)

		# Sampler failure logged
		sampler_warnings = [r for r in caplog.records if '/sample/import' in r.getMessage()]
		assert len(sampler_warnings) == 1
