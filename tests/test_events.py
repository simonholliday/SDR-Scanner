"""Tests for the RadioScanner event emitter (on/off/emit)."""

import asyncio


class TestEventEmitter:

	def test_on_and_emit (self, scanner_instance):
		"""Handlers registered with on() receive emitted events."""
		received = []
		scanner_instance.on('channel_state', lambda **kw: received.append(kw))
		scanner_instance.emit('channel_state',
			band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0,
			ctcss_hz=None, dcs_code=None)
		assert len(received) == 1
		assert received[0]['band'] == 'pmr'

	def test_off_removes_handler (self, scanner_instance):
		"""Handlers removed with off() stop receiving events."""
		received = []
		handler = lambda **kw: received.append(kw)
		scanner_instance.on('channel_state', handler)
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0,
			ctcss_hz=None, dcs_code=None)
		assert len(received) == 1

		scanner_instance.off('channel_state', handler)
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=False, snr_db=5.0,
			ctcss_hz=None, dcs_code=None)
		assert len(received) == 1

	def test_multiple_handlers (self, scanner_instance):
		"""Multiple handlers on the same event all fire."""
		a, b = [], []
		scanner_instance.on('noise_floor', lambda **kw: a.append(kw))
		scanner_instance.on('noise_floor', lambda **kw: b.append(kw))
		scanner_instance.emit('noise_floor', noise_floor_db=-80.0, warmup_complete=True)
		assert len(a) == 1
		assert len(b) == 1

	def test_emit_unknown_event_is_noop (self, scanner_instance):
		"""Emitting an event with no handlers doesn't raise."""
		scanner_instance.emit('nonexistent_event', foo=42)

	def test_handler_error_doesnt_crash (self, scanner_instance):
		"""A handler that raises doesn't prevent other handlers from running."""
		received = []
		scanner_instance.on('channel_state', lambda **kw: 1/0)
		scanner_instance.on('channel_state', lambda **kw: received.append(kw))
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0,
			ctcss_hz=None, dcs_code=None)
		assert len(received) == 1

	def test_backward_compat_add_state_callback (self, scanner_instance):
		"""add_state_callback still works via the event emitter (tone kwargs are ignored)."""
		received = []
		scanner_instance.add_state_callback(
			lambda band, idx, active, snr: received.append((band, idx, active, snr))
		)
		scanner_instance.emit('channel_state',
			band='pmr', index=3, freq=446e6, is_active=True, snr_db=8.5,
			ctcss_hz=136.5, dcs_code=None)
		assert len(received) == 1
		assert received[0] == ('pmr', 3, True, 8.5)

	def test_recording_saved_carries_tone_fields (self, scanner_instance):
		"""recording_saved handlers receive ctcss_hz and dcs_code as kwargs."""
		received = []
		scanner_instance.on('recording_saved', lambda **kw: received.append(kw))

		scanner_instance.emit('recording_saved',
			band='pmr', index=7, freq=446e6, file_path='/tmp/a.wav',
			ctcss_hz=136.5, dcs_code=None)
		scanner_instance.emit('recording_saved',
			band='pmr', index=8, freq=446.0125e6, file_path='/tmp/b.wav',
			ctcss_hz=None, dcs_code=0o023)
		scanner_instance.emit('recording_saved',
			band='pmr', index=9, freq=446.025e6, file_path='/tmp/c.wav',
			ctcss_hz=None, dcs_code=None)

		assert len(received) == 3
		assert received[0]['ctcss_hz'] == 136.5 and received[0]['dcs_code'] is None
		assert received[1]['ctcss_hz'] is None and received[1]['dcs_code'] == 0o023
		assert received[2]['ctcss_hz'] is None and received[2]['dcs_code'] is None

	def test_channel_state_carries_tone_fields (self, scanner_instance):
		"""channel_state handlers receive ctcss_hz and dcs_code as kwargs."""
		received = []
		scanner_instance.on('channel_state', lambda **kw: received.append(kw))

		# ON with a CTCSS tone detected
		scanner_instance.emit('channel_state',
			band='pmr', index=7, freq=446e6, is_active=True, snr_db=12.0,
			ctcss_hz=136.5, dcs_code=None)

		# ON with a DCS code detected
		scanner_instance.emit('channel_state',
			band='pmr', index=8, freq=446.0125e6, is_active=True, snr_db=9.0,
			ctcss_hz=None, dcs_code=0o023)

		# OFF — tone fields must be None
		scanner_instance.emit('channel_state',
			band='pmr', index=7, freq=446e6, is_active=False, snr_db=1.0,
			ctcss_hz=None, dcs_code=None)

		assert len(received) == 3
		assert received[0]['ctcss_hz'] == 136.5 and received[0]['dcs_code'] is None
		assert received[1]['ctcss_hz'] is None and received[1]['dcs_code'] == 0o023
		assert received[2]['ctcss_hz'] is None and received[2]['dcs_code'] is None

	def test_backward_compat_add_recording_callback (self, scanner_instance):
		"""add_recording_callback still works via the event emitter."""
		received = []
		scanner_instance.add_recording_callback(
			lambda band, idx, path: received.append((band, idx, path))
		)
		scanner_instance.emit('recording_saved',
			band='pmr', index=3, freq=446e6, file_path='/tmp/test.wav')
		assert len(received) == 1
		assert received[0] == ('pmr', 3, '/tmp/test.wav')

	def test_sync_handler_dispatched_via_loop (self, scanner_instance):
		"""When emit() is given a loop, sync handlers are marshalled to it via call_soon_threadsafe."""

		received = []
		scanner_instance.on('channel_state', lambda **kw: received.append(kw))

		async def runner ():
			loop = asyncio.get_running_loop()
			scanner_instance.emit('channel_state', loop=loop,
				band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0,
				ctcss_hz=None, dcs_code=None)
			# call_soon_threadsafe schedules for the next loop tick — yield so it runs.
			await asyncio.sleep(0)

		asyncio.run(runner())

		assert len(received) == 1
		assert received[0]['band'] == 'pmr'

	def test_async_handler_dispatched_via_loop (self, scanner_instance):
		"""When emit() is given a loop and the handler is async, the coroutine is scheduled."""

		received = []

		async def handler (**kw):
			received.append(kw)

		scanner_instance.on('noise_floor', handler)

		async def runner ():
			loop = asyncio.get_running_loop()
			scanner_instance.emit('noise_floor', loop=loop, noise_floor_db=-80.0, warmup_complete=True)
			# run_coroutine_threadsafe needs one yield to let the scheduled task run.
			await asyncio.sleep(0.01)

		asyncio.run(runner())

		assert len(received) == 1
		assert received[0]['noise_floor_db'] == -80.0

	def test_async_handler_without_loop_is_silent_noop (self, scanner_instance):
		"""Async handlers registered on events emitted without `loop=` are silently skipped.

		This is the documented behaviour — emit() is sync, so with no loop it
		cannot schedule the coroutine.  Test lives here so any future change
		(e.g. raising instead of skipping) is a visible breakage.
		"""

		received = []

		async def handler (**kw):
			received.append(kw)

		scanner_instance.on('channel_snr', handler)
		# No loop= → handler silently not invoked.
		scanner_instance.emit('channel_snr', channels=[])

		assert received == []
