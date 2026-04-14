"""Tests for the RadioScanner event emitter (on/off/emit)."""


class TestEventEmitter:

	def test_on_and_emit (self, scanner_instance):
		"""Handlers registered with on() receive emitted events."""
		received = []
		scanner_instance.on('channel_state', lambda **kw: received.append(kw))
		scanner_instance.emit('channel_state',
			band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0)
		assert len(received) == 1
		assert received[0]['band'] == 'pmr'

	def test_off_removes_handler (self, scanner_instance):
		"""Handlers removed with off() stop receiving events."""
		received = []
		handler = lambda **kw: received.append(kw)
		scanner_instance.on('channel_state', handler)
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0)
		assert len(received) == 1

		scanner_instance.off('channel_state', handler)
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=False, snr_db=5.0)
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
		scanner_instance.emit('channel_state', band='pmr', index=1, freq=446e6, is_active=True, snr_db=10.0)
		assert len(received) == 1

	def test_backward_compat_add_state_callback (self, scanner_instance):
		"""add_state_callback still works via the event emitter."""
		received = []
		scanner_instance.add_state_callback(
			lambda band, idx, active, snr: received.append((band, idx, active, snr))
		)
		scanner_instance.emit('channel_state',
			band='pmr', index=3, freq=446e6, is_active=True, snr_db=8.5)
		assert len(received) == 1
		assert received[0] == ('pmr', 3, True, 8.5)

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
