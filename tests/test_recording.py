"""Tests for ChannelRecorder: ring buffer, WAV output, BEXT metadata."""

import asyncio
import struct
import threading

import numpy
import pytest
import soundfile

import substation.recording


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder (tmp_path, max_seconds=5.0, sample_rate=16000, noise_reduction=False):
	"""Create a ChannelRecorder writing to tmp_path."""
	return substation.recording.ChannelRecorder(
		channel_freq=446.00625e6,
		channel_index=0,
		band_name="test",
		audio_sample_rate=sample_rate,
		buffer_size_seconds=max_seconds,
		disk_flush_interval_seconds=999,  # we'll flush manually
		audio_output_dir=str(tmp_path),
		modulation="NFM",
		filename_suffix="test",
		soft_limit_drive=2.0,
		noise_reduction_enabled=noise_reduction,
	)


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class TestRingBuffer:

	def test_append_basic (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		samples = numpy.ones(100, dtype=numpy.float32) * 0.5
		rec.append_audio(samples)
		assert rec._ring_frames_written == 100

	def test_append_overflow_drops_oldest (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=0.01, sample_rate=16000)
		# Buffer capacity = 0.01 * 16000 = 160 samples
		cap = rec.max_buffer_samples
		rec.append_audio(numpy.ones(cap, dtype=numpy.float32))
		rec.append_audio(numpy.ones(50, dtype=numpy.float32) * 2.0)
		assert rec._ring_frames_written == cap + 50
		# Flushed pointer should have advanced
		assert rec._ring_frames_flushed > 0

	def test_wrap_around_integrity (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=0.01, sample_rate=16000)
		cap = rec.max_buffer_samples
		# Write 80% capacity
		n1 = int(cap * 0.8)
		rec.append_audio(numpy.ones(n1, dtype=numpy.float32) * 1.0)
		# Write another 40% (wraps)
		n2 = int(cap * 0.4)
		rec.append_audio(numpy.ones(n2, dtype=numpy.float32) * 2.0)
		assert rec._ring_write_head == (n1 + n2) % cap

	def test_flush_contiguous (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		data = numpy.arange(100, dtype=numpy.float32) / 100.0
		rec.append_audio(data)
		# Flush synchronously via the internal method
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.close()
		assert rec._ring_frames_flushed == 100

	def test_flush_wrapped (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=0.01, sample_rate=16000)
		cap = rec.max_buffer_samples
		# Fill to 80%, flush, then write 40% more (wraps)
		n1 = int(cap * 0.8)
		rec.append_audio(numpy.ones(n1, dtype=numpy.float32) * 1.0)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		n2 = int(cap * 0.4)
		rec.append_audio(numpy.ones(n2, dtype=numpy.float32) * 2.0)
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.close()
		assert rec._ring_frames_flushed == n1 + n2

	def test_flush_empty (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.close()
		# Should not crash, nothing written
		assert rec._ring_frames_flushed == 0

	def test_double_flush (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		rec.append_audio(numpy.ones(100, dtype=numpy.float32))
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec._flush_buffer_to_disk())  # second should be no-op
		loop.close()
		assert rec._ring_frames_flushed == 100

	def test_closing_blocks_append (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		rec._closing.set()
		rec.append_audio(numpy.ones(100, dtype=numpy.float32))
		assert rec._ring_frames_written == 0

	def test_huge_chunk_keeps_tail (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=0.01, sample_rate=16000)
		cap = rec.max_buffer_samples
		big = numpy.arange(cap * 3, dtype=numpy.float32)
		rec.append_audio(big)
		# After truncation to tail, ring should contain exactly `cap` samples
		n_unflushed = rec._ring_frames_written - rec._ring_frames_flushed
		assert n_unflushed == cap
		# The ring data should be the last `cap` values from the big array
		numpy.testing.assert_array_equal(rec._ring[:cap], big[-cap:])


# ---------------------------------------------------------------------------
# WAV output
# ---------------------------------------------------------------------------

class TestWavOutput:

	def test_wav_file_created (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		rec.append_audio(numpy.ones(1600, dtype=numpy.float32) * 0.3)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()
		# Check file exists and is valid
		assert rec.filepath.endswith(".wav")
		data, sr = soundfile.read(rec.filepath)
		assert sr == 16000
		assert len(data) > 0

	def test_soft_limiter (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		# Input with values > 1 to test soft limiting
		loud = numpy.ones(1600, dtype=numpy.float32) * 2.0
		rec.append_audio(loud)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()
		data, _ = soundfile.read(rec.filepath)
		# Soft limiter should keep output within [-1, 1]
		assert numpy.max(numpy.abs(data)) <= 1.0


class TestBextMetadata:

	def test_bext_chunk_present (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		rec.append_audio(numpy.ones(1600, dtype=numpy.float32) * 0.3)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()
		# Read the raw file and look for the 'bext' chunk
		with open(rec.filepath, 'rb') as f:
			raw = f.read()
		assert b'bext' in raw


class TestCheckEmpty:

	def test_white_noise_is_empty (self, tmp_path):
		"""White noise has flat spectrum → check_empty returns True."""
		sr = 16000
		noise = numpy.random.RandomState(0).randn(sr * 2).astype(numpy.float32) * 0.01
		path = str(tmp_path / "noise.wav")
		soundfile.write(path, noise, sr)
		assert substation.recording.ChannelRecorder.check_empty(path) is True

	def test_tone_is_not_empty (self, tmp_path):
		"""A sine tone has peaked spectrum → check_empty returns False."""
		sr = 16000
		t = numpy.arange(sr * 2) / sr
		tone = (numpy.sin(2 * numpy.pi * 1000 * t) * 0.5).astype(numpy.float32)
		path = str(tmp_path / "tone.wav")
		soundfile.write(path, tone, sr)
		assert substation.recording.ChannelRecorder.check_empty(path) is False

	def test_voice_like_signal_is_not_empty (self, tmp_path):
		"""Multi-tone signal mimicking voice formants is not empty."""
		sr = 16000
		t = numpy.arange(sr * 2) / sr
		signal = (0.3 * numpy.sin(2 * numpy.pi * 300 * t) +
		          0.2 * numpy.sin(2 * numpy.pi * 800 * t) +
		          0.1 * numpy.sin(2 * numpy.pi * 1500 * t)).astype(numpy.float32)
		path = str(tmp_path / "voice.wav")
		soundfile.write(path, signal, sr)
		assert substation.recording.ChannelRecorder.check_empty(path) is False

	def test_very_short_file_is_empty (self, tmp_path):
		"""Files shorter than 512 samples are always discarded."""
		sr = 16000
		short = numpy.zeros(100, dtype=numpy.float32)
		path = str(tmp_path / "short.wav")
		soundfile.write(path, short, sr)
		assert substation.recording.ChannelRecorder.check_empty(path) is True
