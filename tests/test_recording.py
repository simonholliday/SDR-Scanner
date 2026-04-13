"""Tests for ChannelRecorder: ring buffer, WAV/FLAC output, BEXT metadata."""

import asyncio
import struct
import threading

import mutagen.flac
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
# format_freq
# ---------------------------------------------------------------------------

class TestFormatFreq:

	def test_ghz (self):
		assert substation.recording.format_freq(1420405000) == "1.420405GHz"

	def test_ghz_round (self):
		assert substation.recording.format_freq(2e9) == "2GHz"

	def test_mhz (self):
		assert substation.recording.format_freq(446006250) == "446.00625MHz"

	def test_mhz_trailing_zeros (self):
		assert substation.recording.format_freq(125850000) == "125.85MHz"

	def test_khz (self):
		assert substation.recording.format_freq(14200) == "14.2kHz"

	def test_khz_sub_mhz (self):
		assert substation.recording.format_freq(500000) == "500kHz"

	def test_filename_contains_freq (self, tmp_path):
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		assert "446.00625MHz" in rec.filepath


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


class TestFlacOutput:

	def test_flac_file_created (self, tmp_path):
		"""FLAC recorder creates a .flac file with valid audio."""
		rec = _make_recorder(tmp_path, max_seconds=1.0)
		rec.audio_format = 'flac'
		# Re-create with FLAC format
		rec = substation.recording.ChannelRecorder(
			channel_freq=446.00625e6,
			channel_index=0,
			band_name="test",
			audio_sample_rate=16000,
			buffer_size_seconds=1.0,
			disk_flush_interval_seconds=999,
			audio_output_dir=str(tmp_path),
			modulation="NFM",
			filename_suffix="test",
			soft_limit_drive=2.0,
			noise_reduction_enabled=False,
			audio_format='flac',
		)
		rec.append_audio(numpy.ones(1600, dtype=numpy.float32) * 0.3)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()

		assert rec.filepath.endswith(".flac")
		data, sr = soundfile.read(rec.filepath)
		assert sr == 16000
		assert len(data) > 0

	def test_flac_lossless_roundtrip (self, tmp_path):
		"""FLAC is lossless: int16 samples written and read back are identical."""
		# Write int16 samples directly to a FLAC file via soundfile,
		# then read back and verify bit-identical.
		original = (numpy.sin(numpy.linspace(0, 100, 3200)) * 16000).astype(numpy.int16)
		flac_path = str(tmp_path / "roundtrip.flac")
		soundfile.write(flac_path, original, 16000, subtype='PCM_16')
		readback, sr = soundfile.read(flac_path, dtype='int16')
		assert sr == 16000
		assert numpy.array_equal(original, readback)

	def test_flac_metadata_present (self, tmp_path):
		"""FLAC files have Vorbis comment metadata."""
		rec = substation.recording.ChannelRecorder(
			channel_freq=446.00625e6, channel_index=0, band_name="test",
			audio_sample_rate=16000, buffer_size_seconds=1.0,
			disk_flush_interval_seconds=999, audio_output_dir=str(tmp_path),
			modulation="NFM", noise_reduction_enabled=False, audio_format='flac',
		)
		rec.append_audio(numpy.ones(1600, dtype=numpy.float32) * 0.3)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()

		flac = mutagen.flac.FLAC(rec.filepath)
		assert 'DATE' in flac
		assert 'CREATION_TIME' in flac
		assert 'COMMENT' in flac
		assert '446' in flac['COMMENT'][0]

	def test_flac_no_bext_chunk (self, tmp_path):
		"""FLAC files should not contain a BEXT chunk."""
		rec = substation.recording.ChannelRecorder(
			channel_freq=446e6, channel_index=0, band_name="test",
			audio_sample_rate=16000, buffer_size_seconds=1.0,
			disk_flush_interval_seconds=999, audio_output_dir=str(tmp_path),
			noise_reduction_enabled=False, audio_format='flac',
		)
		rec.append_audio(numpy.ones(1600, dtype=numpy.float32) * 0.3)
		loop = asyncio.new_event_loop()
		loop.run_until_complete(rec._flush_buffer_to_disk())
		loop.run_until_complete(rec.close())
		loop.close()

		with open(rec.filepath, 'rb') as f:
			raw = f.read()
		assert b'bext' not in raw


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


class TestTrimCarrierTransients:

	SR = 16000

	def _make_signal (self, has_start_click: bool = True, has_end_click: bool = True) -> numpy.ndarray:
		"""Build: [noise + click + gap + voice + gap + click + noise]."""
		noise_level = 0.005
		rng = numpy.random.RandomState(42)

		pre_noise = rng.randn(int(self.SR * 0.01)).astype(numpy.float32) * noise_level
		# Realistic carrier transient: ~5ms sharp spike with exponential decay
		click_len = int(self.SR * 0.005)
		click_env = 0.5 * numpy.exp(-numpy.linspace(0, 5, click_len))
		click_on = (click_env * numpy.sign(rng.randn(click_len))).astype(numpy.float32)
		gap = rng.randn(int(self.SR * 0.02)).astype(numpy.float32) * noise_level
		voice = (0.15 * numpy.sin(2 * numpy.pi * 300 * numpy.arange(self.SR) / self.SR)).astype(numpy.float32)
		click_off = (click_env[::-1] * numpy.sign(rng.randn(click_len))).astype(numpy.float32)
		post_noise = rng.randn(int(self.SR * 0.01)).astype(numpy.float32) * noise_level

		parts = []
		parts.append(pre_noise)
		if has_start_click:
			parts.append(click_on)
			parts.append(gap)
		parts.append(voice)
		if has_end_click:
			parts.append(gap.copy())
			parts.append(click_off)
		parts.append(post_noise)
		return numpy.concatenate(parts)

	def test_removes_start_transient (self):
		audio = self._make_signal(has_start_click=True, has_end_click=False)
		original_len = len(audio)
		trimmed = substation.recording._trim_carrier_transient_start(audio, self.SR)
		assert len(trimmed) < original_len
		assert numpy.abs(trimmed[0]) < 0.05

	def test_removes_end_transient (self):
		audio = self._make_signal(has_start_click=False, has_end_click=True)
		original_len = len(audio)
		trimmed = substation.recording._trim_carrier_transient_end(audio, self.SR)
		assert len(trimmed) < original_len
		assert numpy.abs(trimmed[-1]) < 0.05

	def test_preserves_voice_only_signal (self):
		"""Signal without carrier transients should be unchanged."""
		audio = self._make_signal(has_start_click=False, has_end_click=False)
		trimmed_start = substation.recording._trim_carrier_transient_start(audio, self.SR)
		trimmed_end = substation.recording._trim_carrier_transient_end(audio, self.SR)
		assert len(trimmed_start) == len(audio)
		assert len(trimmed_end) == len(audio)

	def test_voice_starting_loud_is_not_trimmed (self):
		"""A signal that starts with loud voice (no preceding silence) must not be trimmed."""
		voice = (0.2 * numpy.sin(2 * numpy.pi * 500 * numpy.arange(self.SR) / self.SR)).astype(numpy.float32)
		trimmed = substation.recording._trim_carrier_transient_start(voice, self.SR)
		assert len(trimmed) == len(voice)

	def test_transient_at_sample_zero (self):
		"""A carrier transient right at sample 0 (no pre-silence) should be trimmed."""
		rng = numpy.random.RandomState(99)
		noise_level = 0.005
		# Transient at sample 0: sharp spike decaying over ~5ms
		click_len = int(self.SR * 0.005)
		click_env = 0.5 * numpy.exp(-numpy.linspace(0, 5, click_len))
		click = (click_env * numpy.sign(rng.randn(click_len))).astype(numpy.float32)
		# Then quiet gap + voice
		gap = rng.randn(int(self.SR * 0.02)).astype(numpy.float32) * noise_level
		voice = (0.02 * numpy.sin(2 * numpy.pi * 300 * numpy.arange(self.SR) / self.SR)).astype(numpy.float32)
		audio = numpy.concatenate([click, gap, voice])
		trimmed = substation.recording._trim_carrier_transient_start(audio, self.SR)
		assert len(trimmed) < len(audio)
		assert numpy.abs(trimmed[0]) < 0.05
