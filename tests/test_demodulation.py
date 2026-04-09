"""Tests for AM and NFM demodulation with synthetic IQ."""

import numpy
import pytest
import scipy.fft

import substation.dsp.demodulation

import iq_generators


def _dominant_freq (audio: numpy.ndarray, sample_rate: int) -> float:
	"""Return the dominant frequency in the audio signal via FFT."""
	spectrum = numpy.abs(scipy.fft.rfft(audio))
	freqs = scipy.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
	# Ignore DC bin
	spectrum[0] = 0
	return float(freqs[numpy.argmax(spectrum)])


class TestNFMDemodulation:

	def test_recovers_tone (self):
		"""Demodulate a 1 kHz FM signal and verify the tone is present."""
		sr = 1_024_000
		audio_rate = 16000
		audio_freq = 1000.0
		deviation = 2500.0
		iq = iq_generators.generate_fm_iq(audio_freq, deviation, sr, 0.1)
		audio, state = substation.dsp.demodulation.demodulate_nfm(iq, sr, audio_rate)
		assert len(audio) > 0
		# Skip the first 20% to avoid filter transients
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], audio_rate)
		assert abs(dominant - audio_freq) < 200  # within 200 Hz

	def test_state_continuity (self):
		"""Two consecutive blocks produce continuous output."""
		sr = 1_024_000
		audio_rate = 16000
		iq = iq_generators.generate_fm_iq(1000.0, 2500.0, sr, 0.2)
		half = len(iq) // 2

		state = None
		audio_a, state = substation.dsp.demodulation.demodulate_nfm(iq[:half], sr, audio_rate, state=state)
		audio_b, state = substation.dsp.demodulation.demodulate_nfm(iq[half:], sr, audio_rate, state=state)
		joined = numpy.concatenate([audio_a, audio_b])

		# No large discontinuity at the boundary
		boundary = len(audio_a)
		if boundary > 0 and boundary < len(joined):
			jump = abs(joined[boundary] - joined[boundary - 1])
			# A smooth signal shouldn't have a big jump
			assert jump < 0.5

	def test_empty_input (self):
		audio, state = substation.dsp.demodulation.demodulate_nfm(
			numpy.array([], dtype=numpy.complex64), 1_024_000, 16000
		)
		assert len(audio) == 0

	def test_output_dtype (self):
		iq = iq_generators.generate_fm_iq(1000.0, 2500.0, 1_024_000, 0.05)
		audio, _ = substation.dsp.demodulation.demodulate_nfm(iq, 1_024_000, 16000)
		assert audio.dtype == numpy.float32


class TestAMDemodulation:

	def test_recovers_tone (self):
		"""Demodulate a 1 kHz AM signal and verify the tone is present."""
		sr = 1_024_000
		audio_rate = 16000
		audio_freq = 1000.0
		iq = iq_generators.generate_am_iq(audio_freq, 0.8, sr, 0.1)
		audio, state = substation.dsp.demodulation.demodulate_am(iq, sr, audio_rate)
		assert len(audio) > 0
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], audio_rate)
		assert abs(dominant - audio_freq) < 200

	def test_empty_input (self):
		audio, state = substation.dsp.demodulation.demodulate_am(
			numpy.array([], dtype=numpy.complex64), 1_024_000, 16000
		)
		assert len(audio) == 0

	def test_output_range (self):
		"""AM output should be within [-1, 1] after AGC and clipping."""
		iq = iq_generators.generate_am_iq(1000.0, 0.8, 1_024_000, 0.1)
		audio, _ = substation.dsp.demodulation.demodulate_am(iq, 1_024_000, 16000)
		assert numpy.all(audio >= -1.0)
		assert numpy.all(audio <= 1.0)


class TestDemodulatorsDict:

	def test_keys (self):
		assert "NFM" in substation.dsp.demodulation.DEMODULATORS
		assert "AM" in substation.dsp.demodulation.DEMODULATORS

	def test_callable (self):
		for key, func in substation.dsp.demodulation.DEMODULATORS.items():
			assert callable(func)
