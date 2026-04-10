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
		assert "USB" in substation.dsp.demodulation.DEMODULATORS
		assert "LSB" in substation.dsp.demodulation.DEMODULATORS

	def test_callable (self):
		for key, func in substation.dsp.demodulation.DEMODULATORS.items():
			assert callable(func)


def _ssb_iq_tone (audio_freq: float, sample_rate: int, duration_s: float) -> numpy.ndarray:

	"""
	Synthesise the IQ baseband for a single SSB tone.

	A USB transmission containing a tone at +audio_freq Hz produces a
	complex sinusoid at +audio_freq in the IQ baseband (the carrier is
	implicit at 0 Hz).  An LSB transmission of the same tone produces
	a complex sinusoid at -audio_freq.
	"""

	t = numpy.arange(int(sample_rate * duration_s), dtype=numpy.float64) / sample_rate
	return numpy.exp(1j * 2 * numpy.pi * audio_freq * t).astype(numpy.complex64)


class TestSSBDemodulation:

	"""Tests for the Weaver-method SSB demodulator."""

	def test_usb_recovers_audio_tone (self):

		"""USB demodulation of a +1 kHz IQ tone yields audio at 1 kHz."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1000.0, sr, 1.0)
		audio, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		# Skip the filter transient at the start
		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], asr)

		assert abs(dominant - 1000.0) < 50, f"Expected ~1000 Hz, got {dominant} Hz"

	def test_lsb_recovers_audio_tone (self):

		"""LSB demodulation of a -1 kHz IQ tone yields audio at 1 kHz."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(-1000.0, sr, 1.0)
		audio, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq, sr, asr)

		settle = len(audio) // 5
		dominant = _dominant_freq(audio[settle:], asr)

		assert abs(dominant - 1000.0) < 50, f"Expected ~1000 Hz, got {dominant} Hz"

	def test_usb_rejects_lsb_tone (self):

		"""Demodulating a +1 kHz IQ tone (USB content) as LSB should
		produce strongly attenuated audio compared to demodulating as USB.

		The test adds a small amount of background noise to the input.
		With pure tones the post-AGC FFT bin ratio is meaningless because
		the AGC normalises any residual to fill the dynamic range; with
		realistic noise the right-sideband demod produces a clean tone
		while the wrong-sideband demod produces noise-dominated output.
		"""

		sr = 192_000
		asr = 16_000
		numpy.random.seed(42)

		iq = _ssb_iq_tone(+1000.0, sr, 1.0)
		noise = (numpy.random.randn(len(iq)) + 1j * numpy.random.randn(len(iq))).astype(numpy.complex64) * 0.01
		iq_noisy = (iq + noise).astype(numpy.complex64)

		audio_right, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq_noisy, sr, asr)
		audio_wrong, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq_noisy, sr, asr)

		# Use the steady-state second half to avoid filter transients
		half = len(audio_right) // 2
		right_ss = audio_right[half:]
		wrong_ss = audio_wrong[half:]

		# Tone-to-rest ratio: dominant bin power vs everything else
		def tone_ratio_db (audio, target_hz, asr):
			fft = numpy.abs(scipy.fft.rfft(audio))
			freqs = scipy.fft.rfftfreq(len(audio), 1.0 / asr)
			bin_idx = int(numpy.argmin(numpy.abs(freqs - target_hz)))
			tone = fft[bin_idx] ** 2
			rest = numpy.sum(fft ** 2) - tone
			return 10 * numpy.log10(tone / max(rest, 1e-30))

		right_db = tone_ratio_db(right_ss, 1000.0, asr)
		wrong_db = tone_ratio_db(wrong_ss, 1000.0, asr)

		# Right sideband should have a much cleaner tone than wrong
		assert right_db > wrong_db + 15.0, (
			f"Sideband rejection too weak: right={right_db:.1f} dB, wrong={wrong_db:.1f} dB"
		)

	def test_lsb_rejects_usb_tone (self):

		"""Symmetric: a -1 kHz LSB tone demodulated as USB should be
		much weaker than the same tone demodulated as LSB."""

		sr = 192_000
		asr = 16_000
		numpy.random.seed(43)

		iq = _ssb_iq_tone(-1000.0, sr, 1.0)
		noise = (numpy.random.randn(len(iq)) + 1j * numpy.random.randn(len(iq))).astype(numpy.complex64) * 0.01
		iq_noisy = (iq + noise).astype(numpy.complex64)

		audio_right, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq_noisy, sr, asr)
		audio_wrong, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq_noisy, sr, asr)

		half = len(audio_right) // 2
		right_ss = audio_right[half:]
		wrong_ss = audio_wrong[half:]

		def tone_ratio_db (audio, target_hz, asr):
			fft = numpy.abs(scipy.fft.rfft(audio))
			freqs = scipy.fft.rfftfreq(len(audio), 1.0 / asr)
			bin_idx = int(numpy.argmin(numpy.abs(freqs - target_hz)))
			tone = fft[bin_idx] ** 2
			rest = numpy.sum(fft ** 2) - tone
			return 10 * numpy.log10(tone / max(rest, 1e-30))

		right_db = tone_ratio_db(right_ss, 1000.0, asr)
		wrong_db = tone_ratio_db(wrong_ss, 1000.0, asr)

		assert right_db > wrong_db + 15.0, (
			f"Sideband rejection too weak: right={right_db:.1f} dB, wrong={wrong_db:.1f} dB"
		)

	def test_state_continuity_across_blocks (self):

		"""Feeding a long signal in one chunk vs two halves should
		produce equivalent audio after the filter transient — proves
		that the per-block oscillator phase and filter state are
		preserved correctly."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1500.0, sr, 0.5)

		# Single-shot
		one_shot, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		# Two-half: feed first half, then second half, share state
		half = len(iq) // 2
		first, state = substation.dsp.demodulation.DEMODULATORS['USB'](iq[:half], sr, asr)
		second, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq[half:], sr, asr, state=state)
		concat = numpy.concatenate([first, second])

		# Both should be the same length
		assert len(one_shot) == len(concat), f"Lengths differ: {len(one_shot)} vs {len(concat)}"

		# After the initial transient, the two should agree closely.
		# Skip the first 20% to bypass the filter warm-up region.
		settle = len(one_shot) // 5
		diff_rms = numpy.sqrt(numpy.mean((one_shot[settle:] - concat[settle:]) ** 2))
		one_shot_rms = numpy.sqrt(numpy.mean(one_shot[settle:] ** 2))

		# Allow up to 5% RMS divergence — looser than NFM's continuity
		# test because the AGC level estimate carries some lag across
		# the block boundary, but tight enough to catch real bugs like
		# a missing phase update or filter state reset.
		assert diff_rms < 0.05 * one_shot_rms, (
			f"State continuity violated: diff RMS {diff_rms:.4f} vs signal RMS {one_shot_rms:.4f}"
		)

	def test_zero_input_returns_zero (self):

		"""Empty IQ → empty audio, no exceptions."""

		sr = 192_000
		asr = 16_000
		iq = numpy.array([], dtype=numpy.complex64)

		audio_usb, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)
		audio_lsb, _ = substation.dsp.demodulation.DEMODULATORS['LSB'](iq, sr, asr)

		assert audio_usb.size == 0
		assert audio_lsb.size == 0

	def test_silent_input_returns_silent (self):

		"""Zero-amplitude IQ → zero-amplitude audio, no NaNs."""

		sr = 192_000
		asr = 16_000
		iq = numpy.zeros(int(sr * 0.1), dtype=numpy.complex64)

		audio, _ = substation.dsp.demodulation.DEMODULATORS['USB'](iq, sr, asr)

		assert not numpy.any(numpy.isnan(audio))
		# AGC has a non-zero floor so the output is bounded but small
		assert numpy.max(numpy.abs(audio)) < 0.5

	def test_invalid_sideband_raises (self):

		"""demodulate_ssb with an invalid sideband string raises ValueError."""

		sr = 192_000
		asr = 16_000
		iq = _ssb_iq_tone(+1000.0, sr, 0.1)

		with pytest.raises(ValueError, match="sideband"):
			substation.dsp.demodulation.demodulate_ssb(iq, sr, asr, sideband='WSB')
