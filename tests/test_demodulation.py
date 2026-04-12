"""Tests for AM and NFM demodulation with synthetic IQ."""

import time

import numpy
import pytest
import scipy.fft
import scipy.signal

import substation.dsp.demodulation
import substation.dsp.filters

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


class TestPickIfDecimation:

	"""
	Tests for the _pick_if_decimation helper.

	Regression tests for two separate issues:

	1. AirSpy R2 hang: with sample_rate=2500000 and audio_sample_rate=16000,
	   the previous naive round(sample_rate / target_if_rate) produced
	   if_decimation=39 → if_rate=64103 (coprime with 2500000) → a 50-million-
	   tap rational resampling filter that locked up the system.  The helper
	   must always pick a value that exactly divides sample_rate so the IF
	   decimation step uses fast integer downsampling.

	2. AirSpy HF+ block-boundary click: with sample_rate=912000 and
	   audio_sample_rate=16000, the earlier helper picked if_decimation=15
	   → if_rate=60800, which is an integer divisor of 912000 but NOT a
	   multiple of 16000.  That forced the downstream decimate_audio call
	   into its rational resample_poly path, which emits a short tail
	   transient at every block boundary and produces an audible ~5 Hz click
	   in recordings.  The helper must prefer "clean chain" candidates where
	   *both* sample_rate % d == 0 AND (sample_rate // d) % audio_sample_rate
	   == 0, so the audio stage stays on its stateful integer path.
	"""

	def test_airspy_r2_2_5mhz (self):
		"""
		AirSpy R2 native rate must produce a clean integer divisor.

		No clean-chain candidate exists for 2_500_000 → 16_000 (prime
		factorisations have incompatible power-of-two factors), so the
		helper falls back to any integer divisor of 2500000 closest to
		the ideal — which is still d=40.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(2_500_000, 16_000, 4.0)
		assert 2_500_000 % dec == 0, f"if_decimation {dec} must evenly divide 2500000"
		assert dec == 40

	def test_rtlsdr_1024khz_unchanged (self):
		"""
		RTL-SDR PMR config must still pick if_decimation=16 — 16 is both
		a divisor of 1024000 AND yields a clean 64000 → 16000 audio stage,
		so it's the clean-chain winner and matches the pre-change behaviour.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(1_024_000, 16_000, 4.0)
		assert dec == 16
		assert (1_024_000 // dec) % 16_000 == 0

	def test_hackrf_2_4mhz (self):
		"""
		HackRF 2.4 MHz path prefers d=30 (clean chain) over d=40 (dirty
		chain).  At ideal=38, clean candidates in the window are
		{25, 30, 50, 75}; d=30 is closest.  Yields if_rate=80000 which
		divides cleanly into 16000 (factor 5).
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(2_400_000, 16_000, 4.0)
		assert 2_400_000 % dec == 0
		assert (2_400_000 // dec) % 16_000 == 0, \
			f"if_rate {2_400_000 // dec} must be a multiple of 16000 (clean chain)"
		assert dec == 30

	def test_hackrf_dmr_12_5mhz (self):
		"""
		HackRF DMR wide-band path picks a clean integer divisor.

		No clean-chain candidate exists for 12_500_000 → 16_000 (same
		power-of-two mismatch as AirSpy R2), so the helper falls back to
		d=200 — still the closest integer divisor to the ideal.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(12_500_000, 16_000, 4.0)
		assert 12_500_000 % dec == 0
		assert dec == 200

	def test_airspy_hf_912khz_clean_chain (self):
		"""
		AirSpy HF+ Discovery native rate picks d=19, yielding if_rate=48000
		which is exactly 3 × 16000.  This is the fix for the block-boundary
		click on the air_civil_bristol_airspyhf band: before the clean-chain
		preference was added, the helper picked d=15 (closer to the ideal
		14 but NOT a clean chain), and the audio decimation stage fell into
		the rational resample_poly path with its tail transient.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(912_000, 16_000, 4.0)
		assert dec == 19
		assert 912_000 % dec == 0
		if_rate = 912_000 // dec
		assert if_rate == 48_000
		assert if_rate % 16_000 == 0, \
			"if_rate must be a multiple of audio_sample_rate so the audio stage stays on the integer path"

	def test_clean_chain_is_preferred_over_closer_dirty_chain (self):
		"""
		Proves the clean-chain tier genuinely wins over the nearest-divisor
		tier when both are available.  For 912_000 → 16_000, the absolute
		closest integer divisor to ideal=14 is d=15 (distance 1), but d=15
		gives the dirty if_rate=60800 (not a multiple of 16000).  The
		clean-chain candidate d=19 is further from the ideal (distance 5)
		but gives the clean if_rate=48000.  The helper must pick d=19.
		"""
		dec = substation.dsp.demodulation._pick_if_decimation(912_000, 16_000, 4.0)
		# If the clean-chain preference were absent, we'd expect 15 here.
		assert dec != 15, "clean-chain preference must override nearest-divisor"
		assert dec == 19

	def test_returns_at_least_one (self):
		"""For very low sample rates, the helper must never return 0."""
		dec = substation.dsp.demodulation._pick_if_decimation(48_000, 16_000, 4.0)
		assert dec >= 1


class TestNFMAt2_5MHz:

	"""
	Regression tests proving the AirSpy R2 NFM hang is fixed.  Each test
	caps execution time at 1 second — the previously-broken path tried to
	allocate a 50-million-tap filter (~400 MB) inside scipy.signal.resample_poly
	and could lock up a 2 GB Pi for minutes (or OOM-kill it).
	"""

	def test_nfm_2_5mhz_completes_quickly (self):
		"""demodulate_nfm at 2.5 MHz must process a 200ms slice in well under 1s."""
		sr = 2_500_000
		asr = 16_000
		duration_s = 0.2

		# Generate a simple FM-modulated 1 kHz tone
		n = int(sr * duration_s)
		t = numpy.arange(n, dtype=numpy.float64) / sr
		audio_freq = 1000.0
		deviation = 2500.0
		phase = 2 * numpy.pi * deviation * numpy.cumsum(numpy.sin(2 * numpy.pi * audio_freq * t)) / sr
		iq = numpy.exp(1j * phase).astype(numpy.complex64)

		start = time.perf_counter()
		audio, _ = substation.dsp.demodulation.demodulate_nfm(iq, sr, asr)
		elapsed = time.perf_counter() - start

		assert elapsed < 1.0, f"demodulate_nfm took {elapsed:.2f}s — should be well under 1s"
		assert len(audio) > 0
		# Sanity check the recovered tone is in the right ballpark
		dominant = _dominant_freq(audio[len(audio) // 5:], asr)
		assert abs(dominant - audio_freq) < 200

	def test_am_2_5mhz_completes_quickly (self):
		"""demodulate_am at 2.5 MHz must also be fast (now uses the same IF helper)."""
		sr = 2_500_000
		asr = 16_000
		iq = iq_generators.generate_am_iq(1000.0, 0.8, sr, 0.2)

		start = time.perf_counter()
		audio, _ = substation.dsp.demodulation.demodulate_am(iq, sr, asr)
		elapsed = time.perf_counter() - start

		assert elapsed < 1.0, f"demodulate_am took {elapsed:.2f}s — should be well under 1s"
		assert len(audio) > 0


class TestRationalResampleContinuity:

	"""
	Verify the streaming polyphase resampler produces seamless audio across
	block boundaries.  The old resample_poly overlap-save approach drifted
	by ~0.74 output samples per block (cumulative phase error of ~27°/block
	at 1 kHz) and had a 3 dB SNR vs. whole-signal processing.
	"""

	IF_RATE = 62500
	AUDIO_RATE = 16000
	BLOCK_SIZE = 13107

	def test_blockwise_matches_whole_signal (self):
		"""Block-by-block output must be bit-identical to whole-signal."""
		n_blocks = 10
		total = self.BLOCK_SIZE * n_blocks
		t = numpy.arange(total) / self.IF_RATE
		signal = numpy.sin(2 * numpy.pi * 1000 * t).astype(numpy.float32)

		state = {}
		blocks = []
		for b in range(n_blocks):
			blk = signal[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_audio(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		blockwise = numpy.concatenate(blocks)

		whole_state: dict = {}
		whole, _ = substation.dsp.filters.decimate_audio(signal, self.IF_RATE, self.AUDIO_RATE, whole_state)

		n = min(len(blockwise), len(whole))
		numpy.testing.assert_array_equal(blockwise[:n], whole[:n])

	def test_phase_continuity_at_boundaries (self):
		"""Phase jumps at block boundaries must be < 1 degree."""
		n_blocks = 10
		total = self.BLOCK_SIZE * n_blocks
		t = numpy.arange(total) / self.IF_RATE
		signal = numpy.sin(2 * numpy.pi * 1000 * t).astype(numpy.float32)

		state = {}
		blocks = []
		for b in range(n_blocks):
			blk = signal[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_audio(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		audio = numpy.concatenate(blocks)

		boundaries = numpy.cumsum([len(b) for b in blocks[:-1]])
		for b in boundaries:
			if b < 50 or b + 50 >= len(audio):
				continue
			seg = audio[b - 50:b + 50]
			analytic = scipy.signal.hilbert(seg)
			phases = numpy.unwrap(numpy.angle(analytic))
			jump = abs(phases[51] - phases[50] - (phases[50] - phases[49]))
			assert jump < numpy.radians(1.0), (
				f"Phase jump {numpy.degrees(jump):.1f}° at boundary sample {b} exceeds 1°"
			)

	def test_unity_passband_gain (self):
		"""DC and voice-band signals must pass at unity gain."""
		dc = numpy.ones(50000, dtype=numpy.float32)
		state: dict = {}
		dc_out, _ = substation.dsp.filters.decimate_audio(dc, self.IF_RATE, self.AUDIO_RATE, state)
		assert abs(dc_out[200:].mean() - 1.0) < 0.01

	def test_complex_signal_supported (self):
		"""The rational path must handle complex64 IQ (used by decimate_iq)."""
		t = numpy.arange(self.BLOCK_SIZE * 3) / self.IF_RATE
		iq = numpy.exp(1j * 2 * numpy.pi * 500 * t).astype(numpy.complex64)
		state: dict = {}
		blocks = []
		for b in range(3):
			blk = iq[b * self.BLOCK_SIZE:(b + 1) * self.BLOCK_SIZE]
			out, state = substation.dsp.filters.decimate_iq(blk, self.IF_RATE, self.AUDIO_RATE, state)
			blocks.append(out)
		result = numpy.concatenate(blocks)
		assert numpy.iscomplexobj(result)
		assert len(result) > 0


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
