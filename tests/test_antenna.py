"""Tests for the scripts/antenna.py utility."""

import math

import pytest

import scripts.antenna


class TestAntennaCalculator:

	"""Unit tests for the pure functions in scripts.antenna."""

	def test_dipole_at_4mhz (self):

		"""At 4 MHz the half-wave dipole should be 0.95 * c / 4e6 / 2 ≈ 35.601 m."""

		expected = 0.95 * 299_792_458.0 / 4_000_000.0 / 2.0
		result = scripts.antenna.compute_antenna_lengths(4_000_000.0)

		assert math.isclose(result['dipole_total'], expected, abs_tol=1e-3)
		assert math.isclose(result['dipole_leg'], expected / 2.0, abs_tol=1e-3)

	def test_dipole_at_uvb76 (self):

		"""4625 kHz (UVB-76) should give the worked-example values from the report."""

		result = scripts.antenna.compute_antenna_lengths(4_625_000.0)

		# Total dipole ≈ 30.79 m, each leg ≈ 15.39 m
		assert result['dipole_total'] == pytest.approx(30.79, abs=0.01)
		assert result['dipole_leg'] == pytest.approx(15.39, abs=0.01)

	def test_quarter_wave_at_446mhz (self):

		"""For PMR446 (446.1 MHz centre), λ/4 with the 0.95 factor ≈ 16 cm."""

		result = scripts.antenna.compute_antenna_lengths(446_100_000.0)

		# 0.95 * 299_792_458 / 446.1e6 / 4 ≈ 0.1596 m
		assert result['quarter_wave_vertical'] == pytest.approx(0.1596, abs=0.001)

	def test_full_wave_loop_uses_loop_factor (self):

		"""Full-wave loop perimeter at 4625 kHz should use the 0.97 factor, not 0.95."""

		result = scripts.antenna.compute_antenna_lengths(4_625_000.0)

		expected = 0.97 * 299_792_458.0 / 4_625_000.0
		assert result['full_wave_loop'] == pytest.approx(expected, abs=1e-3)

		# Sanity check: must differ from what 0.95 would have produced
		wrong = 0.95 * 299_792_458.0 / 4_625_000.0
		assert not math.isclose(result['full_wave_loop'], wrong, abs_tol=0.01)

	def test_five_eighths_vertical_no_velocity_factor (self):

		"""5/8-wave should be exactly 0.625 * λ — no wire correction applied."""

		result = scripts.antenna.compute_antenna_lengths(4_625_000.0)

		expected = 0.625 * 299_792_458.0 / 4_625_000.0
		assert result['five_eighths_vertical'] == pytest.approx(expected, abs=1e-6)

		# Sanity check: must differ from what 0.95 * 0.625 * λ would give
		wrong = 0.95 * 0.625 * 299_792_458.0 / 4_625_000.0
		assert not math.isclose(result['five_eighths_vertical'], wrong, abs_tol=0.01)

	def test_zero_frequency_raises (self):

		"""compute_antenna_lengths(0) should raise ValueError."""

		with pytest.raises(ValueError, match="positive"):
			scripts.antenna.compute_antenna_lengths(0.0)

	def test_negative_frequency_raises (self):

		"""compute_antenna_lengths(-1) should raise ValueError."""

		with pytest.raises(ValueError, match="positive"):
			scripts.antenna.compute_antenna_lengths(-1.0)

	def test_format_report_band_with_warning (self):

		"""hf_night_4mhz spans ~14.7% — the warning footer must appear."""

		report = scripts.antenna.format_antenna_report(
			4_425_000.0,
			band_name='hf_night_4mhz',
			freq_start_hz=4_100_000.0,
			freq_end_hz=4_750_000.0,
		)

		assert 'hf_night_4mhz' in report
		assert 'NOTE' in report
		assert '14.7%' in report
		assert 'dipole' in report.lower()
		assert '5/8-wave' in report
		assert 'loop' in report.lower()

		# Edge-frequency lengths should appear in the footer
		assert '4.100 MHz' in report
		assert '4.750 MHz' in report

	def test_format_report_narrow_band_no_warning (self):

		"""PMR's 0.04% spread is well below the 4% threshold — no warning."""

		report = scripts.antenna.format_antenna_report(
			446_100_000.0,
			band_name='pmr',
			freq_start_hz=446_006_250.0,
			freq_end_hz=446_193_750.0,
		)

		assert 'pmr' in report
		assert 'NOTE' not in report

	def test_format_report_freq_only (self):

		"""Calling without band metadata should use the 'Frequency:' header."""

		report = scripts.antenna.format_antenna_report(4_625_000.0)

		assert 'Frequency:' in report
		assert 'Band:' not in report
		assert 'NOTE' not in report
