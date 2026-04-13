"""Smoke tests for tuning constants."""

import substation.constants


def test_welch_segments_minimum ():
	assert substation.constants.WELCH_SEGMENTS >= 2


def test_ema_alpha_range ():
	alpha = substation.constants.NOISE_FLOOR_EMA_ALPHA
	assert 0.0 < alpha <= 1.0


def test_warmup_slices_positive ():
	assert substation.constants.NOISE_FLOOR_WARMUP_SLICES >= 1


def test_trim_constants_positive ():
	assert substation.constants.TRIM_AMPLITUDE_THRESHOLD > 0
	assert substation.constants.TRIM_PRE_SAMPLES >= 0
	assert substation.constants.TRIM_POST_SAMPLES >= 0


def test_nfm_constants_types ():
	assert isinstance(substation.constants.NFM_DEEMPHASIS_TAU, float)
	assert isinstance(substation.constants.NFM_DEVIATION_HZ, float)
	assert substation.constants.NFM_DEVIATION_HZ > 0
	assert substation.constants.NFM_IF_OVERSAMPLE >= 2.0


def test_am_constants ():
	assert substation.constants.AM_AGC_ATTACK_MS > 0
	assert substation.constants.AM_AGC_RELEASE_MS > substation.constants.AM_AGC_ATTACK_MS
	assert 0.0 < substation.constants.AM_OUTPUT_GAIN <= 1.0
	assert substation.constants.AM_IF_OVERSAMPLE >= 2.0
