"""
Global constants for SDR Scanner
"""

# NFM demodulation constants
NFM_DEEMPHASIS_TAU = 300e-6  # 300µs time constant (standard for NFM)
NFM_DEVIATION_HZ = 5000      # Max frequency deviation for normalization

# AM AGC constants
AM_AGC_MIN_UPDATE_LEVEL = 0.005  # Minimum level required to update AGC (prevents noise ramp)
AM_AGC_FLOOR = 0.02          # Lower bound for AGC level to avoid boosting noise
AM_AGC_ATTACK_MS = 10.0      # AGC attack time constant in ms
AM_AGC_RELEASE_MS = 200.0    # AGC release time constant in ms
AM_OUTPUT_GAIN = 0.5         # Post-AGC gain for AM audio to reduce clipping

# RadioScanner class constants
# Hysteresis margin in dB - channel turns ON at threshold, OFF at threshold minus HYSTERESIS_DB
HYSTERESIS_DB = 3.0

# Fraction of channel spacing used as default channel width
CHANNEL_WIDTH_FRACTION = 0.84

# Number of DC bins to exclude around center frequency (RTL-SDR DC spike)
DC_SPIKE_BINS = 3

# Number of FFT segments for Welch averaging
WELCH_SEGMENTS = 8
