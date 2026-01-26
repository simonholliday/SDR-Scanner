"""
Global constants for SDR Scanner.

This module contains configuration constants used throughout the application.
These are separated from runtime configuration (config.yaml) because they are
typically not changed by users and represent technical parameters tuned for
specific algorithms and hardware characteristics.
"""

# ==============================================================================
# NFM (Narrow FM) Demodulation Constants
# ==============================================================================

# De-emphasis time constant (τ = RC time constant of high-pass filter)
# Transmitters pre-emphasize high frequencies to improve SNR, receivers must de-emphasize
# 300µs is the standard for narrow FM (PMR, amateur radio, etc.)
# Different from broadcast FM which uses 75µs (USA) or 50µs (Europe)
NFM_DEEMPHASIS_TAU = 300e-6  # 300 microseconds

# Maximum frequency deviation for NFM
# Used to normalize the demodulated audio to the range [-1, 1]
# NFM typically uses ±5 kHz deviation (narrow compared to broadcast FM's ±75 kHz)
NFM_DEVIATION_HZ = 2.5e3  # 2.5 kHz deviation

# ==============================================================================
# AM (Amplitude Modulation) AGC Constants
# ==============================================================================

# Automatic Gain Control (AGC) compensates for varying signal strengths
# Too-fast AGC sounds "pumpy", too-slow AGC doesn't adapt quickly enough

# Minimum signal level required to update AGC gain
# Prevents AGC from ramping up during noise-only periods (would amplify noise)
AM_AGC_MIN_UPDATE_LEVEL = 0.005  # 0.5% of full scale

# Minimum AGC gain level (floor)
# Prevents excessive amplification of noise when signal is very weak
AM_AGC_FLOOR = 0.02  # 2% minimum gain

# AGC attack time: how quickly gain decreases when signal gets stronger
# Fast attack prevents distortion from sudden loud signals
AM_AGC_ATTACK_MS = 10.0  # 10 milliseconds

# AGC release time: how quickly gain increases when signal gets weaker
# Slow release sounds more natural (avoids "pumping" artifacts)
AM_AGC_RELEASE_MS = 200.0  # 200 milliseconds

# Post-AGC output gain scaling
# AM demodulation can produce peaks, so we scale down to prevent clipping
AM_OUTPUT_GAIN = 0.5  # 50% (-6 dB)

# ==============================================================================
# Channel Detection and Scanning Constants
# ==============================================================================

# Hysteresis margin for channel state detection
# Channel turns ON when SNR > threshold, OFF when SNR < (threshold - HYSTERESIS_DB)
# This prevents rapid on/off toggling (chattering) when SNR hovers near threshold
# 3 dB is ~2x power ratio, provides stable switching
HYSTERESIS_DB = 3.0  # 3 dB margin

# Fraction of channel spacing to use as channel bandwidth
# For example, with 12.5 kHz spacing: channel width = 12.5 * 0.84 = 10.5 kHz
# Leaves a small guard band (0.16 * spacing) between channels to reduce crosstalk
# 0.84 is empirically chosen to balance channel separation vs signal capture
CHANNEL_WIDTH_FRACTION = 0.84

# Number of FFT bins to exclude around DC (0 Hz offset from center frequency)
# Most SDR receivers have a DC spike caused by LO leakage and I/Q imbalance
# Excluding ±3 bins typically removes the spike without losing too much signal
# For a 2 MHz sample rate with 8192-bin FFT: ±3 bins = ±732 Hz excluded
DC_SPIKE_BINS = 3

# Number of overlapping segments for Welch's method of PSD estimation
# More segments = lower variance (smoother PSD) but lower frequency resolution
# 8 segments provides good balance: 50% overlap gives 15 independent estimates
# Higher values reduce noise but make narrowband signals harder to distinguish
WELCH_SEGMENTS = 8
