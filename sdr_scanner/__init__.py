"""
SDR Scanner - Software-defined radio band scanner.

A Python application for scanning and recording activity on radio bands
using RTL-SDR or HackRF hardware. Features include:
- Automatic channel detection using SNR (Signal-to-Noise Ratio)
- Support for NFM (Narrow FM) and AM modulation
- Automatic recording with noise reduction
- PPM calibration for frequency drift correction
- Broadcast WAV format with embedded metadata

Typical usage:
    python -m sdr_scanner --band pmr
    python -m sdr_scanner --list-bands
    python -m sdr_scanner --band airband --device-type hackrf
"""
