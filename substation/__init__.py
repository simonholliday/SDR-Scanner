"""
Substation - Software-defined radio band scanner.

A Python application for scanning and recording activity on radio bands.

Supported hardware:
- RTL-SDR (native driver)
- HackRF One (native driver)
- AirSpy R2, AirSpy HF+ Discovery, and any other SoapySDR-supported
  device (via the SoapySDR wrapper)

Features:
- Automatic channel detection using SNR (Signal-to-Noise Ratio) with
  hysteresis and temporal variance checks to reject stationary noise
- Demodulation of NFM, AM, and SSB (USB/LSB via Weaver's method)
- Automatic per-channel recording with spectral-subtraction noise
  reduction and optional experimental dynamics-curve expander
- PPM frequency calibration against a known reference signal
- Broadcast WAV (BWF) output with embedded frequency / timestamp /
  modulation metadata
- Optional OSC event forwarding to downstream tools (MIDI sequencer,
  sampler, VJ software, ...) via substation.osc_sender — install with
  pip install -e ".[osc]"

Typical usage:
    substation --band pmr
    substation --list-bands
    substation --band air_civil_1 --device-type hackrf
    substation --band air_civil_bristol_airspyhf --device-type airspyhf
"""

__version__ = "0.1.0"
