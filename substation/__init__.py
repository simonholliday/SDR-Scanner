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
  hysteresis and three-layer noise rejection (RF variance, audio spectral
  flatness, post-recording flatness check) to eliminate false recordings
- Audio silence timeout to stop recording when an AM carrier persists
  after voice ends
- Demodulation of NFM, AM, and SSB (USB/LSB via Weaver's method) with
  streaming polyphase FIR resampler for artifact-free block processing
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
