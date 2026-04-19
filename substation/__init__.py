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
  per-band configurable hysteresis and three-layer noise rejection
  (RF variance, audio spectral flatness, post-recording flatness
  check) to eliminate false recordings
- Audio silence timeout to stop recording when an AM carrier persists
  after voice ends
- Demodulation of NFM, AM, and SSB (USB/LSB via Weaver's method) with
  streaming polyphase FIR resampler for artifact-free block processing
- CTCSS (51 standard tones) and DCS (23-bit Golay-coded) subaudible tone
  detection on NFM, embedded in recording metadata
- Automatic per-channel recording in WAV (Broadcast WAV with embedded
  frequency / timestamp / modulation / tone metadata) or FLAC (lossless,
  Vorbis comments) with spectral-subtraction noise reduction and optional
  experimental dynamics-curve expander
- PPM frequency calibration against a known reference signal
- Unified event emitter (on / off / emit) — six events covering channel
  state, recording lifecycle, noise floor, and per-slice SNR snapshots;
  the channel_state event carries the detected CTCSS tone or DCS code
  as a property of each activation; used by the OSC bridge and the
  Supervisor dashboard integration
- Optional OSC event forwarding to downstream tools (MIDI sequencer,
  sampler, VJ software, ...) via substation.osc_sender — install with
  pip install -e ".[osc]"
- Optional real-time Supervisor dashboard — broadcasts scanner state
  over WebSocket for a web UI — install with
  pip install -e ".[supervisor]"

Typical usage:
    substation --band pmr
    substation --list-bands
    substation --band air_civil_1 --device-type hackrf
    substation --band air_civil_bristol_airspyhf --device-type airspyhf
"""

__version__ = "0.1.0"
