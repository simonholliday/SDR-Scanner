# Substation

## Overview
Substation is a high-performance tool for monitoring and recording radio activity using Software Defined Radio (SDR) hardware. It is designed to be used in two ways:

1.  **As a Command-Line Tool**: Quickly scan and record bands using simple terminal commands.
2.  **As a Python Module**: Integrate radio scanning, detection, and callbacks directly into your own Python applications.

By connecting a supported USB receiver (like an RTL-SDR or HackRF), you can scan wide ranges of the radio spectrum - such as Airband or Maritime frequencies - and automatically record transmissions as they occur. The software handles the technical signal processing and hardware management in the background, allowing for efficient 24/7 monitoring even on modest hardware like a Raspberry Pi.

## Contents

- [Supported Devices](#supported-devices)
    - [Quick Reference](#quick-reference)
    - [RTL-SDR Blog V4 / V3](#rtl-sdr-blog-v4-v3)
    - [HackRF One](#hackrf-one)
    - [AirSpy R2](#airspy-r2)
    - [AirSpy HF+ Discovery](#airspy-hf-discovery)
    - [Other SoapySDR Devices](#other-soapysdr-devices)
- [Key Features & Optimizations](#key-features-optimizations)
- [Quick Start](#quick-start)
- [Utility scripts](#utility-scripts)
- [Command Line](#command-line)
- [Python Module Usage](#python-module-usage)
    - [OSC event forwarding](#osc-event-forwarding)
- [Configuration](#configuration)
- [SoapySDR Installation](#soapysdr-installation-airspy-and-other-devices)
- [Broadcast WAV Metadata](#broadcast-wav-metadata)
- [Gain Tuning](#gain-tuning)
- [Rejecting Empty/Noise Recordings](#rejecting-emptynoise-recordings)
- [Dynamics Curve (Experimental)](#dynamics-curve-experimental)
- [Parallel Scans](#parallel-scans-multiple-devices)
- [Resource and Performance Notes](#resource-and-performance-notes)
- [Limitations](#limitations)
- [Author](#author)
- [License](#license)

## Supported Devices

To use this software, a compatible Software Defined Radio (SDR) USB device is required. Each supported device below has a self-contained card with its specifications, recommended starting configuration, common gotchas, and a copy-pasteable example band so you can get a working scan in a few minutes. Different SDR devices have very different capabilities — settings that work well on one device may need adjusting on another, and the cards capture the differences that actually matter in practice.

### Quick Reference

| Device                  | Frequency range                | Max BW   | ADC    | Best for                  |
| :---------------------- | :----------------------------- | :------- | :----- | :------------------------ |
| RTL-SDR Blog V4 / V3    | 24 MHz - 1.766 GHz             | 2.4 MHz  | 8-bit  | General VHF/UHF, low cost |
| HackRF One              | 1 MHz - 6 GHz                  | 20 MHz   | 8-bit  | Wideband monitoring       |
| AirSpy R2               | 24 MHz - 1.8 GHz               | 10 MHz   | 12-bit | High-quality VHF/UHF      |
| AirSpy HF+ Discovery    | 0.5 kHz - 31 MHz, 60 - 260 MHz | 768 kHz  | 18-bit | HF / VHF precision        |

Any other device with a SoapySDR driver module installed can be used too — see [Other SoapySDR Devices](#other-soapysdr-devices) below.

### RTL-SDR Blog V4 / V3

A high-quality, low-cost general-purpose receiver. The natural starting point for new users — well-supported, easy to drive, and good enough for most VHF/UHF scanning. Limited dynamic range from its 8-bit ADC.

| Spec               | Value                                                  |
| :----------------- | :----------------------------------------------------- |
| Frequency range    | 24 MHz - 1.766 GHz (with gaps)                         |
| Max bandwidth      | 2.4 MHz                                                |
| Sample rates       | Continuous, up to 2.4 MHz (typical: 2.048 MHz)         |
| ADC resolution     | 8-bit                                                  |
| Gain architecture  | Single stage                                           |
| AGC                | Hardware AGC                                           |
| Driver             | `pyrtlsdr` (>=0.3.0,<0.4.0) — Python binding           |
| `--device-type`    | `rtl`, `rtlsdr`, `rtl-sdr`                             |
| Best for           | General VHF/UHF scanning, low cost, easy setup         |

**Setup** — see [installation.txt §1](installation.txt) for the librtlsdr fork build and the DVB-T driver blacklist step.

**Recommended starting config**
- `snr_threshold_db: 4.5`
- `sdr_gain_db: auto` (engages hardware AGC, which is well-tuned for most bands)
- `activation_variance_db: 3.0` (default — leave alone unless you see false triggers)
- `sample_rate: 2.048e6` for most bands

**Gotchas**
- The Blog V4 needs the [rtl-sdr-blog fork](https://github.com/rtlsdrblog/rtl-sdr-blog) of librtlsdr. The standard distro `librtlsdr` is missing the `rtlsdr_set_dithering` symbol that newer pyrtlsdr versions need; this is why the project pins `pyrtlsdr<0.4.0`.
- The default Linux DVB-T driver claims the device on insertion as a TV tuner — it must be blacklisted (installation.txt covers this).
- The 8-bit ADC limits dynamic range. A strong adjacent station can desensitise weak ones in the same capture.
- Manual gain values are typically 20-40 dB if you don't want AGC.

**Working example band**

```yaml
air_civil_bristol:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 1.024e6
```

**References**
- Manufacturer page: [https://www.rtl-sdr.com/about-rtl-sdr/](https://www.rtl-sdr.com/about-rtl-sdr/)
- Driver fork: [https://github.com/rtlsdrblog/rtl-sdr-blog](https://github.com/rtlsdrblog/rtl-sdr-blog)
- Python binding: [https://github.com/pyrtlsdr/pyrtlsdr](https://github.com/pyrtlsdr/pyrtlsdr)

### HackRF One

A wideband transceiver covering 1 MHz to 6 GHz with up to 20 MHz of instantaneous bandwidth — by far the widest single-tune capture of any device here. The trade-off is no hardware AGC and the same 8-bit ADC dynamic-range limit as the RTL-SDR.

| Spec               | Value                                                              |
| :----------------- | :----------------------------------------------------------------- |
| Frequency range    | 1 MHz - 6 GHz                                                      |
| Max bandwidth      | 20 MHz (16 MHz is the practical reliable maximum)                  |
| Sample rates       | Continuous, 2 - 20 MHz                                             |
| ADC resolution     | 8-bit                                                              |
| Gain architecture  | LNA (0-40 dB, 8 dB steps) + VGA (0-62 dB, 2 dB steps)              |
| AGC                | None — `auto` falls back to a sensible default and warns           |
| Driver             | `python_hackrf` (with fallback to `hackrf` / `pyhackrf`)           |
| `--device-type`    | `hackrf`, `hackrf-one`, `hackrfone`                                |
| Best for           | Wideband monitoring, multi-band capture in a single tune           |

**Setup** — see [installation.txt §2](installation.txt) for the USB buffer tuning (`usbcore.usbfs_memory_mb=1000`) and [installation.txt §3](installation.txt) for the `libhackrf-dev` system package.

**Recommended starting config**
- `snr_threshold_db: 6`
- `sdr_gain_db: 36` (or `auto` to accept the LNA=32 / VGA=30 default)
- `activation_variance_db: 3.0`
- `sample_rate: 16e6` for the widest single capture; lower (2-4 MHz) for narrow bands

**Gotchas**
- **No hardware AGC.** Setting `sdr_gain_db: auto` does not enable AGC — there isn't one. The wrapper logs a warning and sets sensible defaults (LNA=32, VGA=30) so the device still works.
- A numeric `sdr_gain_db` is silently clamped and stepped to the LNA's 8 dB grid and the VGA's 2 dB grid. Asking for 35 dB gets you 32. Check the startup log if the actual values matter.
- High sample rates (~16-20 MHz) require raising the kernel USB buffer limit; otherwise samples will be dropped. See [installation.txt §2](installation.txt).
- The 8-bit ADC has the same dynamic-range caveats as the RTL-SDR — wide captures including a strong station can desensitise weak ones.
- Multiple Python bindings exist (`python_hackrf`, `hackrf`, `pyhackrf`) with different APIs; the wrapper auto-detects whichever is installed.

**Working example band**

```yaml
dmr:
    type: DMR
    freq_start: 452.5e+6
    freq_end: 460.5e+6
    sample_rate: 12.5e+6
```

**References**
- Manufacturer page: [https://greatscottgadgets.com/hackrf/one/](https://greatscottgadgets.com/hackrf/one/)
- Python binding: [https://pypi.org/project/python-hackrf/](https://pypi.org/project/python-hackrf/)

### AirSpy R2

A high-dynamic-range VHF/UHF receiver with a 12-bit ADC (≈16-bit effective from oversampling) and three independently tuneable gain stages. Considerably more sensitive than the RTL-SDR for the same money tier, with enough bandwidth (10 MHz) to cover practical surveillance bands in a single tune.

| Spec               | Value                                                                         |
| :----------------- | :---------------------------------------------------------------------------- |
| Frequency range    | 24 MHz - 1.8 GHz                                                              |
| Max bandwidth      | 10 MHz                                                                        |
| Sample rates       | Discrete: 2.5 MHz or 10 MHz                                                   |
| ADC resolution     | 12-bit (≈16-bit effective from oversampling)                                  |
| Gain architecture  | LNA + Mixer + VGA (per-element control via `sdr_gain_elements`)               |
| AGC                | None — `sdr_gain_db: auto` is mapped to a fixed manual default (see below)    |
| Driver             | SoapySDR + `soapysdr-module-airspy` (system package)                          |
| `--device-type`    | `airspy`, `airspy-r2`, `airspyr2`                                             |
| Best for           | High-quality VHF/UHF, wide single-band capture, weak-signal work              |

**Setup** — see [installation.txt §4](installation.txt) for the SoapySDR core and the AirSpy module. The Python venv **must** be created with `--system-site-packages` so it can access the system-installed SoapySDR Python bindings.

**Recommended starting config**
- `snr_threshold_db: 6` (the higher sensitivity makes the RTL default 4.5 dB too noisy)
- `sdr_gain_db: auto` is fine to start with — see the AGC gotcha below for what it actually does
- `activation_variance_db: 3.0`
- `sample_rate: 2.5e6` for narrow bands, `10e6` for wide ones

**Gotchas**
- **Sample rates are discrete.** Asking for anything other than 2.5 MHz or 10 MHz silently snaps to the nearest supported rate and logs a warning. Always check the startup log to confirm the rate the device actually accepted.
- **`sdr_gain_db: auto` is not real AGC.** SoapyAirspy reports `hasGainMode == True` but the underlying R2 hardware does not provide a working closed-loop AGC. Substation detects this and falls back to a fixed manual gain of `LNA=10, MIX=5, VGA=12` (27 dB total) — the same LNA-first values you would set by hand. This works well for typical PMR / VHF / UHF reception. If you want different values, set `sdr_gain_db` (numeric) or `sdr_gain_elements` (per-stage dict) explicitly in your band config.
- For per-element tuning, **maximise LNA first**, set Mixer moderate, fine-tune with VGA (this is the LNA-first principle described in [Gain Tuning](#gain-tuning) below). The element names and ranges are logged at INFO level when the device starts up.
- Requires a venv built with `--system-site-packages`.

**Working example band** — PMR446 with per-element gain control:

```yaml
pmr_airspy:
    type: PMR
    freq_start: 446.00625e+6
    freq_end: 446.19375e+6
    sample_rate: 2.5e6
    sdr_gain_elements:
      LNA: 10
      MIX: 5
      VGA: 12
```

Run with:

```bash
substation --band pmr_airspy --device-type airspy --device-index 0
```

**References**
- Manufacturer page: [https://airspy.com/airspy-r2/](https://airspy.com/airspy-r2/)
- SoapySDR driver: [https://github.com/pothosware/SoapyAirspy](https://github.com/pothosware/SoapyAirspy)
- SoapySDR project: [https://github.com/pothosware/SoapySDR](https://github.com/pothosware/SoapySDR)

### AirSpy HF+ Discovery

A precision HF and lower-VHF receiver. Exceptional sensitivity and dynamic range in its bands; not a wideband scanner — its maximum bandwidth is 768 kHz. Best in class for HF listening, weak-signal work, and narrow-band airband / amateur scanning.

| Spec               | Value                                                                              |
| :----------------- | :--------------------------------------------------------------------------------- |
| Frequency range    | 0.5 kHz - 31 MHz, 60 - 260 MHz (two separate bands, not contiguous)                |
| Max bandwidth      | 768 kHz                                                                            |
| Sample rates       | Discrete: typically 0.192, 0.228, 0.384, 0.456, 0.650, 0.768, 0.912 MHz (see log)  |
| ADC resolution     | 18-bit                                                                             |
| Gain architecture  | LNA on/off (0 or +6 dB) + RF *attenuator* (-48 to 0 dB)                            |
| AGC                | Hardware multi-loop AGC (recommended starting point)                               |
| Driver             | SoapySDR + `soapysdr-module-airspyhf` (system package)                             |
| `--device-type`    | `airspyhf`, `airspy-hf`, `airspyhf+`                                               |
| Best for           | HF and lower-VHF precision work, weak-signal listening, narrow-band scanning       |

**Setup** — see [installation.txt §4](installation.txt). On Raspberry Pi OS the `soapysdr-module-airspyhf` package may not be available in the distro repos; the install guide covers building it from source. As with the AirSpy R2, the venv **must** be created with `--system-site-packages`.

**Recommended starting config**
- `snr_threshold_db: 6` (essential — the device is sensitive enough that the RTL default 4.5 dB triggers on near-noise)
- `sdr_gain_db: auto` (engages the well-tuned hardware multi-loop AGC)
- `activation_variance_db: 3.0` (**also essential** — without it the high sensitivity surfaces stationary noise as false channel activations; see [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings))
- `sample_rate: 0.912e6` for the widest capture

**Gotchas**
- **Sample rates are discrete.** The exact list depends on firmware — check the startup log for the rates your device actually reports. Asking for an unsupported rate silently snaps to the nearest and logs a warning.
- **The RF gain element is an *attenuator*, not an amplifier.** Negative dB. `RF: 0` means *no* attenuation (maximum signal); `RF: -24` means 24 dB of attenuation. This is the opposite of every other device here.
- The LNA is binary (0 or 6 dB) — there is no smooth manual control of the front end.
- **CF32 samples are delivered well below the [-1, 1] range** that the demodulator expects. The wrapper auto-calibrates this on startup by measuring the median RMS of warmup blocks and applying a normalisation scale; you'll see an `IQ calibration: ...` line in the startup log. No user action required.
- Front-end overload looks like duplicate signals on adjacent channels. If you see them, increase RF attenuation (`RF: -24` or lower).
- Requires a venv built with `--system-site-packages`.

**Working example band** — Bristol airband:

```yaml
air_civil_bristol_airspyhf:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 0.912e6
    snr_threshold_db: 6
    sdr_gain_db: auto
    activation_variance_db: 3.0
```

Run with:

```bash
substation --band air_civil_bristol_airspyhf --device-type airspyhf --device-index 0
```

**References**
- Manufacturer page: [https://airspy.com/airspy-hf-discovery/](https://airspy.com/airspy-hf-discovery/)
- SoapySDR driver: [https://github.com/pothosware/SoapyAirspyHF](https://github.com/pothosware/SoapyAirspyHF)
- SoapySDR project: [https://github.com/pothosware/SoapySDR](https://github.com/pothosware/SoapySDR)

### Other SoapySDR Devices

Any device with a SoapySDR driver module installed can be used via `--device-type soapy:<driver>` (for example, `soapy:lime` or `soapy:plutosdr`). To discover what's connected and what driver name to use, run:

```bash
SoapySDRUtil --find
```

The same `sdr_gain_db`, `sdr_gain_elements`, and `sdr_device_settings` config keys apply, and the wrapper's startup log will show the available gain elements, sample rates, antennas, and device-specific settings reported by the driver — use these to guide your configuration in the same way as the AirSpy cards above.

**Reference:** [SoapySDR project](https://github.com/pothosware/SoapySDR)

## Key Features & Optimizations
- **Advanced Signal Detection**: Uses Welch's Power Spectral Density (PSD) estimation for stable, low-variance activity detection. The noise floor is EMA-smoothed across slices to eliminate jitter, with a warmup period that absorbs SDR hardware startup transients before detection begins.
- **Three-Layer Noise Rejection**: Eliminates the "empty hiss" recordings that plague high-sensitivity receivers. Layer 1: RF power variance rejects stationary noise at the PSD level. Layer 2: spectral flatness of speculatively demodulated audio rejects noise that passes the variance check. Layer 3: post-recording flatness check discards files where signal content was brief relative to hold-timer padding. All three are modulation-agnostic. See [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings) below.
- **Parallel Multi-Channel Recording**: Simultaneously detects and records all active channels in a band - unlike traditional handheld scanners which only play one channel at a time.
- **High-Fidelity Demodulation**: Implements stateful AM and NFM demodulation with continuous phase tracking and DC-blocking, eliminating pops and discontinuities between audio blocks.
- **Precise Transition Trimming**: After coarse PSD-based detection, demodulated audio is scanned at sample level to find exact signal boundaries. Padding is added around the boundary and faded with a half-cosine S-curve, preserving signal content (including attack transients) while eliminating clicks.
- **Hardware Efficiency**:
    - **Vectorized Math**: Heavy processing is delegated to NumPy and SciPy for maximum throughput.
    - **Zero-Copy Architecture**: Uses memory stride tricks for overlapping FFT segments, avoiding expensive data copying.
    - **Lazy Evaluation**: Computationally expensive segment analysis is performed only when transitions are detected, drastically reducing idle CPU load.
    - **Pre-Allocated Ring Buffer**: Per-channel audio buffering uses a fixed NumPy array with modulo wrap-around, eliminating per-flush concatenation and GC pressure.
- **State-of-the-Art Processing**:
    - **Vectorized AGC**: High-quality, smooth automatic gain control for AM with independent attack and release timings.
    - **Noise-Floor-Guided Spectral Subtraction**: The band-wide PSD noise floor is passed to the spectral subtraction stage for more reliable noise frame classification, reducing musical noise artifacts compared to percentile-only heuristics.
    - **Float64 Filter State**: IIR filter states (channel extraction, decimation) use double precision to prevent rounding drift in long-running sessions.
- **Parallel Scanning**: Supports multiple SDR devices (RTL-SDR and HackRF) simultaneously with asynchronous I/O.
- **Archive Ready**: Automatic recording to Broadcast WAV (BWF) with embedded metadata (frequency, timestamps, modulation).

## Quick Start
1) Install dependencies (see `installation.txt` for SDR drivers).
2) Configure bands in `config.yaml`.
3) Install package in editable mode:
```bash
pip install -e .
```
4) Run:

```bash
substation --band air_civil_bristol --device-type rtlsdr --device-index 0
```

Audio files are written to:
```
./audio/YYYY-MM-DD/<band>/<timestamp>_<band>_<channel>_<snr>dB_<device>_<index>.wav
```

## Utility scripts

Substation ships with a small `scripts/` directory of one-shot user utilities. These are not part of the main scanner — they're tools that read the config or work with frequencies, and are run with `python -m scripts.<name>`.

### Antenna length calculator

Calculate optimal antenna lengths (half-wave dipole, quarter-wave vertical, 5/8-wave vertical, full-wave loop) for any configured band or any frequency:

```bash
python -m scripts.antenna --band hf_night_4mhz   # use a configured band's centre frequency
python -m scripts.antenna --freq 4625e3          # use a manual frequency in Hz
python -m scripts.antenna --list                 # list all configured bands
```

For HF bands wider than ±2% of their centre frequency the report also shows the dipole's natural SWR window and the antenna lengths at the band edges, so you can decide whether to cut for the centre, an edge, or use a tuner. Lengths are reported in metres for HF/VHF and centimetres for UHF.

## Command Line
```bash
substation --band <band> [--config <path>] [--device-type rtlsdr|hackrf|airspy|airspyhf|soapy:<driver>] [--device-index N]
substation --list-bands
```

## Python Module Usage
You can also use the scanner as a library in your own code. This allows you to respond to radio events programmatically.

```python
import asyncio

import substation.config
import substation.scanner

# State Callback: Triggered whenever a signal starts or stops
def my_state_handler (band: str, ch: int, active: bool, snr: float) -> None:
	print (f"Channel {ch} is now {'ON' if active else 'OFF'} ({snr:.1f} dB)")

# Recording Callback: Triggered when a file is finalized and closed
def my_recording_handler (band: str, ch: int, file_path: str) -> None:
	print (f"Recording finished: {file_path}")

async def main () -> None:

	"""
	Initialize the scanner and respond to real-time events.
	"""

	# Load configuration
	config_data = substation.config.load_config ("config.yaml")

	# Initialize scanner instance
	scanner = substation.scanner.RadioScanner (
		config=config_data,
		band_name="pmr",
		device_type="rtlsdr"
	)

	# Register the handlers
	scanner.add_state_callback (my_state_handler)
	scanner.add_recording_callback (my_recording_handler)

	# Start the asynchronous scan loop
	await scanner.scan ()

if __name__ == "__main__":
	asyncio.run (main ())
```

See [examples/scan_demo.py](examples/scan_demo.py) for a more detailed implementation.

### OSC event forwarding

Substation can forward channel state changes and saved recordings as OSC (Open Sound Control) messages, so downstream tools — MIDI sequencers, sample players, VJ software, lighting rigs — can react to radio activity in real time. Install the optional dependency:

```bash
pip install -e ".[osc]"
```

Then attach an `OscEventSender` to any `RadioScanner` instance:

```python
import substation.osc_sender

osc_sender = substation.osc_sender.OscEventSender(
    host='127.0.0.1', port=9000,          # sequencer endpoint
    sampler_host='127.0.0.1',             # optional: also notify a sampler
    sampler_port=9002,
)
osc_sender.attach(scanner)
```

The sender emits the following OSC messages:

| Address | When | Arguments |
| :--- | :--- | :--- |
| `/radio/state` | Channel turns ON or OFF | `band_name:str, channel_index:int, is_active:int(0/1), snr_db:float` |
| `/radio/recording` | Recording finalised on disk | `band_name:str, channel_index:int, file_path:str` |
| `/sample/import` | Recording finalised (only if `sampler_host` set) | `file_path:str` |

Sends are non-blocking UDP (fire-and-forget); transient socket errors are logged as warnings and never raised back into the scanner. See [examples/scan_osc.py](examples/scan_osc.py) for a working script.

Options:
- `--config`, `-c`: path to config file (default `config.yaml`).
- `--band`, `-b`: band name to scan (required unless `--list-bands`).
- `--device-type`, `-t`: `rtlsdr`, `hackrf`, `airspy`, `airspyhf`, or `soapy:<driver>` (default `rtlsdr`).
- `--device-index`, `-i`: device index (default `0`).
- `--list-bands`: list available bands and exit.

## Configuration
Config file is YAML. The top-level keys are `scanner`, `recording`, `band_defaults`, and `bands`.

Scanner
```
scanner:
  sdr_device_sample_size: 131072
  band_time_slice_ms: 200
  sample_queue_maxsize: 30
  calibration_frequency_hz: 93.7e+6
  stuck_channel_threshold_seconds: 60
```
- `sdr_device_sample_size`: number of IQ samples per SDR callback. Higher values reduce callback overhead but increase latency.
- `band_time_slice_ms`: time slice used for PSD/SNR detection. Must be a multiple of `sdr_device_sample_size` (rounded up internally).
- `sample_queue_maxsize`: async queue depth. 10-50 is typical; higher tolerates bursts but uses more RAM.
- `calibration_frequency_hz`: optional known signal for PPM correction; set to `null` to disable.
- `stuck_channel_threshold_seconds`: optional duration in seconds after which a constant signal will trigger a "Stuck Channel" warning. Useful for identifying interference or stuck transmitters. Set to `null` to disable.

Recording
```
recording:
  buffer_size_seconds: 30
  disk_flush_interval_seconds: 5
  audio_sample_rate: 16000
  audio_output_dir: "./audio"
  fade_in_ms: 15
  fade_out_ms: 50
  soft_limit_drive: 1.25
```
- `buffer_size_seconds`: max in-memory audio per channel before drops.
- `disk_flush_interval_seconds`: how often to flush to disk.
- `audio_sample_rate`: output WAV rate (Hz).
- `fade_in_ms`/`fade_out_ms`: half-cosine fades applied to the padding region at channel start/stop (signal content is never attenuated).
- `soft_limit_drive`: post-processing soft limiter drive. Typical range 1.5-3.0 (higher = stronger limiting).
- `noise_reduction_enabled`: toggle spectral subtraction noise reduction (default: true).
- `recording_hold_time_ms`: duration in ms to continue recording after signal drops below threshold (default: 500).
- `discard_empty_enabled`: automatically discard noise-only recordings using spectral flatness analysis (default: true). Applies at two points: before activation (rejects noise triggers without starting a recording) and after recording close (catches recordings that became mostly noise). See [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings).
- `min_recording_seconds`: discard recordings shorter than this duration (default: 0.5). Catches brief transients (radar pulses, ignition noise) that pass the spectral checks but produce useless sub-second files. Set to `0` to disable.
- `audio_silence_timeout_ms`: stop recording when demodulated audio has been silent for this duration (default: 3000). Catches AM carriers that persist after voice stops, where RF SNR stays above threshold but there is no useful content. Set to `0` to disable and rely on RF-only detection.
- `trim_carrier_transients`: remove the sharp key-on/key-off click transients that AM transmitters produce (default: false). Only trims transients bordered by silence — voice transients (consonants) are never affected. Recommended for AM airband listening.

Band Defaults
```
band_defaults:
  AIR:
    channel_spacing: 8.333e+3
    modulation: AM
    snr_threshold_db: 4.5
    sdr_gain_db: 30
```
These settings are merged into each band of the same `type`.

Bands
```
bands:
  air_civil_bristol:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 1.0e+6
    exclude_channel_indices: [33, 34]
```
Per-band keys:
- `freq_start` / `freq_end`: Hz.
- `channel_spacing`: Hz.
- `sample_rate`: Hz. Must cover the band plus margins; higher rates increase CPU.
- `channel_width`: optional; defaults to `channel_spacing * 0.84`.
- `type`: used to inherit defaults from `band_defaults`.
- `modulation`: `AM`, `NFM`, `USB`, or `LSB`. USB/LSB use a Weaver-method SSB demodulator and are the right choice for HF voice — amateur convention is LSB below 10 MHz, USB above 10 MHz; HFGCS, VOLMET, and marine HF are all USB.
- `recording_enabled`: enable recording for this band. Optional, defaults to `false` (can also be set in `band_defaults`).
- `snr_threshold_db`: detection threshold (dB above noise floor).
- `activation_variance_db`: optional minimum power variance (dB) across the detection window required for a channel to be considered active. Filters out stationary-noise triggers. Applies to all bands regardless of recording state. See [Rejecting empty/noise recordings](#rejecting-emptynoise-recordings) below. Defaults to `3.0`; set to `0` to disable.
- `sdr_gain_db`: numeric or `auto`.
- `sdr_gain_elements`: optional dict mapping gain element names to dB values for per-stage control (e.g., `{LNA: 10, MIX: 5, VGA: 12}`). Available elements are logged at startup. Takes priority over `sdr_gain_db`.
- `sdr_device_settings`: optional dict of device-specific settings passed via SoapySDR (e.g., `{biastee: "true"}`). Available settings are logged at DEBUG level on startup.
- `exclude_channel_indices`: 1-based channel numbers to skip (no analysis, no recording). These match the channel numbers shown in log output and filenames.

## SoapySDR Installation (AirSpy and other devices)

AirSpy devices (and any other `soapy:<driver>` device) require SoapySDR, which is installed at the system level:

```bash
# Raspberry Pi OS / Debian
sudo apt install -y soapysdr-tools python3-soapysdr
sudo apt install -y soapysdr-module-airspy      # AirSpy R2
sudo apt install -y soapysdr-module-airspyhf    # AirSpy HF+ Discovery

# If soapysdr-module-airspyhf is not in your distro's repos (e.g., Raspberry Pi OS),
# build from source instead:
sudo apt install -y libairspyhf-dev libsoapysdr-dev cmake
git clone https://github.com/pothosware/SoapyAirspyHF.git
cd SoapyAirspyHF && mkdir build && cd build
cmake .. && make && sudo make install && cd ../..

# Verify SoapySDR can see connected devices
SoapySDRUtil --find
```

The Python virtual environment **must** be created with `--system-site-packages` to access the system-installed SoapySDR bindings:

```bash
python3 -m venv --system-site-packages /home/si/venvs/substation
```

## Broadcast WAV Metadata
Each recording captures industry-standard **Broadcast WAV (BWF)** metadata (EBU Tech 3285). This embeds technical details directly into the audio file, making it ideal for archival and automated post-processing.

**Compatibility**: These are standard `.wav` files. They will play perfectly in any normal audio player (VLC, Windows Media Player, Audacity, mobile devices, etc.).

### Metadata Example
If you open a recording in a professional audio tool or a BWF viewer, you will see fields like these:

| Field | Example Value | Description |
| :--- | :--- | :--- |
| **Description** | `{"band":"pmr","channel_index":0,"channel_freq":446006250.0}` | Machine-readable JSON with channel details |
| **Coding History** | `A=PCM,F=16000,W=16,M=mono,T=NFM;Frequency=446.00625MHz` | Technical signal chain (Algorithm, Rate, Modulation) |
| **Originator** | `Substation` | The software that created the file |
| **Origination Date** | `2026-01-27` | Date the recording started |
| **Time Reference** | `1152000` | Sample count since midnight (for precise timing) |


## Gain Tuning

Each device card above carries the gain settings that work as a starting point for that specific device. This section explains the *why* behind those settings — the principles that apply to any SDR with multiple gain stages, so you can reason about adjustments when the defaults aren't quite right.

SDR gain controls how much the received signal is amplified before digitisation. Too little gain and weak signals are lost in the noise floor; too much and strong signals overdrive the ADC, causing distortion and spurious detections.

**Simple approach (recommended starting point)**: set `sdr_gain_db` to a numeric value or `auto`. When set to a single number, the driver distributes the gain across the device's internal stages automatically — this produces good results for most setups without any per-element knowledge. Start here and only move to per-element tuning if you want to squeeze out the last bit of performance.

**Per-element tuning (advanced)**: devices with multiple gain stages (like the AirSpy R2) allow individual control via `sdr_gain_elements`. This can improve reception quality because the *order* of gain stages matters for noise performance:

| Stage | Role | Tuning guidance |
| :--- | :--- | :--- |
| **LNA** (Low-Noise Amplifier) | First amplifier in the chain. Has the greatest impact on overall noise figure. | Set as high as possible without overloading from strong nearby signals. This is where sensitivity is won or lost. |
| **Mixer** | Frequency conversion stage. | Moderate gain. Too high increases intermodulation distortion (ghost signals from mixing products of strong stations). |
| **VGA** (Variable Gain Amplifier) | Final gain stage before the ADC. | Use to bring the overall signal level into the ADC's optimal range. Boosting here amplifies noise from earlier stages equally, so it contributes the least to sensitivity. |

The general principle is: **maximise gain early in the chain** (LNA) and **minimise gain late** (VGA), within the limits of what doesn't cause overload. This keeps the signal-to-noise ratio as high as possible through the receive chain.

**SNR threshold tuning**:

The `snr_threshold_db` setting controls how far above the noise floor a signal must be before it's detected. Each device card above lists a sensible starting value for that hardware. To adjust:

- If you're getting recordings that are mostly noise, raise the threshold by 1-2 dB at a time, *and* enable [`activation_variance_db`](#rejecting-emptynoise-recordings) if you haven't already — variance rejection catches the noise triggers that the SNR check can't distinguish.
- If you're missing transmissions you can hear on a handheld scanner, lower the threshold.
- The OFF threshold is always 3 dB below the ON threshold (hysteresis) to prevent rapid toggling.

**General tips**:
- Available gain element names and their valid ranges are logged at INFO level on startup. Check these before setting values.
- Optimal values depend on your antenna, band, and local RF environment — a rooftop antenna in a city needs different gain from a small whip in a rural area.
- Airband (AM, 118-137 MHz) typically needs less gain than PMR (NFM, 446 MHz) because aircraft transmitters are more powerful (5-25W) than PMR handhelds (0.5W).

## Rejecting empty/noise recordings

### The problem

SNR thresholds detect any signal that's louder than the noise floor — but they can't distinguish a *real* signal from a *noisy* one. With sensitive receivers like the AirSpy HF+ Discovery, you'll often see channels register 6-10 dB SNR yet contain only hissing static when played back. Raising `snr_threshold_db` doesn't help: the SNR is genuinely high, because the noise in that channel really is louder than the band-wide noise floor.

What's needed is a way to tell **noise** apart from **real signals** — and a single check isn't enough, because noise comes in different flavours that fool different detectors.

### The solution: three-layer noise rejection

The scanner applies three independent gates, each catching a different kind of false positive. All three are modulation-agnostic — they work for voice, data, tones, beacons, and any future modulation type.

#### Gate 1 — RF power variance (`activation_variance_db`)

Real signals fluctuate over time: syllables, frame structure, burst patterns all produce 5-15 dB power swings within a 200 ms detection window. Stationary noise produces near-constant power (standard deviation ~1-2 dB).

At the moment a channel turns ON, the scanner measures the standard deviation of the channel's power across the 8 Welch PSD segments. If the standard deviation falls below `activation_variance_db` (default 3.0 dB), the activation is suppressed — no ON event fires, no recording starts.

This is the cheapest check (~0.1 ms, reuses already-computed PSD data). It catches broadband stationary noise that happens to sit a few dB above the noise floor.

#### Gate 2 — Audio spectral flatness (`discard_empty_enabled`)

Some noise passes Gate 1 — for example, narrowband interference with enough temporal variance to look "active" in the RF domain, but no actual signal content when demodulated. Gate 2 catches this by speculatively demodulating the first IQ block and computing the **spectral flatness** (Wiener entropy) of the resulting audio.

Noise has a flat power spectrum (flatness 0.3-0.5). Any real signal — voice, data, tones — has a peaked spectrum (flatness < 0.04). The threshold of 0.15 sits in the large gap between the two groups, providing robust separation without per-modulation tuning.

If the flatness exceeds 0.15, the activation is suppressed — same as Gate 1. The speculative demodulation result is discarded; the main demodulation path runs fresh with proper trim boundaries if the check passes.

This check is more expensive (~10-20 ms, requires demodulation + FFT) so it only runs after Gate 1 passes. Controlled by `discard_empty_enabled` (default: true).

#### Gate 3 — Post-recording spectral flatness (`discard_empty_enabled`)

Gates 1 and 2 both operate at turn-ON time. Gate 3 operates at turn-OFF time, on the finished recording.

A signal can legitimately pass Gates 1 and 2 (the first block has real content) but produce a mostly-empty recording — for example, a brief 200 ms transmission followed by several seconds of hold-timer noise. The overall recording's spectral flatness will be high even though the first block was clean.

After the WAV file is closed, the scanner reads it back and computes spectral flatness on the full audio. If the flatness exceeds 0.15, the file is deleted before any recording-finished callbacks fire.

### How the gates differ

| Gate | Domain | When | What it catches | Cost |
| :--- | :--- | :--- | :--- | :--- |
| 1. Variance | RF PSD | Turn-ON | Broadband stationary noise | ~0.1 ms |
| 2. Flatness (preview) | Demodulated audio | Turn-ON | Narrowband noise that passes Gate 1 | ~10-20 ms |
| 3a. Min duration | Recording metadata | Turn-OFF | Brief transients (radar, ignition) that pass spectral checks | ~0 ms |
| 3b. Flatness (whole file) | Demodulated audio | Turn-OFF | Recordings that started real but became mostly noise | ~10-20 ms |

### Example

Imagine a "noisy" channel with average power 9 dB above the noise floor and a real voice transmission also at 9 dB SNR:

| Source | Avg SNR | Per-segment power (dB above floor) | Std dev | Audio flatness |
| :--- | :--- | :--- | :--- | :--- |
| Stationary noise | 9 dB | 9.1, 8.8, 9.0, 9.2, 8.9, 9.1, 8.7, 9.2 | **0.18 dB** | 0.38 |
| Voice transmission | 9 dB | 4.0, 12.5, 14.1, 7.0, 13.8, 11.2, 5.5, 3.9 | **4.3 dB** | 0.003 |

The noise is caught by Gate 1 (variance 0.18 < 3.0). If it somehow passed Gate 1, Gate 2 would catch it (flatness 0.38 > 0.15). The voice passes both cleanly.

### Configuration

```yaml
recording:
  discard_empty_enabled: true   # Gates 2 and 3 (default: true)

bands:
  air_civil_bristol_airspyhf:
    type: AIR
    freq_start: 125.5e+6
    freq_end: 126.0e+6
    sample_rate: 0.912e6
    snr_threshold_db: 6
    activation_variance_db: 3.0  # Gate 1 threshold (default: 3.0)
    sdr_gain_db: auto
```

### How it interacts with other settings

| Setting | Relationship |
| :--- | :--- |
| `snr_threshold_db` | Runs first. Channels below the SNR threshold never reach the noise gates. |
| `activation_variance_db` | Gate 1, only on turn-on transitions, only when the SNR check passed. |
| `discard_empty_enabled` | Gates 2 and 3b. Gate 2 runs after Gate 1 passes. Gate 3b runs on recording close. |
| `min_recording_seconds` | Gate 3a. Runs on recording close, before Gate 3b. Set to `0` to disable. |
| Hysteresis (built-in 3 dB margin) | Unchanged. Once a recording starts, it continues until SNR drops below `snr_threshold_db - 3`. |
| Hold time (`recording_hold_time_ms`) | Unchanged. Brief drops in SNR during active recording are tolerated. Gate 3b may discard if the hold timer extends the recording far beyond the actual signal. |

All three gates suppress silently — no ON callback fires, no recording file is kept. Downstream consumers (OSC bridge, user scripts) only see activations and recordings that passed all applicable gates.

### Tuning guidance

| Symptom | Action |
| :--- | :--- |
| Defaults work | Leave them — `activation_variance_db: 3.0` and `discard_empty_enabled: true` handle most cases |
| Real signals (voice, data) being rejected by Gate 1 | Lower `activation_variance_db`: try `2.0` or `2.5` |
| Noise still triggers recordings (passes Gate 1) | Gate 2 should catch it automatically; if not, raise `activation_variance_db` to `4.0` or `5.0` |
| Want to disable Gate 1 | Set `activation_variance_db: 0` |
| Want to disable Gates 2 and 3 | Set `discard_empty_enabled: false` |

### How to confirm it's working

Gate 1 suppression is logged at **DEBUG** level:
```
Channel 18 suppressed: power variance 0.4 dB below threshold 3.0 dB (likely noise)
```

Gate 2 suppression is logged at **DEBUG** level:
```
Channel 18 suppressed: audio is noise-only (spectral flatness 0.38)
```

Gate 3 discards are logged at **INFO** level:
```
Discarded empty recording: 2026-04-11_15-09-28_air_civil_bristol_airspyhf_59_6.0dB.wav
```

### Generality

All three gates are modulation-agnostic:

- Gate 1 operates on raw channel power from FFT bins — works for any signal type
- Gates 2 and 3 operate on spectral flatness of demodulated audio — any non-noise signal (voice, data, tones, beacons) produces a peaked spectrum that passes the check
- No demodulator-specific tuning is needed

## Dynamics Curve (Experimental)

An optional per-sample noise-reduction stage that runs during recording, after spectral subtraction and before the soft limiter. It applies a smooth nonlinear transfer curve in dBFS:

- **Below the threshold** (the "cut" region), quiet samples are progressively reduced — a downward expander that suppresses background noise. The curve is a smoothstep S-curve with zero slope at both endpoints, so there is no audible kink at the threshold or the floor. Samples below the floor are hard-zeroed.
- **Above the threshold** (the "boost" region), loud samples are gently boosted — an upward expander that gives voice presence. The curve is a sin² hump with zero boost at both endpoints (so 0 dBFS samples pass through unchanged).

Together the two regions widen the overall dynamic range. It works for any modulation type, has no envelope follower, and adds negligible CPU.

This is **off by default** and is intended for A/B comparison testing. To enable it on your installation:

```yaml
recording:
    dynamics_curve_enabled: true
    dynamics_curve:
        threshold_dbfs: -25.0   # Dividing line between cut and boost regions
        cut_db: 6.0             # Reduction at the midpoint of the cut S-curve (max = 2× at floor)
        boost_db: 1.5           # Peak boost in the boost hump
        floor_dbfs: -60.0       # Hard silence below this level
        cut_curve: 0.5          # 0..1; 0.5 = symmetric, <0.5 steeper near threshold
        boost_curve: 0.5        # 0..1; same skew control for the boost hump
```

The function operates per-sample (no envelope follower, no attack/release), so very aggressive parameter values can introduce mild harmonic distortion on signals near the threshold. The defaults are conservative enough that this is benign on voice; if you hear an "edge" on the loudest syllables, lower `cut_db` and `boost_db`. If a recording sounds completely silent, you have probably set `floor_dbfs` too high — try `-60` or lower.

The function clamps its output to the ±1.0 range as belt-and-braces speaker protection. If your configuration would otherwise drive the boost region above 0 dBFS, a warning is logged at startup so you can dial it back before listening.

## Parallel Scans (Multiple Devices)
Run one process per device:

```bash
substation --band air_civil_bristol --device-type rtlsdr --device-index 0
substation --band pmr --device-type rtlsdr --device-index 1
```

If you need stricter real-time behavior, you can pin each scan to a CPU core:

```bash
taskset -c 2 substation --band air_civil_bristol --device-index 0
taskset -c 3 substation --band pmr --device-index 1
```

## Resource and Performance Notes
- **Sample rate dominates CPU**. Large bands at high sample rates increase FFT/PSD load.
- **Overrun warnings** indicate the processing of a slice exceeded its real-time window. This can lead to dropped IQ blocks (`Sample queue full`).
- **Noise reduction** runs during write/flush if enabled (default). It uses `apply_spectral_subtraction` which is efficient and receives the band-wide noise floor for improved frame classification. The alternative `apply_noisereduce` implementation exists in `substation/dsp/noise_reduction.py` for reference but is not used by default as it is significantly more CPU-intensive.
- **Queue size** provides burst tolerance but uses RAM (each slice can be several MB).

If you see repeated `Sample queue full` warnings, reduce the band's `sample_rate`, exclude channels, or increase `sample_queue_maxsize`.

## Limitations
- Processing is slice-based; extremely wide bands or multiple high-rate scans can exceed real-time capacity on low-power CPUs.
- If you enable `apply_noisereduce` (requires code change), it is CPU-intensive for long chunks; on constrained devices, stick with the default `apply_spectral_subtraction` or reduce `disk_flush_interval_seconds`.

## Author
Written by Simon Holliday ([https://simonholliday.com/](https://simonholliday.com/))

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- **Copyleft**: Any modifications or improvements to this software must be shared back under the same license, even if used over a network.
- **Attribution**: You must give appropriate credit to the original author (Simon Holliday).
- **Commercial Use**: Permitted, provided you comply with the copyleft obligations of the AGPL-3.0.

See the [LICENSE](LICENSE) file for the full legal text.
