# SDR Scanner

## Overview
SDR Scanner is a high-performance tool for monitoring and recording radio activity using Software Defined Radio (SDR) hardware. It is designed to be used in two ways:

1.  **As a Command-Line Tool**: Quickly scan and record bands using simple terminal commands.
2.  **As a Python Module**: Integrate radio scanning, detection, and callbacks directly into your own Python applications.

By connecting a supported USB receiver (like an RTL-SDR or HackRF), you can scan wide ranges of the radio spectrum - such as Airband or Maritime frequencies - and automatically record transmissions as they occur. The software handles the technical signal processing and hardware management in the background, allowing for efficient 24/7 monitoring even on modest hardware like a Raspberry Pi.

## Hardware Requirements
To use this software, a compatible Software Defined Radio (SDR) USB device is required. The code has been specifically tested and verified with the following hardware:

*   **[RTL-SDR Blog V4 and V3](https://www.rtl-sdr.com/about-rtl-sdr/)**: High-quality, low-cost receivers.
*   **[HackRF One](https://greatscottgadgets.com/hackrf/one/)**: A wideband transceiver capable of monitoring much larger frequency spans.

## Key Features & Optimizations
- **Advanced Signal Detection**: Uses Welch's Power Spectral Density (PSD) estimation for stable, low-variance activity detection.
- **Parallel Multi-Channel Recording**: Simultaneously detects and records all active channels in a band - unlike traditional handheld scanners which only play one channel at a time.
- **High-Fidelity Demodulation**: Implements stateful AM and NFM demodulation with continuous phase tracking and DC-blocking, eliminating pops and discontinuities between audio blocks.
- **Hardware Efficiency**:
    - **Vectorized Math**: Heavy processing is delegated to NumPy and SciPy for maximum throughput.
    - **Zero-Copy Architecture**: Uses memory stride tricks for overlapping FFT segments, avoiding expensive data copying.
    - **Lazy Evaluation**: Computationally expensive segment analysis is performed only when transitions are detected, drastically reducing idle CPU load.
- **State-of-the-Art Processing**:
    - **Vectorized AGC**: High-quality, smooth automatic gain control for AM with independent attack and release timings.
    - **Adaptive Noise Reduction**: Custom spectral subtraction provides significant hiss reduction with minimal overhead compared to standard libraries.
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
sdr-scanner --band air_civil_bristol --device-type rtlsdr --device-index 0
```

Audio files are written to:
```
./audio/YYYY-MM-DD/<band>/<timestamp>_<band>_<channel>_<snr>dB_<device>_<index>.wav
```

## Command Line
```bash
sdr-scanner --band <band> [--config <path>] [--device-type rtlsdr|hackrf] [--device-index N]
sdr-scanner --list-bands
```

## Python Module Usage
You can also use the scanner as a library in your own code. This allows you to respond to radio events programmatically.

```python
import asyncio

import sdr_scanner.config
import sdr_scanner.scanner

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
	config_data = sdr_scanner.config.load_config ("config.yaml")

	# Initialize scanner instance
	scanner = sdr_scanner.scanner.RadioScanner (
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

Options:
- `--config`, `-c`: path to config file (default `config.yaml`).
- `--band`, `-b`: band name to scan (required unless `--list-bands`).
- `--device-type`, `-t`: `rtlsdr` or `hackrf` (default `rtlsdr`).
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
  fade_in_ms: 3
  fade_out_ms: 5
  soft_limit_drive: 2.0
```
- `buffer_size_seconds`: max in-memory audio per channel before drops.
- `disk_flush_interval_seconds`: how often to flush to disk.
- `audio_sample_rate`: output WAV rate (Hz).
- `fade_in_ms`/`fade_out_ms`: fades applied at channel start/stop.
- `soft_limit_drive`: post-processing soft limiter drive. Typical range 1.5-3.0 (higher = stronger limiting).
- `noise_reduction_enabled`: toggle spectral subtraction noise reduction (default: true).
- `recording_hold_time_ms`: duration in ms to continue recording after signal drops below threshold (default: 500).

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
- `modulation`: `AM` or `NFM`.
- `recording_enabled`: enable recording for this band. Optional, defaults to `false` (can also be set in `band_defaults`).
- `snr_threshold_db`: detection threshold (dB above noise floor).
- `sdr_gain_db`: numeric or `auto`.
- `exclude_channel_indices`: 0-based indices to skip (no analysis, no recording).

## Broadcast WAV (BWF) & Metadata
Each recording captures industry-standard **Broadcast WAV (BWF)** metadata (EBU Tech 3285). This embeds technical details directly into the audio file, making it ideal for archival and automated post-processing.

**Compatibility**: These are standard `.wav` files. They will play perfectly in any normal audio player (VLC, Windows Media Player, Audacity, mobile devices, etc.).

### Metadata Example
If you open a recording in a professional audio tool or a BWF viewer, you will see fields like these:

| Field | Example Value | Description |
| :--- | :--- | :--- |
| **Description** | `{"band":"pmr","channel_index":0,"channel_freq":446006250.0}` | Machine-readable JSON with channel details |
| **Coding History** | `A=PCM,F=16000,W=16,M=mono,T=NFM;Frequency=446.00625MHz` | Technical signal chain (Algorithm, Rate, Modulation) |
| **Originator** | `SDR Scanner` | The software that created the file |
| **Origination Date** | `2026-01-27` | Date the recording started |
| **Time Reference** | `1152000` | Sample count since midnight (for precise timing) |


## Parallel Scans (Multiple Devices)
Run one process per device:

```bash
sdr-scanner --band air_civil_bristol --device-type rtlsdr --device-index 0
sdr-scanner --band pmr --device-type rtlsdr --device-index 1
```

If you need stricter real-time behavior, you can pin each scan to a CPU core:

```bash
taskset -c 2 sdr-scanner --band air_civil_bristol --device-index 0
taskset -c 3 sdr-scanner --band pmr --device-index 1
```

## Resource and Performance Notes
- **Sample rate dominates CPU**. Large bands at high sample rates increase FFT/PSD load.
- **Overrun warnings** indicate the processing of a slice exceeded its real-time window. This can lead to dropped IQ blocks (`Sample queue full`).
- **Noise reduction** runs during write/flush if enabled (default). It uses `apply_spectral_subtraction` which is efficient. The alternative `apply_noisereduce` implementation exists in `sdr_scanner/dsp/noise_reduction.py` for reference but is not used by default as it is significantly more CPU-intensive.
- **Queue size** provides burst tolerance but uses RAM (each slice can be several MB).

If you see repeated `Sample queue full` warnings, reduce the band's `sample_rate`, exclude channels, or increase `sample_queue_maxsize`.

## Limitations
- Processing is slice-based; extremely wide bands or multiple high-rate scans can exceed real-time capacity on low-power CPUs.
- If you enable `apply_noisereduce` (requires code change), it is CPU-intensive for long chunks; on constrained devices, stick with the default `apply_spectral_subtraction` or reduce `disk_flush_interval_seconds`.

## Author
Written by Simon Holliday ([https://simonholliday.com/](https://simonholliday.com/))

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- **Non-Commercial**: You may use, adapt, and build upon this work for any non-commercial purpose.
- **Attribution**: You must give appropriate credit to the original author (Simon Holliday).
- **Commercial Use**: For commercial licensing inquiries, please contact the author.

See the [LICENSE](LICENSE) file for the full legal text.
