SDR Scanner
===========

Overview
--------
SDR Scanner scans configured RF bands, detects active channels by SNR, demodulates audio (AM/NFM), and records mono WAV files with Broadcast WAV metadata. It is designed for continuous monitoring on modest hardware (e.g., Raspberry Pi), and supports multiple devices in parallel (RTL-SDR and HackRF).

Key Features
------------
- Band scanning with SNR-based activity detection.
- AM and NFM demodulation with stateful DSP.
- Per-channel recording to WAV/BWF.
- Optional noise reduction and soft limiting.
- Multi-device parallel scanning.
- Per-band exclusions (skip unwanted channel indices).

Quick Start
-----------
1) Install dependencies (see `installation.txt` for SDR drivers).
2) Configure bands in `config.yaml`.
3) Run:

```bash
python -m sdr_scanner --band=air_civil_bristol --device-type=rtlsdr --device-index=0
```

Audio files are written to:
```
./audio/YYYY-MM-DD/<band>/<timestamp>_<band>_<channel>_<device>.wav
```

Command Line
------------
```
python -m sdr_scanner --band <band> [--config <path>] [--device-type rtlsdr|hackrf] [--device-index N]
python -m sdr_scanner --list-bands
```

Options:
- `--config`, `-c`: path to config file (default `config.yaml`).
- `--band`, `-b`: band name to scan (required unless `--list-bands`).
- `--device-type`, `-t`: `rtlsdr` or `hackrf` (default `rtlsdr`).
- `--device-index`, `-i`: device index (default `0`).
- `--list-bands`: list available bands and exit.

Configuration
-------------
Config file is YAML. The top-level keys are `scanner`, `recording`, `band_defaults`, and `bands`.

Scanner
```
scanner:
  sdr_device_sample_size: 131072
  band_time_slice_ms: 200
  sample_queue_maxsize: 30
  calibration_frequency_hz: 93.7e+6
```
- `sdr_device_sample_size`: number of IQ samples per SDR callback. Higher values reduce callback overhead but increase latency.
- `band_time_slice_ms`: time slice used for PSD/SNR detection. Must be a multiple of `sdr_device_sample_size` (rounded up internally).
- `sample_queue_maxsize`: async queue depth. 10–50 is typical; higher tolerates bursts but uses more RAM.
- `calibration_frequency_hz`: optional known signal for PPM correction; set to `null` to disable.

Recording
```
recording:
  enabled: true
  buffer_size_seconds: 30
  disk_flush_interval_seconds: 5
  audio_sample_rate: 16000
  audio_output_dir: "./audio"
  fade_in_ms: 3
  fade_out_ms: 3
  soft_limit_drive: 2.0
```
- `buffer_size_seconds`: max in-memory audio per channel before drops.
- `disk_flush_interval_seconds`: how often to flush to disk.
- `audio_sample_rate`: output WAV rate (Hz).
- `fade_in_ms`/`fade_out_ms`: fades applied at channel start/stop.
- `soft_limit_drive`: post-processing soft limiter drive. Typical range 1.5–3.0 (higher = stronger limiting).

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
- `snr_threshold_db`: detection threshold (dB above noise floor).
- `sdr_gain_db`: numeric or `auto`.
- `exclude_channel_indices`: 0-based indices to skip (no analysis, no recording).

Parallel Scans (Multiple Devices)
---------------------------------
Run one process per device:

```bash
python -m sdr_scanner --band=air_civil_bristol --device-type=rtlsdr --device-index=0
python -m sdr_scanner --band=pmr --device-type=rtlsdr --device-index=1
```

If you need stricter real-time behavior, you can pin each scan to a CPU core:

```bash
taskset -c 2 python -m sdr_scanner --band=air_civil_bristol --device-index=0
taskset -c 3 python -m sdr_scanner --band=pmr --device-index=1
```

Resource and Performance Notes
------------------------------
- **Sample rate dominates CPU**. Large bands at high sample rates increase FFT/PSD load.
- **Overrun warnings** indicate the processing of a slice exceeded its real-time window. This can lead to dropped IQ blocks (`Sample queue full`).
- **Noise reduction** runs during write/flush; long chunks can still cause buffer overflow if disk/CPU is slow.
- **Queue size** provides burst tolerance but uses RAM (each slice can be several MB).

If you see repeated `Sample queue full` warnings, reduce the band’s `sample_rate`, exclude channels, or increase `sample_queue_maxsize`.

Limitations
-----------
- Processing is slice-based; extremely wide bands or multiple high-rate scans can exceed real-time capacity on low-power CPUs.
- `noisereduce` is CPU-intensive for long chunks; on very small devices, consider reducing `disk_flush_interval_seconds` or lowering `n_fft` in the noise reduction function if needed.

License
-------
See project license or repository metadata.
