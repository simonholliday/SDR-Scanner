"""
Command-line interface for Substation.

Provides a simple command-line tool for running the scanner with arguments
for selecting bands, SDR devices, and listing available configuration.

Typical usage:
	python -m substation --band pmr                 # Scan PMR band
	python -m substation --list-bands               # Show available bands
	python -m substation --band airband --device-type hackrf  # Use HackRF
	python -m substation --band pmr --iq-file recording.wav --center-freq 446059313
"""

import argparse
import asyncio
import datetime
import logging
import pathlib
import sys
import typing

import substation
import substation.config
import substation.scanner

logger = logging.getLogger(__name__)

def list_bands (config_path: pathlib.Path | None) -> None:

	"""
	Display a summary of all bands defined in the configuration.

	Loads the configuration and prints a formatted table showing each band's
	name, frequency range, modulation type, and channel spacing. Useful for
	discovering what bands are available before starting a scan.

	Args:
		config_path: Optional path to user config override file

	Exits:
		Exits with code 1 if configuration cannot be loaded
	"""

	try:
		config_data = substation.config.load_config(config_path)
		bands = config_data.bands

		print("\nAvailable bands:")
		print("=" * 60)

		for band_name, band_config in bands.items():
			# Convert frequencies to MHz for readability
			freq_start = band_config.freq_start / 1e6
			freq_end = band_config.freq_end / 1e6
			modulation = band_config.modulation or 'Unknown'
			# Convert spacing to kHz for readability
			channel_spacing = band_config.channel_spacing / 1e3

			print(f"\n{band_name}:")
			print(f"  Frequency range: {freq_start:.3f} - {freq_end:.3f} MHz")
			print(f"  Modulation: {modulation}")
			print(f"  Channel spacing: {channel_spacing:.1f} kHz")

		print()

	except Exception as e:
		print(f"Error loading configuration: {e}", file=sys.stderr)
		sys.exit(1)

async def run_scanner (config_path: pathlib.Path | None, band_name: str, device_type: str, device_index: int) -> None:

	"""
	Initialize and run the scanner with a live SDR device.

	Args:
		config_path: Optional path to user config override file
		band_name: Name of the band to scan (must exist in config.bands)
		device_type: SDR device type ('rtlsdr' or 'hackrf')
		device_index: Device index for multi-device setups (0 for first device)

	Exits:
		Exits with code 1 if configuration is invalid or band doesn't exist
	"""

	try:
		config_data = substation.config.load_config(config_path)

		if not band_name:
			logger.error("No band specified. Use --band to select a band.")
			sys.exit(1)

		if band_name not in config_data.bands:
			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		scan = substation.scanner.RadioScanner(
			config_path=config_path,
			config=config_data,
			band_name=band_name,
			device_type=device_type,
			device_index=device_index
		)

		sv = None
		if config_data.supervisor.enabled:
			sv = _start_supervisor(scan, config_data.supervisor.port)
			if sv:
				await sv.start()

		try:
			await scan.scan ()
		finally:
			if sv:
				await sv.stop()

	except Exception as e:
		logger.error(f"Error running scanner: {e}", exc_info=True)
		sys.exit(1)


async def run_scanner_file (config_path: pathlib.Path | None, band_name: str, iq_file: str, center_freq: float, start_time: datetime.datetime) -> None:

	"""
	Process an IQ WAV file through the scanner pipeline.

	Streams the file at full speed (no real-time pacing) using a virtual
	clock that advances with sample position.  Output recordings use the
	virtual timestamps for directory and file naming.

	Args:
		config_path: Optional path to user config override file
		band_name: Name of the band to scan
		iq_file: Path to 2-channel IQ WAV file
		center_freq: Center frequency of the recording in Hz
		start_time: Start datetime for the recording (used for output timestamps)
	"""

	try:
		config_data = substation.config.load_config(config_path)

		if band_name not in config_data.bands:
			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		# Read sample rate from the WAV file to initialise the virtual clock
		import soundfile
		info = soundfile.info(iq_file)
		file_sample_rate = float(info.samplerate)

		clock = substation.scanner.VirtualClock(start_time, file_sample_rate)

		scan = substation.scanner.RadioScanner(
			config=config_data,
			band_name=band_name,
			device_type='file',
			clock=clock,
			device_kwargs={
				'file_path': iq_file,
				'center_freq': center_freq,
			},
		)

		logger.info(
			f"IQ file playback: {iq_file} — "
			f"center {center_freq/1e6:.6f} MHz, "
			f"start {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
		)

		sv = None
		if config_data.supervisor.enabled:
			sv = _start_supervisor(scan, config_data.supervisor.port)
			if sv:
				await sv.start()

		try:
			await scan.scan()
		finally:
			if sv:
				await sv.stop()

	except Exception as e:
		logger.error(f"Error processing IQ file: {e}", exc_info=True)
		sys.exit(1)


def _start_supervisor (scanner: typing.Any, port: int) -> typing.Any | None:

	"""Create a SubstationSupervisor if dependencies are installed, else warn."""

	try:
		import supervisor.app.substation as _sv_mod
		return _sv_mod.SubstationSupervisor(scanner, port=port)
	except ImportError as exc:
		logger.warning(
			f"Supervisor enabled but missing dependency ({exc}). "
			f"Install with: pip install -e \".[supervisor]\""
		)
		return None


def main () -> int:

	"""
	Main entry point for the command-line interface.

	Parses command-line arguments, sets up logging, and dispatches to either
	list_bands(), run_scanner(), or run_scanner_file() based on the arguments.

	Returns:
		Exit code: 0 for success, 1 for error
	"""

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

	parser = argparse.ArgumentParser(
		description='Substation - Software-defined radio band scanner',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  substation --band pmr                    # Scan PMR band with RTL-SDR
  substation --band marine --device-type hackrf  # Scan marine band with HackRF
  substation --list-bands                  # List all available bands
  substation --band pmr --iq-file rec.wav --center-freq 446059313  # File playback
		"""
	)

	# User config override file (merged on top of config.yaml.default)
	parser.add_argument(
		'--config', '-c',
		default=None,
		help='Path to user config override file (default: config.yaml in CWD if it exists)'
	)

	# Which band to scan (required for scanning, not for --list-bands)
	parser.add_argument(
		'--band', '-b',
		default=None,
		help='Band to scan (required unless using --list-bands)'
	)

	# Which SDR hardware to use
	parser.add_argument(
		'--device-type', '-t',
		default='rtlsdr',
		help='SDR device type: rtlsdr, hackrf, airspy, airspyhf, soapy:<driver> (default: rtlsdr)'
	)

	# Device index for systems with multiple SDRs
	parser.add_argument(
		'--device-index', '-i',
		type=int,
		default=0,
		help='SDR device index for multi-device setups (default: 0)'
	)

	# Utility flag to list available bands
	parser.add_argument(
		'--list-bands',
		action='store_true',
		help='List available bands and exit'
	)

	# IQ file playback
	parser.add_argument(
		'--iq-file',
		default=None,
		help='IQ WAV file to process (2-channel I/Q). Replaces live SDR device.'
	)

	parser.add_argument(
		'--center-freq',
		type=float,
		default=None,
		help='Center frequency of the IQ recording in Hz (required with --iq-file)'
	)

	parser.add_argument(
		'--start-time',
		default=None,
		help='Start time of the recording as "YYYY-MM-DD HH:MM:SS" (default: 2000-01-01 00:00:00)'
	)

	args = parser.parse_args()

	config_path = pathlib.Path(args.config) if args.config else None

	# Handle --list-bands mode (doesn't require --band)
	if args.list_bands:
		list_bands(config_path)
		return 0

	# Validate that --band is provided for scanning mode
	if not args.band:
		print("Error: --band is required unless using --list-bands.", file=sys.stderr)
		return 1

	# IQ file playback mode
	if args.iq_file:
		if args.center_freq is None:
			print("Error: --center-freq is required with --iq-file.", file=sys.stderr)
			return 1

		if args.start_time:
			try:
				start_dt = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
			except ValueError:
				print('Error: --start-time must be "YYYY-MM-DD HH:MM:SS".', file=sys.stderr)
				return 1
		else:
			start_dt = datetime.datetime(2000, 1, 1)

		try:
			asyncio.run(run_scanner_file(
				config_path=config_path,
				band_name=args.band,
				iq_file=args.iq_file,
				center_freq=args.center_freq,
				start_time=start_dt,
			))
			return 0
		except KeyboardInterrupt:
			return 0
		except Exception:
			return 1

	# Live SDR scanning mode
	try:
		asyncio.run(run_scanner(config_path=config_path, band_name=args.band, device_type=args.device_type, device_index=args.device_index))
		return 0

	except KeyboardInterrupt:
		return 0

	except Exception:
		return 1

if __name__ == '__main__':
	sys.exit(main())
