"""
Command-line interface for SDR Scanner.

Provides a simple command-line tool for running the scanner with arguments
for selecting bands, SDR devices, and listing available configuration.

Typical usage:
	python -m sdr_scanner --band pmr                 # Scan PMR band
	python -m sdr_scanner --list-bands               # Show available bands
	python -m sdr_scanner --band airband --device-type hackrf  # Use HackRF
"""

import argparse
import asyncio
import logging
import sys

import sdr_scanner
import sdr_scanner.config
import sdr_scanner.scanner

logger = logging.getLogger(__name__)

def list_bands (config_path: str) -> None:

	"""
	Display a summary of all bands defined in the configuration file.

	Loads the configuration and prints a formatted table showing each band's
	name, frequency range, modulation type, and channel spacing. Useful for
	discovering what bands are available before starting a scan.

	Args:
		config_path: Path to YAML configuration file

	Exits:
		Exits with code 1 if configuration cannot be loaded
	"""

	try:
		config_data = sdr_scanner.config.load_config(config_path)
		bands = config_data.bands

		print(f"\nAvailable bands in {config_path}:")
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

async def run_scanner (config_path: str, band_name: str, device_type: str, device_index: int) -> None:

	"""
	Initialize and run the SDR scanner.

	This is the main async entry point that:
	1. Loads configuration from file
	2. Validates that the requested band exists
	3. Creates a RadioScanner instance
	4. Runs the scan loop until interrupted (Ctrl+C) or error

	Args:
		config_path: Path to YAML configuration file
		band_name: Name of the band to scan (must exist in config.bands)
		device_type: SDR device type ('rtlsdr' or 'hackrf')
		device_index: Device index for multi-device setups (0 for first device)

	Exits:
		Exits with code 1 if configuration is invalid or band doesn't exist
	"""

	try:
		# Load and validate configuration
		config_data = sdr_scanner.config.load_config(config_path)

		if not band_name:
			logger.error("No band specified. Use --band to select a band.")
			sys.exit(1)

		# Verify band exists in configuration
		if band_name not in config_data.bands:
			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		# Create scanner instance
		scan = sdr_scanner.scanner.RadioScanner(
			config_path=config_path,
			config=config_data,
			band_name=band_name,
			device_type=device_type,
			device_index=device_index
		)

		# Run the scan loop (blocks until interrupted)
		await scan.scan ()

	except Exception as e:
		# Log unexpected errors with full traceback
		logger.error(f"Error running scanner: {e}", exc_info=True)
		sys.exit(1)

def main () -> int:

	"""
	Main entry point for the command-line interface.

	Parses command-line arguments, sets up logging, and dispatches to either
	list_bands() or run_scanner() based on the arguments.

	The logging format includes timestamps and log levels for easier troubleshooting.
	Default level is INFO, which shows scanner activity without excessive detail.

	Returns:
		Exit code: 0 for success, 1 for error
	"""

	# Configure logging with timestamps for all output
	# Level INFO shows scanning activity without excessive debug detail
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

	# Set up argument parser with help text and examples
	parser = argparse.ArgumentParser(
		description='SDR Scanner - Software-defined radio band scanner',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  sdr-scanner --band pmr                    # Scan PMR band with RTL-SDR
  sdr-scanner --band marine --device-type hackrf  # Scan marine band with HackRF
  sdr-scanner --list-bands                  # List all available bands
		"""
	)

	# Configuration file location
	parser.add_argument(
		'--config', '-c',
		default='config.yaml',
		help='Path to configuration file (default: config.yaml)'
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
		help='SDR device type: rtlsdr, hackrf (default: rtlsdr)'
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

	args = parser.parse_args()

	# Handle --list-bands mode (doesn't require --band)
	if args.list_bands:
		list_bands(args.config)
		return 0

	# Validate that --band is provided for scanning mode
	if not args.band:
		print("Error: --band is required unless using --list-bands.", file=sys.stderr)
		return 1

	# Run the scanner asynchronously
	try:
		asyncio.run(run_scanner(config_path=args.config, band_name=args.band, device_type=args.device_type, device_index=args.device_index))
		return 0

	except KeyboardInterrupt:
		# Ctrl+C is a normal way to exit, not an error
		return 0

	except Exception:
		# Errors are already logged by run_scanner()
		return 1

if __name__ == '__main__':
	sys.exit(main())
