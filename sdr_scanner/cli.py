"""
Command-line interface for SDR Scanner
"""

import argparse
import asyncio
import logging
import sys

import sdr_scanner
import sdr_scanner.config
import sdr_scanner.scanner

logger = logging.getLogger(__name__)

def list_bands(config_path: str) -> None:

	"""
	List available bands from configuration

	Args:
		config_path: Path to configuration file
	"""

	try:

		config_data = sdr_scanner.config.load_config(config_path)
		bands = config_data.bands

		print(f"\nAvailable bands in {config_path}:")
		print("=" * 60)

		for band_name, band_config in bands.items():

			freq_start = band_config.freq_start / 1e6
			freq_end = band_config.freq_end / 1e6
			modulation = band_config.modulation or 'Unknown'
			channel_spacing = band_config.channel_spacing / 1e3

			print(f"\n{band_name}:")
			print(f"  Frequency range: {freq_start:.3f} - {freq_end:.3f} MHz")
			print(f"  Modulation: {modulation}")
			print(f"  Channel spacing: {channel_spacing:.1f} kHz")

		print()

	except Exception as e:

		print(f"Error loading configuration: {e}", file=sys.stderr)
		sys.exit(1)

async def run_scanner (config_path:str, band_name:str, device_type:str, device_index:int) -> None:

	"""
	Run the scanner with specified parameters

	Args:
		config_path: Path to configuration file
		band_name: Band to scan
		device_type: SDR device type
		device_index: SDR device index
	"""

	try:

		config_data = sdr_scanner.config.load_config(config_path)

		if not band_name:
			logger.error("No band specified. Use --band to select a band.")
			sys.exit(1)

		if band_name not in config_data.bands:

			available = ', '.join(config_data.bands.keys())
			logger.error(f"Band '{band_name}' not found. Available bands: {available}")
			sys.exit(1)

		scan = sdr_scanner.scanner.RadioScanner(
			config_path=config_path,
			config=config_data,
			band_name=band_name,
			device_type=device_type,
			device_index=device_index
		)

		await scan.scan()

	except KeyboardInterrupt:

		logger.info("Scan interrupted by user")

	except Exception as e:

		logger.error(f"Error running scanner: {e}", exc_info=True)
		sys.exit(1)

def main() -> int:

	"""
	Main entry point for CLI

	Returns:
		Exit code (0 for success, non-zero for error)
	"""

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

	parser = argparse.ArgumentParser(
		description='SDR Scanner - Software-defined radio band scanner',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  %(prog)s --band pmr                    # Scan PMR band with RTL-SDR
  %(prog)s --band marine --device-type hackrf  # Scan marine band with HackRF
  %(prog)s --list-bands                  # List all available bands
		"""
	)

	parser.add_argument(
		'--config', '-c',
		default='config.yaml',
		help='Path to configuration file (default: config.yaml)'
	)

	parser.add_argument(
		'--band', '-b',
		default=None,
		help='Band to scan (required unless using --list-bands)'
	)

	parser.add_argument(
		'--device-type', '-t',
		default='rtlsdr',
#		default='hackrf',
		help='SDR device type: rtlsdr, hackrf (default: rtlsdr)'
	)

	parser.add_argument(
		'--device-index', '-i',
		type=int,
		default=0,
		help='SDR device index for multi-device setups (default: 0)'
	)

	parser.add_argument(
		'--list-bands',
		action='store_true',
		help='List available bands and exit'
	)

	args = parser.parse_args()

	if args.list_bands:

		list_bands(args.config)
		return 0

	if not args.band:
		print("Error: --band is required unless using --list-bands.", file=sys.stderr)
		return 1

	try:

		asyncio.run(run_scanner(config_path=args.config, band_name=args.band, device_type=args.device_type, device_index=args.device_index))
		return 0

	except KeyboardInterrupt:
		return 0

	except Exception:
		return 1

if __name__ == '__main__':

	sys.exit(main())
