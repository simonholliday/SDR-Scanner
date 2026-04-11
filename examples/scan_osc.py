"""
Example: Radio scanning with OSC event forwarding.

This script runs Substation with an OscEventSender attached, so channel
state changes and saved recordings are emitted as OSC messages to a
downstream tool — typically the Subsequence MIDI sequencer listening on
UDP port 9000.  When sampler_host is also set, finalised recordings
additionally trigger a /sample/import message at the Subsample sampler
(UDP port 9002) so it can load the new WAV file directly without
watching the output directory.

Key concepts:
1. Loading the configuration and constructing a RadioScanner.
2. Constructing an OscEventSender and attaching it to the scanner.
3. Adding additional custom callbacks alongside the OSC forwarder.
4. Running the asynchronous scan loop.

Requires the optional OSC extra:  pip install -e ".[osc]"
"""

import asyncio
import logging
import sys

import substation.config
import substation.osc_sender
import substation.scanner

logger = logging.getLogger(__name__)

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)


def my_state_handler (band: str, channel_index: int, is_active: bool, snr_db: float) -> None:

	"""
	Custom callback that runs alongside the OSC forwarder.

	Demonstrates that add_state_callback can be called multiple times —
	the scanner dispatches to every registered callback, so your own
	logging / dashboards / webhooks keep working when OSC is attached.
	"""

	state_desc = "ACTIVE" if is_active else "INACTIVE"
	print(f">>> {band} channel {channel_index} is {state_desc} ({snr_db:.1f} dB SNR)")


async def run_scanner_with_osc () -> None:

	"""
	Initialise the scanner, attach an OSC event sender, and start scanning.
	"""

	config_path = './config.yaml'

	try:
		config_data = substation.config.load_config(config_path)

		scanner = substation.scanner.RadioScanner(
			config=config_data,
			band_name='pmr',        # Must match a band in your config.yaml
			device_type='rtlsdr',   # Or 'hackrf', 'airspy', 'airspyhf', etc.
			device_index=0,
		)

		# Forward scanner events to the sequencer on 9000, and also
		# notify the sampler on 9002 whenever a recording is saved.
		# Leave sampler_host=None to disable the sampler path.
		osc_sender = substation.osc_sender.OscEventSender(
			host='127.0.0.1',
			port=9000,
			sampler_host='127.0.0.1',
			sampler_port=9002,
		)
		osc_sender.attach(scanner)

		# You can still register your own callbacks alongside OSC.
		scanner.add_state_callback(my_state_handler)

		print(f"Starting scanner with OSC forwarding on band: {scanner.band_name}")
		print("Press Ctrl+C to stop.")

		await scanner.scan()

	except KeyboardInterrupt:
		print("\nStopping scanner...")

	except FileNotFoundError:
		print(f"Error: Configuration file not found at {config_path}")
		sys.exit(1)

	except Exception as exc:
		logger.error(f"Unexpected error: {exc}", exc_info=True)
		sys.exit(1)


if __name__ == "__main__":
	asyncio.run(run_scanner_with_osc())
