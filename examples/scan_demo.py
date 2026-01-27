"""
Example: Using SDR Scanner as a Python Module

This script demonstrates how to integrate the SDR Scanner directly into your
own Python projects. This is useful if you want to build custom automation,
dashboards, or integration with other systems (like Home Assistant or MQTT).

Key concepts:
1. Loading the configuration.
2. Initializing the RadioScanner object.
3. Adding state change callbacks (events).
4. Running the asynchronous scan loop.
"""

import asyncio
import logging
import sys

# Import the sdr_scanner package components
import sdr_scanner.config
import sdr_scanner.scanner

# Professional logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def my_channel_event_handler (band: str, channel_index: int, is_active: bool, snr_db: float) -> None:
    """
    This callback is triggered whenever a channel turns ON (signal detected)
    or OFF (signal lost).

    Args:
        band: The name of the band being scanned
        channel_index: The numerical index of the channel
        is_active: True if the signal is now present, False if lost
        snr_db: The measured Signal-to-Noise ratio
    """
    state_desc = "ACTIVE" if is_active else "INACTIVE"
    print(f"\n>>> Event: Band [{band}] Channel {channel_index} is now {state_desc} ({snr_db:.1f} dB SNR)")

async def run_custom_scanner ():
    """
    Initialize and run the scanner module.
    """
    try:
        # 1. Load configuration from a YAML file
        # This contains your hardware settings and band definitions
        config_path = 'config.yaml'
        config_data = sdr_scanner.config.load_config(config_path)

        # 2. Instantiate the RadioScanner
        # You can specify the band, device type, and device index here
        scanner = sdr_scanner.scanner.RadioScanner(
            config=config_data,
            band_name='pmr',      # Must match a band in your config.yaml
            device_type='rtlsdr', # 'rtlsdr' or 'hackrf'
            device_index=0
        )

        # 3. Register a callback function
        # High-performance: this runs without blocking the radio processing
        scanner.add_state_callback(my_channel_event_handler)

        print(f"Starting custom scanner on band: {scanner.band_name}...")
        print("Press Ctrl+C to stop.")

        # 4. Start the asynchronous scan loop
        # This will run until the program is interrupted
        await scanner.scan()

    except KeyboardInterrupt:
        print("\nStopping scanner...")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # The scanner uses asyncio for high-performance non-blocking I/O
    asyncio.run(run_custom_scanner())
