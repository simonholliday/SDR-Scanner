"""
Entry point for running SDR Scanner as a module.

This file allows the package to be executed as:
    python -m sdr_scanner [args]

It simply delegates to the CLI main() function which handles
all argument parsing and program execution.
"""

import sys
import sdr_scanner.cli

if __name__ == '__main__':
	sys.exit(sdr_scanner.cli.main())
