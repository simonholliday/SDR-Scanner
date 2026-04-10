"""
Antenna length calculator for Substation bands.

Calculates and prints the optimal length of common antennae (half-wave
dipole, quarter-wave vertical, 5/8-wave vertical, full-wave loop) for
either a configured Substation band or a manually-specified frequency.

Usage:
    python -m scripts.antenna --band hf_night_4mhz
    python -m scripts.antenna --freq 4625e3
    python -m scripts.antenna --list

Run with --help for the full argument list.

The math is the standard amateur-radio antenna formula:

    wavelength       = c / frequency
    half-wave dipole = velocity_factor * wavelength / 2
    quarter wave     = velocity_factor * wavelength / 4
    5/8 wave         = 0.625 * wavelength
    full-wave loop   = loop_velocity_factor * wavelength

The "velocity factor" is a small correction (~0.95) that accounts for
end-effect capacitance and the wave slowing down slightly in real wire
versus theoretical free-space propagation.  Loop antennas use a
slightly different correction (~0.97).  These values are universal
amateur-radio constants and are hard-coded — almost no user knows what
to put there, and the small variations between different wire gauges
are dwarfed by the practical tuning that follows construction.

Note that a configured Substation band is a frequency *range*, not a
single frequency.  At UHF, where the bands span well under 1% of the
centre frequency, one antenna covers the whole band easily.  On HF the
bands often span 10% or more, and no single antenna length is optimal
for the entire range.  When the band span exceeds BAND_SPREAD_WARN_FRACTION
the report appends a caveat showing the antenna's natural ~4% SWR
window and the dipole lengths at the band edges, so the user can
decide whether to cut for the centre, an edge, or use a tuner.
"""

import argparse
import sys

import substation.config


# Speed of light in metres per second.  The standard physical value;
# used directly in the wavelength calculation.
SPEED_OF_LIGHT_M_S = 299_792_458.0

# Velocity factor / wire correction for thin-wire dipoles and verticals.
# Accounts for end-effect capacitance and the small reduction in
# propagation velocity in real wire versus free space.  0.95 is the
# universal amateur-radio formula.
WIRE_VELOCITY_FACTOR = 0.95

# Velocity factor for full-wave loops.  Loops behave slightly
# differently from straight-wire antennas; 0.97 is the conventional
# value used in amateur-radio loop calculators.
LOOP_VELOCITY_FACTOR = 0.97

# A dipole's natural ±2% SWR window is adequate for transmit/receive
# without a tuner.  When a Substation band spans more than 4% of its
# centre frequency (i.e., the band is wider than the dipole's natural
# usable bandwidth), the report adds a caveat showing the dipole window
# and the band-edge antenna lengths.
BAND_SPREAD_WARN_FRACTION = 0.04


def compute_antenna_lengths (frequency_hz: float) -> dict:

	"""
	Compute optimal antenna lengths for a single frequency.

	Args:
		frequency_hz: Target frequency in hertz.  Must be positive.

	Returns:
		A dict containing the wavelength and the lengths of four common
		antenna types, all in metres:

			wavelength            — free-space wavelength c / f
			dipole_total          — tip-to-tip half-wave dipole
			dipole_leg            — one leg of the same dipole (half of dipole_total)
			quarter_wave_vertical — λ/4 vertical / monopole
			five_eighths_vertical — 5/8 λ vertical (no wire correction)
			full_wave_loop        — full-wave loop perimeter

	Raises:
		ValueError: If frequency_hz is zero or negative.
	"""

	if frequency_hz <= 0:
		raise ValueError(f"frequency_hz must be positive, got {frequency_hz}")

	wavelength = SPEED_OF_LIGHT_M_S / frequency_hz

	dipole_total = WIRE_VELOCITY_FACTOR * wavelength / 2.0
	dipole_leg = dipole_total / 2.0
	quarter_wave_vertical = WIRE_VELOCITY_FACTOR * wavelength / 4.0
	five_eighths_vertical = 0.625 * wavelength
	full_wave_loop = LOOP_VELOCITY_FACTOR * wavelength

	return {
		'wavelength': wavelength,
		'dipole_total': dipole_total,
		'dipole_leg': dipole_leg,
		'quarter_wave_vertical': quarter_wave_vertical,
		'five_eighths_vertical': five_eighths_vertical,
		'full_wave_loop': full_wave_loop,
	}


def _format_length (metres: float) -> str:

	"""
	Format a length in metres as either metres (>= 1 m, two decimal
	places) or centimetres (< 1 m, one decimal place).

	Used so VHF/UHF antennas read as "16.0 cm" instead of "0.16 m" while
	HF antennas read naturally as "30.79 m".
	"""

	if metres >= 1.0:
		return f"{metres:.2f} m"

	return f"{metres * 100.0:.1f} cm"


def format_antenna_report (
	frequency_hz: float,
	band_name: str | None = None,
	freq_start_hz: float | None = None,
	freq_end_hz: float | None = None,
) -> str:

	"""
	Format a human-readable antenna report.

	Args:
		frequency_hz: Centre frequency to compute against, in hertz.
		band_name: Optional band name; if supplied (along with start
			and end frequencies) the report shows a band header and may
			append a band-spread warning footer.
		freq_start_hz: Lower edge of the band, in hertz.  Required when
			band_name is set.
		freq_end_hz: Upper edge of the band, in hertz.  Required when
			band_name is set.

	Returns:
		The multi-line report as a string.
	"""

	lengths = compute_antenna_lengths(frequency_hz)

	lines = []
	lines.append("Antenna calculator — Substation")

	# Header — different shape for band vs single frequency
	if band_name is not None and freq_start_hz is not None and freq_end_hz is not None:
		span_hz = freq_end_hz - freq_start_hz
		span_pct = (span_hz / frequency_hz) * 100.0

		# Choose a sensible unit for the span: kHz for narrow HF bands,
		# MHz for wider bands and VHF/UHF.
		if span_hz < 1e6:
			span_str = f"{span_hz / 1e3:.0f} kHz"
		else:
			span_str = f"{span_hz / 1e6:.3f} MHz"

		lines.append(f"Band: {band_name}")
		lines.append(
			f"Frequency range: {freq_start_hz / 1e6:.3f} - {freq_end_hz / 1e6:.3f} MHz "
			f"(centre {frequency_hz / 1e6:.3f} MHz, span {span_str} / {span_pct:.2f}%)"
		)
	else:
		lines.append(f"Frequency: {frequency_hz / 1e6:.3f} MHz")

	# Wavelength: prefix with "at centre" for band reports so users know
	# the value depends on the band's centre rather than its edges; show
	# centimetres alongside metres for sub-metre VHF/UHF wavelengths.
	wl = lengths['wavelength']
	wl_label = "Wavelength at centre" if band_name else "Wavelength"
	if wl < 1.0:
		lines.append(f"{wl_label}: {wl:.2f} m / {wl * 100.0:.1f} cm")
	else:
		lines.append(f"{wl_label}: {wl:.2f} m")

	lines.append("")

	# Antenna table
	heading = "Antenna lengths (calculated for band centre):" if band_name else "Antenna lengths:"
	lines.append(heading)
	lines.append("")

	lines.append(
		f"  Half-wave dipole         total {_format_length(lengths['dipole_total'])}"
		f"   each leg {_format_length(lengths['dipole_leg'])}"
	)
	lines.append(f"  Quarter-wave vertical    {_format_length(lengths['quarter_wave_vertical'])}")
	lines.append(f"  5/8-wave vertical        {_format_length(lengths['five_eighths_vertical'])}")
	lines.append(f"  Full-wave loop           {_format_length(lengths['full_wave_loop'])}  (perimeter)")

	# Band-spread warning footer (only for band reports with span > threshold)
	if band_name is not None and freq_start_hz is not None and freq_end_hz is not None:
		span_fraction = (freq_end_hz - freq_start_hz) / frequency_hz

		if span_fraction > BAND_SPREAD_WARN_FRACTION:
			# Dipole's natural ~±2% usable window
			window_low_hz = frequency_hz * 0.98
			window_high_hz = frequency_hz * 1.02

			# Dipole lengths at the actual band edges
			edge_low = compute_antenna_lengths(freq_start_hz)
			edge_high = compute_antenna_lengths(freq_end_hz)

			lines.append("")
			lines.append(
				f"NOTE: This band spans {span_fraction * 100.0:.1f}% of its centre frequency.  A single dipole"
			)
			lines.append("optimised for the centre will work acceptably across roughly:")
			lines.append(
				f"  centre ± ~2%  →  {window_low_hz / 1e6:.3f} - {window_high_hz / 1e6:.3f} MHz"
				f"  (~4% useful bandwidth)"
			)
			lines.append("The full band is wider than this — if you want both the band edges and")
			lines.append("the middle, consider:")
			lines.append("  - cutting the dipole for the lower edge (longer = covers low freqs better)")
			lines.append("  - using a tuner / matching network")
			lines.append("  - using a wideband antenna (random wire, magnetic loop, discone)")
			lines.append("")
			lines.append("Lengths at the band edges for comparison:")
			lines.append(
				f"  {freq_start_hz / 1e6:.3f} MHz: dipole {_format_length(edge_low['dipole_total'])} total"
			)
			lines.append(
				f"  {freq_end_hz / 1e6:.3f} MHz: dipole {_format_length(edge_high['dipole_total'])} total"
			)

	return "\n".join(lines)


def main () -> int:

	"""
	Command-line entry point.

	Parses arguments, optionally loads the Substation config, and
	prints either an antenna report or a list of bands.  Returns the
	process exit code.
	"""

	parser = argparse.ArgumentParser(
		prog='scripts.antenna',
		description='Calculate optimal antenna lengths for a Substation band or frequency.',
	)

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('--band', metavar='NAME', help='Configured band name from config.yaml')
	group.add_argument('--freq', type=float, help='Frequency in Hz (e.g. 4625e3 or 446.0e6)')
	group.add_argument('--list', action='store_true', help='List all configured bands')

	parser.add_argument(
		'--config',
		default='config.yaml',
		help='Path to config.yaml (default: ./config.yaml)',
	)

	args = parser.parse_args()

	# --list mode
	if args.list:

		try:
			cfg = substation.config.load_config(args.config)
		except FileNotFoundError:
			print(f"error: config file not found: {args.config}", file=sys.stderr)
			return 2

		for name in sorted(cfg.bands):
			print(name)

		return 0

	# --freq mode (no config needed)
	if args.freq is not None:

		if args.freq <= 0:
			print("error: --freq must be positive", file=sys.stderr)
			return 2

		print(format_antenna_report(args.freq))
		return 0

	# --band mode (config required)
	try:
		cfg = substation.config.load_config(args.config)
	except FileNotFoundError:
		print(f"error: config file not found: {args.config}", file=sys.stderr)
		return 2

	if args.band not in cfg.bands:
		print(f"error: band {args.band!r} not found in {args.config}", file=sys.stderr)
		print("available bands: " + ", ".join(sorted(cfg.bands)), file=sys.stderr)
		return 2

	band = cfg.bands[args.band]
	centre = (band.freq_start + band.freq_end) / 2.0

	print(format_antenna_report(
		centre,
		band_name=args.band,
		freq_start_hz=band.freq_start,
		freq_end_hz=band.freq_end,
	))

	return 0


if __name__ == '__main__':
	sys.exit(main())
