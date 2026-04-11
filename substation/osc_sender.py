"""
OSC event forwarding for Substation.

Bridges RadioScanner's existing channel-state and recording callbacks
onto OSC (Open Sound Control) messages, so downstream tools can react
to radio activity in real time.  Two OSC endpoints are supported:

- **Sequencer** (always): receives `/radio/state` and `/radio/recording`.
  Defaults to 127.0.0.1:9000, matching the Subsequence generative MIDI
  sequencer's OSC server.
- **Sampler** (optional): if `sampler_host` is provided, also receives
  `/sample/import` whenever a recording is finalised, so a sample-based
  instrument (e.g. Subsample on 127.0.0.1:9002) can load the new WAV
  without having to watch the output directory.

This module is **not** imported by any other part of Substation.  It
relies on python-osc, which is only installed when the `osc` optional
extra is present — keep the import here at module level so the
ImportError is immediate and obvious if a user forgets to install it.
Run `pip install -e ".[osc]"` to enable OSC support.

OSC address / argument reference (the two Substation outbound addresses):

    /radio/state      band_name:str  channel_index:int  is_active:int(0/1)  snr_db:float
    /radio/recording  band_name:str  channel_index:int  file_path:str
    /sample/import    file_path:str                  (only when sampler_host is set)
"""

import logging
import typing

import pythonosc.udp_client

if typing.TYPE_CHECKING:
	# Only needed for the type annotation on attach(); avoids a runtime
	# import cycle so this module stays independent of the scanner.
	import substation.scanner


logger = logging.getLogger(__name__)


class OscEventSender:

	"""
	Forwards Substation scanner events to an OSC receiver.

	Usage:

		osc = substation.osc_sender.OscEventSender(
			host='127.0.0.1', port=9000,
			sampler_host='127.0.0.1', sampler_port=9002,
		)
		osc.attach(scanner)

	After attach(), channel state changes and saved recordings will be
	emitted as OSC messages on the event loop thread.  The send calls are
	non-blocking (UDP sendto) and are wrapped in a narrow exception
	handler so a transient socket error cannot stall the scanner.
	"""

	def __init__ (
		self,
		host: str = '127.0.0.1',
		port: int = 9000,
		sampler_host: str | None = None,
		sampler_port: int = 9002,
	) -> None:

		"""
		Construct an OSC sender and open the UDP client(s).

		Args:
			host: Sequencer OSC receiver hostname or IP.  Defaults to
				localhost, which is correct for the common single-machine
				setup where Subsequence runs alongside Substation.
			port: Sequencer OSC receiver UDP port.  Defaults to 9000, the
				Subsequence default.
			sampler_host: Optional sampler OSC receiver hostname.  When
				set, recording-saved events also emit `/sample/import`
				to this address so the sampler can import the WAV
				directly.  Leave as None to skip the sampler path.
			sampler_port: Sampler OSC receiver UDP port.  Defaults to
				9002, the Subsample default.  Only used when
				sampler_host is not None.
		"""

		self._client = pythonosc.udp_client.SimpleUDPClient(host, port)

		self._sampler_client: pythonosc.udp_client.SimpleUDPClient | None
		if sampler_host is not None:
			self._sampler_client = pythonosc.udp_client.SimpleUDPClient(sampler_host, sampler_port)
			logger.info(
				f"OSC sender → sequencer {host}:{port}, sampler {sampler_host}:{sampler_port}"
			)
		else:
			self._sampler_client = None
			logger.info(f"OSC sender → sequencer {host}:{port} (no sampler)")

	def on_state_change (
		self,
		band_name: str,
		channel_index: int,
		is_active: bool,
		snr_db: float,
	) -> None:

		"""
		Scanner state-change callback: emit /radio/state to the sequencer.

		Matches the sync-callback signature documented on
		RadioScanner.add_state_callback().  Runs on the scanner's event
		loop thread via loop.call_soon_threadsafe(), so it must return
		quickly and never raise.
		"""

		# OSC has no native boolean — encode as 0 or 1.  Explicit ternary
		# instead of int(is_active) so the intent is obvious at a glance.
		active_int = 1 if is_active else 0

		try:
			self._client.send_message(
				'/radio/state',
				[band_name, int(channel_index), active_int, float(snr_db)],
			)

		except (OSError, ValueError, TypeError) as exc:
			# OSError covers UDP socket failures (host unreachable, EMFILE,
			# etc.).  ValueError / TypeError cover pythonosc's argument
			# encoding errors if an unexpected type ever slips through.
			# Anything else (AttributeError, KeyError, ...) is a programming
			# bug and is deliberately allowed to propagate so it surfaces
			# via the event loop's default exception handler.
			logger.warning(f"OSC /radio/state send failed: {exc}")

	def on_recording_saved (
		self,
		band_name: str,
		channel_index: int,
		file_path: str,
	) -> None:

		"""
		Scanner recording-saved callback: emit /radio/recording to the
		sequencer, and /sample/import to the sampler if one is configured.

		Matches the sync-callback signature documented on
		RadioScanner.add_recording_callback().  Runs on the scanner's
		event loop thread.
		"""

		path_str = str(file_path)

		try:
			self._client.send_message(
				'/radio/recording',
				[band_name, int(channel_index), path_str],
			)

		except (OSError, ValueError, TypeError) as exc:
			logger.warning(f"OSC /radio/recording send failed: {exc}")

		if self._sampler_client is not None:
			try:
				self._sampler_client.send_message('/sample/import', [path_str])

			except (OSError, ValueError, TypeError) as exc:
				logger.warning(f"OSC /sample/import send failed: {exc}")

	def attach (self, scanner: "substation.scanner.RadioScanner") -> None:

		"""
		Register this sender's methods as scanner callbacks.

		Equivalent to calling scanner.add_state_callback(self.on_state_change)
		and scanner.add_recording_callback(self.on_recording_saved), but
		presented as a single call so the consumer script stays tidy.
		"""

		scanner.add_state_callback(self.on_state_change)
		scanner.add_recording_callback(self.on_recording_saved)
