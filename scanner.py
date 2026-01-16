import asyncio
import fieldsequencer as fs
import logging
import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks, lfilter
from scipy.io import wavfile

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class SamplerData ():

	def __init__ (self, ref:str, samples:list):

		self.ref = ref
		self.samples = samples
		self.created_timestamp = time.time()

class Sampler:

	def __init__ (self, config, executor, event_emitter):

		self.executor = executor
		self.event_emitter = event_emitter
		self.loop = asyncio.get_running_loop()

		self.queue = asyncio.Queue()

		self.sdr = fs.sdr.Sdr(0) # Device Index 0..n
		self.band = None

		self.bands = {}

		self.load_config(config)

		self.event_emitter.on('load_config', self.load_config)


	def load_config (self, global_config):

		logger.info('Loading config')
		self.config = global_config.get('sampler')
		self.import_bands()


	async def init (self, calibration_frequency_hz=None):

		logger.info('Initializing')

		if None != calibration_frequency_hz:
			self.sdr.calibrate(calibration_frequency_hz, iterations=3)


	async def set_band (self, band):

		self.band = band

		# Switching band is CPU intensive so run in a thread
		result = await self.loop.run_in_executor (
			self.executor,
			self.sdr.set_up_for_band,
			band
		)

	def import_bands (self):

		for band_name in self.config.get('active_bands'):

			if band_name not in self.bands:

				band_config = self.config.get('bands').get(band_name)
				band = fs.dataclasses.Band(band_name, band_config, type=fs.dataclasses.Band.TYPE_DMR)
				self.bands[band_name] = band

	async def run (self):

		# TETRA has bandwidth 25kHz and runs on ranges 380 - 430 / 350 - 390 / 806 - 870 MHz

		# To allow the code below to work with the legacy band index iteration. Looping below needs rewriting.
		band_list = list(self.bands.values())

		band_list_length = len(band_list)
		band_index = 0
		last_band_index = 0

		await self.set_band(band_list[band_index])

		# Read in chunks of band_buffer_size_ms approx (exact size may be adjusted to make the maths work)
		async for sample_data in self.sample_band(band_buffer_size_ms=self.config.get('band_time_slice_ms'), band_buffer_cols=4096):

			band = self.bands[sample_data.ref]

			band.sample_buffer_index = (band.sample_buffer_index + 1) % band.sample_buffer_rows

#			logger.info('Queuing samples for band %s', band_name)
			await self.queue.put(sample_data)

			band_index = (band_index + 1) % band_list_length

			if band_index != last_band_index:

#				logger.info('Switching to band %d (%s)', band_index, band_list[band_index].get_range_as_string())
				await self.set_band(band_list[band_index])
				last_band_index = band_index


	async def sample_band (self, band_buffer_size_ms=50, band_buffer_cols=4096):

		device_buffer_cols = device_sample_size = self.config.get('sdr_device_sample_size')

		# Convert ms to samples
		band_buffer_size_samples = math.ceil(band_buffer_size_ms * self.sdr.sample_rate / 1e3)

		# band_buffer_size_samples must be a multiple of device_buffer_cols and band_buffer_cols
		adjusted_band_buffer_size_samples = fs.tools.first_multiple(band_buffer_size_samples, device_buffer_cols, band_buffer_cols)

		if adjusted_band_buffer_size_samples / band_buffer_size_samples > 1.25:
			logger.warning('Buffer size has been adjusted from %d samples to %d because this value must be a multiple of device_buffer_cols and band_buffer_cols. Tweak your numbers to avoid this.', band_buffer_size_samples, adjusted_band_buffer_size_samples)

		band_buffer_size_samples = adjusted_band_buffer_size_samples

		logger.debug('Band buffer duration: %0.2f ms', round(1000 * band_buffer_size_samples/self.sdr.sample_rate, 2))

		device_buffer_rows = band_buffer_size_samples // device_buffer_cols

		device_sample_buffer = np.zeros((device_buffer_rows, device_buffer_cols), dtype=np.complex64)
		device_buffer_index = 0

		# Read in samples until we have band_buffer_size_samples and yield back to parent
		# Samples are read in rows of device_sample_size wide, and reshaped to band_buffer_cols wide

		async for samples in self.sdr.stream(device_sample_size):

			device_sample_buffer[device_buffer_index] = samples

			device_buffer_index = (device_buffer_index + 1) % device_buffer_rows

			if device_buffer_index == 0:
				sample_data = SamplerData(self.band.name, np.reshape(device_sample_buffer, (band_buffer_size_samples // band_buffer_cols, band_buffer_cols)))
				yield sample_data


	async def process_sample_queue (self):

		"""
		Async function: fetch data from queue, offload heavy CPU processing
		to a thread (via run_in_executor), then emit processed result.
		"""

		while True:

			sample_data = await self.queue.get()

			logger.info('Got queued samples for band %s', sample_data.ref)

			band_name, channel_states, channel_mean_powers = await self.loop.run_in_executor (
				self.executor,
				self.process_band_samples,
				sample_data.ref,
				sample_data.samples
			)

			await self.event_emitter.emit_async('sampler_data', band_name, channel_states, channel_mean_powers)


	def process_band_samples (self, band_name, band_samples):

		band_name, channel_states, channel_mean_powers = self.scan_band_channels(band_name, band_samples)

		return band_name, channel_states, channel_mean_powers

	def get_psd_db (self, band_samples):

#		print('band samples shape:', band_samples.shape)
#		print('band samples dimensions:', len(band_samples.shape))

		if len(band_samples.shape) == 1:

			# If this is a 1d array, window it

			analysis_buffer_cols = len(band_samples)
			fft_window = fs.tools.get_window(analysis_buffer_cols)
			fft_sample = fft_window * band_samples

		else:

			# Otherwise, average windows

			# This size of band_samples may not be exactly what you requested in sample_band() [see that def for reasons]
			analysis_buffer_cols = band_samples.shape[1]

			fft_window = fs.tools.get_window(analysis_buffer_cols)

			# Multiply by an array of windows [1, analysis_buffer_cols] in shape

			fft_sample = band_samples * fft_window[None, :]

			fft_sample = fft_sample.mean(axis=0)

		fft_sample = np.fft.fft(fft_sample)

		psd_data = np.abs(fft_sample) ** 2

		psd_data_db = 10 * np.log10(psd_data + 1e-12) # Add a small value to avoid np.log10(0)

		psd_data_db_shifted = np.fft.fftshift(psd_data_db)

		freqs_shifted = self.sdr.center_freq + np.fft.fftshift(np.fft.fftfreq(analysis_buffer_cols, d=1.0/self.sdr.sample_rate))

		# Remove negative samples
#		psd_data_db_shifted[psd_data_db_shifted < 0] = 0

		return freqs_shifted, psd_data_db_shifted


	def scan_band_channels (self, band_name, band_samples):

		band = self.bands[band_name]

		# Reduce to fewer rows
#		band_samples = band_samples[0] # 1 row, less work, more noise
#		band_samples = band_samples.mean(axis=0) # All the rows, more work, less noise
		band_samples = np.reshape(band_samples[:10, :], -1) # 10 rows re-shaped to a 1d array, average

		freqs, psd_data_db = self.get_psd_db(band_samples)

		band_sample_size = len(band_samples)

		band_sum_mean_power_db = 0

		for frequency, channel in band.channels.items():

			# Set stats on the channel
			channel_analysis_samples = channel.analyze_in_band(band_samples)
			band_sum_mean_power_db += channel.stats.get_mean_power_db()

		# Analyze the band's null channel if present
		if band.null_channel is not None:
			band.null_channel.analyze_in_band(band_samples)

		channel.band.stats.set_mean_power_db(band_sum_mean_power_db / band.num_channels)

		channel_frequencies_hz = list(band.channels.keys())

		channel_states = []
		channel_mean_powers = []

		for frequency, channel in band.channels.items():

			channel_on, channel_snr, channel_mean_power = channel.is_on_snr()

			channel_current_on_state = True if channel_on > 0 else False

			if channel.last_known_on_state != channel_current_on_state:

				channel.last_known_on_state = channel_current_on_state

				channel_on_state_string = 'ON' if channel_current_on_state is True else 'OFF'

				logger.debug('Channel %d (%s) state changed to %s', channel.index, fs.tools.get_formatted_frequency_string(channel.frequency_hz), channel_on_state_string)

			if channel_current_on_state is True:

				"""
				# This won't work as we're using a tiny sample. Need to filter the entire channel data not just the little sample that was used to get channel_analysis_samples

				# Demodulate and save audio
				demodulated = np.angle(channel_analysis_samples[1:] * np.conj(channel_analysis_samples[:-1]))
				#demodulated = 0.5 * np.angle(samples[0:-1] * np.conj(samples[1:])) # Quadrature Demodulation

				current_sample_rate = band.sample_rate_hz // band.sample_rate_decimation_division

				bz, az = fs.tools.get_deemphasis_filter(current_sample_rate)
				audio_samples = lfilter(bz, az, channel_analysis_samples)

				# Normalize
				audio_samples = np.abs(audio_samples)
				max_val = np.max(audio_samples)
				if max_val > 0:  # Avoid division by zero
					audio_samples = (audio_samples / max_val * 32767).astype(np.int16)
				else:
					audio_samples = audio_samples.astype(np.int16)

				print('Writing audio')
				wavfile.write('./output.wav', int(current_sample_rate), audio_samples)

				"""

#				index = channel_frequencies_hz.index(channel.frequency_hz)

				channel_states.append(1)

			else:

				channel_states.append(0)

			channel_mean_powers.append(channel_mean_power)

		return band_name, channel_states, channel_mean_powers
