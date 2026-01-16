import fieldsequencer as fs
import logging
import numpy as np

from rtlsdr import RtlSdr

logger = logging.getLogger(__name__)

class RtlSdr (RtlSdr):

	def calibrate (self, known_frequency_hz=None, bandwidth_hz=300e3, iterations=10):

		if known_frequency_hz is None:
			raise Exception('Frequency must be specified')

		initial_center_freq = self.center_freq
		initial_sample_rate = self.sample_rate

		self.set_center_frequency(known_frequency_hz)
		self.set_sample_rate(bandwidth_hz)

		freq_correction_ppm_list = []

		frequency_display_value, frequency_display_unit = fs.tools.format_frequency(known_frequency_hz)
		bandwidth_display_value, bandwidth_display_unit = fs.tools.format_frequency(bandwidth_hz)

		print('Calibrating SDR to', frequency_display_value, frequency_display_unit, 'within a', bandwidth_display_value, bandwidth_display_unit, 'band ...', end='', flush=True)

		sample_size = 256 * 1024

		for i in range (iterations, 0, -1):

			print('', i, end='', flush=True)

			samples = self.read_samples(sample_size)
			fft_result = np.fft.fftshift(np.fft.fft(samples * fs.tools.get_window(sample_size)))
			freqs = np.fft.fftshift(np.fft.fftfreq(sample_size, 1 / self.sample_rate))
			magnitude = 20 * np.log10(np.abs(fft_result))
			
			peak_index = np.argmax(magnitude)
			measured_frequency_hz = self.center_freq + freqs[peak_index]
			
			freq_correction_ppm_list.append((measured_frequency_hz - known_frequency_hz) / self.center_freq * 1e6)

		freq_correction_ppm = round(np.mean(freq_correction_ppm_list))

		print(' Done :)')

		self.set_center_frequency(initial_center_freq)
		self.set_sample_rate(initial_sample_rate)

		if 0 != freq_correction_ppm:

			self.set_frequency_correction(freq_correction_ppm)
			logger.info('Calibrated with offset of %d PPM', freq_correction_ppm)

		else:
			logger.info('No calibration needed')


	def set_up_for_band (self, band):

		self.set_center_frequency(band.get_center_frequency())
		self.set_sample_rate(band.get_sample_rate())


	def set_center_frequency (self, frequency_hz):

		self.center_freq = frequency_hz

		return self

	def set_sample_rate (self, sample_rate):

		self.sample_rate = sample_rate

		return self

	def set_frequency_correction (self, ppm=None):

		self.freq_correction = ppm

		return self
