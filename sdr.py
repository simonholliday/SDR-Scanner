import logging

import fieldsequencer as fs

logger = logging.getLogger(__name__)

# Wrapper for rtlsdr
# In future this file should handle multiple devices

class Sdr (fs.rtlsdr.RtlSdr):
	pass
