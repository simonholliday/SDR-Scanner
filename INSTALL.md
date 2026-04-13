# Substation Installation Guide

Platform-specific setup for SDR drivers, system dependencies, and the Python environment. See [README.md](README.md) for usage, configuration, and features.

Tested on:
- Debian 12 / Ubuntu 24.04 / Raspberry Pi OS (Bookworm)
- Fedora 43 Server (x86_64)

---

## 1. RTL-SDR Blog V4 Driver

The RTL-SDR Blog V4 requires the rtlsdrblog fork — the standard osmocom drivers are missing required symbols (e.g. `rtlsdr_set_dithering`).

### Debian / Ubuntu / Raspberry Pi OS

```bash
# Remove any existing RTL-SDR packages to avoid conflicts
sudo apt purge ^librtlsdr
sudo rm -rvf /usr/lib/librtlsdr* /usr/include/rtl-sdr* \
  /usr/local/lib/librtlsdr* /usr/local/include/rtl-sdr* \
  /usr/local/include/rtl_* /usr/local/bin/rtl_*

# Install build tools
sudo apt update
sudo apt install -y libusb-1.0-0-dev git cmake pkg-config build-essential

# Clone and build the RTL-SDR Blog fork
git clone https://github.com/rtlsdrblog/rtl-sdr-blog.git rtl-sdr
cd rtl-sdr
mkdir build && cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo make install
sudo cp ../rtl-sdr.rules /etc/udev/rules.d/
sudo ldconfig

# Blacklist the default DVB-T driver so it doesn't claim the SDR as a TV tuner
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee --append /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf

cd ../..
```

### Fedora

```bash
# Remove any existing RTL-SDR packages
sudo dnf remove 'rtl-sdr*' 'librtlsdr*'
sudo rm -rvf /usr/lib/librtlsdr* /usr/lib64/librtlsdr* \
  /usr/include/rtl-sdr* /usr/local/lib/librtlsdr* \
  /usr/local/lib64/librtlsdr* /usr/local/include/rtl-sdr* \
  /usr/local/include/rtl_* /usr/local/bin/rtl_*

# Install build tools
sudo dnf group install -y development-tools
sudo dnf install -y gcc gcc-c++ libusb1-devel git cmake pkgconf

# Clone and build the RTL-SDR Blog fork
git clone https://github.com/rtlsdrblog/rtl-sdr-blog.git rtl-sdr
cd rtl-sdr
mkdir build && cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo make install
sudo cp ../rtl-sdr.rules /etc/udev/rules.d/
sudo ldconfig

# Blacklist the default DVB-T driver
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee --append /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf

cd ../..
```

---

## 2. System Optimisation (USB Buffering)

High sample rates (e.g. HackRF at 20 MHz) require more USB buffer memory than the kernel default.

### Debian / Ubuntu / Raspberry Pi OS

```bash
# Edit the kernel command line
sudo nano /boot/firmware/cmdline.txt

# Add the following to the end of the existing (single) line:
usbcore.usbfs_memory_mb=1000

# Reboot for the change to take effect
sudo reboot
```

### Fedora

```bash
# Add the kernel parameter via grubby (Fedora's preferred method)
sudo grubby --update-kernel=ALL --args="usbcore.usbfs_memory_mb=1000"

# Reboot for the change to take effect
sudo reboot

# After reboot, verify the parameter is active
cat /proc/cmdline | grep usbfs_memory_mb
```

---

## 3. OS Dependencies

### Debian / Ubuntu / Raspberry Pi OS

```bash
# Audio and maths libraries
sudo apt install -y libsndfile1 libsndfile1-dev python3-setuptools python3-dev

# HackRF drivers and utilities
sudo apt install -y libhackrf-dev hackrf
```

### Fedora

```bash
# Audio and maths libraries
sudo dnf install -y libsndfile libsndfile-devel python3-setuptools python3-devel

# HackRF drivers and utilities
sudo dnf install -y hackrf hackrf-devel
```

---

## 4. SoapySDR + AirSpy Support

Required only if using AirSpy R2, AirSpy HF+ Discovery, or other SoapySDR-compatible devices.

### Debian / Ubuntu / Raspberry Pi OS

```bash
# Install SoapySDR core and Python bindings
sudo apt install -y soapysdr-tools python3-soapysdr

# Install device-specific SoapySDR modules (install only what you need)
sudo apt install -y soapysdr-module-airspy      # AirSpy R2
sudo apt install -y soapysdr-module-airspyhf    # AirSpy HF+ Discovery

# If soapysdr-module-airspyhf is not available in your distro's repos
# (e.g., Raspberry Pi OS), build from source instead:
sudo apt install -y libairspyhf-dev libsoapysdr-dev cmake
git clone https://github.com/pothosware/SoapyAirspyHF.git
cd SoapyAirspyHF
mkdir build && cd build
cmake ..
make
sudo make install
cd ../..

# Verify SoapySDR can see connected devices
SoapySDRUtil --find
```

### Fedora

```bash
# Install SoapySDR core and Python bindings
sudo dnf install -y SoapySDR SoapySDR-devel python3-SoapySDR

# Install device modules from the Fedora repos
sudo dnf install -y soapy-rtlsdr        # RTL-SDR via SoapySDR
sudo dnf install -y soapy-airspyhf      # AirSpy HF+ Discovery

# AirSpy R2 module is not in the Fedora repos — build from source:
sudo dnf install -y airspyone_host-devel SoapySDR-devel cmake
git clone https://github.com/pothosware/SoapyAirspy.git
cd SoapyAirspy
mkdir build && cd build
cmake ..
make
sudo make install
cd ../..

# Verify SoapySDR can see connected devices
SoapySDRUtil --find
```

---

## 5. Python Environment

The same for all platforms.

```bash
# If using SoapySDR devices (AirSpy, etc.), the venv MUST inherit system
# packages so it can access the system-installed python3-soapysdr bindings:
python3 -m venv --system-site-packages venv

# If you already have an existing venv, enable system packages on it:
# sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' /path/to/venv/pyvenv.cfg

# If only using RTL-SDR or HackRF (no SoapySDR), a standard venv is fine:
# python3 -m venv venv

source venv/bin/activate

# Install the package in editable mode
pip install -e .
```

---

## 6. Verification

```bash
# List available bands from the default configuration
substation --list-bands

# Or using the Python module directly
python3 -m substation --list-bands
```

---

## Platform-Specific Notes

### Fedora

- **SELinux**: Fedora enables SELinux in enforcing mode by default. If USB devices aren't accessible even after udev rules are in place, check for denials with `sudo ausearch -m avc -ts recent`. Typically the udev rules are sufficient.
- **Firewalld**: If using OSC event forwarding or other network features, you may need to open ports: `sudo firewall-cmd --add-port=9000/udp --permanent && sudo firewall-cmd --reload`.
- **lib vs lib64**: Fedora uses `/usr/lib64` for 64-bit libraries. The `ldconfig` step after building RTL-SDR should handle this, but if you get "library not found" errors, check that `/usr/local/lib64` is listed in `/etc/ld.so.conf.d/` and re-run `sudo ldconfig`.
