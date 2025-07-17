import radarsimpy
import numpy as np
from radarsimpy import Radar, Transmitter, Receiver

print("`RadarSimPy` used in this example is version: " + str(radarsimpy.__version__))

antenna_gain = 12

az_angle = np.arange(-80, 81, 1)
az_pattern = 20 * np.log10(np.cos(az_angle / 180 * np.pi) ** 4) + antenna_gain

el_angle = np.arange(-80, 81, 1)
el_pattern = 20 * np.log10((np.cos(el_angle / 180 * np.pi)) ** 20) + antenna_gain

tx_channel = dict(
    location=(0, 0, 0),
    azimuth_angle=az_angle,
    azimuth_pattern=az_pattern,
    elevation_angle=el_angle,
    elevation_pattern=el_pattern,
)

tx = Transmitter(
    f=[76.3e9, 76.7e9],
    t=5.12e-05,
    tx_power=13,
    prp=5.5e-05,
    pulses=512,
    channels=[tx_channel],
)

rx_channel = dict(
    location=(0, 0, 0),
    azimuth_angle=az_angle,
    azimuth_pattern=az_pattern,
    elevation_angle=el_angle,
    elevation_pattern=el_pattern,
)

rx = Receiver(
    fs=20e6,
    noise_figure=11,
    rf_gain=20,
    load_resistor=500,
    baseband_gain=30,
    bb_type="real",
    channels=[rx_channel],
)

radar = Radar(transmitter=tx, receiver=rx)

target_1 = {
    "model": "../radarsimpy/models/cr.stl",
    "unit": "m",
    "location": (100, 0, 0),
    "speed": (0, 0, 0),
}

targets = [target_1]

from radarsimpy.simulator import sim_radar

data = sim_radar(radar, targets)
timestamp = data["timestamp"]
baseband = data["baseband"]+data["noise"]
noise = data["noise"]

import radarsimpy.processing as proc

range_doppler = np.fft.fftshift(
    proc.range_doppler_fft(baseband), axes=1
)

noise_range_doppler = np.fft.fftshift(
    proc.range_doppler_fft(noise), axes=1
)

max_per_range_bin = np.max(np.abs(range_doppler), axis=1)
noise_mean = np.mean(np.abs(noise_range_doppler), axis=1)

valid_range_bins = int(radar.sample_prop["samples_per_pulse"]/2)

max_range = (
    3e8
    * radar.radar_prop["receiver"].bb_prop["fs"]
    * radar.radar_prop["transmitter"].waveform_prop["pulse_length"]
    / radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
    / 4
)

print(max_range)

import matplotlib.pyplot as plt

range_axis = np.linspace(
    0, max_range, valid_range_bins, endpoint=False
)
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('WebAgg')
plt.style.use('_mpl-gallery')

fig, ax = plt.subplots(figsize=(8,4), layout='constrained', frameon=False)
plt.tight_layout(pad=3.0)  # Add extra padding around the figure

ax.set_xlabel('Range (m)', fontsize=20)
ax.set_ylabel('Amplitude (dB)', fontsize=20)

ax.plot(
    range_axis,
    20*np.log10(max_per_range_bin[0, 0:valid_range_bins]),
)


ax.plot(
    range_axis,
    20*np.log10(noise_mean[0, 0:valid_range_bins]),
)

ax.set(xlim=(-10, 200),
       ylim=(0, 50))


plt.show()
