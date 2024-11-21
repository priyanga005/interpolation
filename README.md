import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Create a sample signal (sine wave) for demonstration
fs = 10  # Original sampling rate (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector (1 second duration)
x = np.sin(2 * np.pi * 1 * t)  # Original signal with frequency 1 Hz

# Interpolation factor (up-sampling factor)
L = 4  # Upsample by a factor of 4

# Step 1: Upsample by inserting zeros between samples
x_upsampled = np.zeros(len(x) * L)
x_upsampled[::L] = x  # Insert original samples at every Lth position

# Step 2: Low-pass filtering (using a sinc filter or FIR filter)
# Design an FIR low-pass filter (ideal for sinc interpolation)
nyquist = fs / 2
cutoff = nyquist / L  # Cutoff frequency is below the Nyquist of the upsampled signal
numtaps = 64  # Number of filter taps (filter length)

# Use firwin to create a low-pass filter
b = signal.firwin(numtaps, cutoff / (fs * L), window='hamming')
x_interpolated = signal.lfilter(b, 1.0, x_upsampled)

# Plotting the results
plt.figure(figsize=(12, 6))

# Original signal
plt.subplot(3, 1, 1)
plt.plot(t, x, label="Original Signal", color='b')
plt.title("Original Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.grid(True)

# Upsampled signal (with inserted zeros)
t_upsampled = np.arange(0, len(x_upsampled)) / (fs * L)
plt.subplot(3, 1, 2)
plt.stem(t_upsampled, x_upsampled, label="Upsampled Signal (with zeros)", basefmt=" ", use_line_collection=True)
plt.title("Upsampled Signal (with Zeros)")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.grid(True)

# Interpolated signal (after low-pass filtering)
t_interpolated = np.arange(0, len(x_interpolated)) / (fs * L)
plt.subplot(3, 1, 3)
plt.plot(t_interpolated, x_interpolated, label="Interpolated Signal", color='r')
plt.title("Interpolated Signal")
plt.xlabel("Time [seconds]")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
