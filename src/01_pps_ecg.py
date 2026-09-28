import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --------------------------------------------------
# 1. Load ECG
# --------------------------------------------------

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", len(ecg))


# --------------------------------------------------
# 2. Design band-pass filter
# --------------------------------------------------

low_cutoff = 0.5
high_cutoff = 40.0
order = 4

nyquist = fs / 2

low = low_cutoff / nyquist
high = high_cutoff / nyquist

b, a = butter(order, [low, high], btype="band")


# --------------------------------------------------
# 3. Apply filter
# --------------------------------------------------

filtered_ecg = filtfilt(b, a, ecg)


# --------------------------------------------------
# 4. Calculate removed component
# --------------------------------------------------

removed_component = ecg - filtered_ecg


# --------------------------------------------------
# 5. Create time axis
# --------------------------------------------------

time = np.arange(len(ecg)) / fs


# --------------------------------------------------
# 6. Plot all three signals in one figure
# --------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Raw ECG
axes[0].plot(time, ecg)
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Raw ECG")
axes[0].grid()

# Filtered ECG
axes[1].plot(time, filtered_ecg)
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Filtered ECG (0.5–40 Hz)")
axes[1].grid()

# Removed component
axes[2].plot(time, removed_component)
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (seconds)")
axes[2].set_title("Component Removed by Filter")
axes[2].grid()

# Show only first 20 seconds
axes[0].set_xlim(0, 20)

plt.suptitle("ECG Preprocessing Analysis - Record a01")
plt.tight_layout()

plt.show()