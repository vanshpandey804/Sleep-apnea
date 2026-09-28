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
# 2. Function to create a Butterworth band-pass filter
# --------------------------------------------------

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=4):

    nyquist = fs / 2

    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    b, a = butter(order, [low, high], btype="band")

    filtered = filtfilt(b, a, signal)

    return filtered


# --------------------------------------------------
# 3. Create two different filtered versions
# --------------------------------------------------

# Filter A: current approach
filtered_05 = bandpass_filter(
    ecg,
    low_cutoff=0.5,
    high_cutoff=40,
    fs=fs
)

# Filter B: lower high-pass cutoff
filtered_005 = bandpass_filter(
    ecg,
    low_cutoff=0.05,
    high_cutoff=40,
    fs=fs
)


# --------------------------------------------------
# 4. Calculate removed components
# --------------------------------------------------

removed_05 = ecg - filtered_05
removed_005 = ecg - filtered_005


# --------------------------------------------------
# 5. Create time axis
# --------------------------------------------------

time = np.arange(len(ecg)) / fs


# --------------------------------------------------
# 6. Plot comparison
# --------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

# Raw ECG
axes[0].plot(time, ecg)
axes[0].set_title("Raw ECG")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Filter A
axes[1].plot(time, filtered_05)
axes[1].set_title("Filter A: 0.5–40 Hz")
axes[1].set_ylabel("Amplitude")
axes[1].grid()

# Filter B
axes[2].plot(time, filtered_005)
axes[2].set_title("Filter B: 0.05–40 Hz")
axes[2].set_ylabel("Amplitude")
axes[2].grid()

# Difference between the two filtered signals
axes[3].plot(time, filtered_05 - filtered_005)
axes[3].set_title("Difference: Filter A − Filter B")
axes[3].set_ylabel("Amplitude")
axes[3].set_xlabel("Time (seconds)")
axes[3].grid()

# Show first 20 seconds
axes[0].set_xlim(0, 20)

plt.suptitle("ECG Filter Comparison - Record a01")
plt.tight_layout()

plt.show()