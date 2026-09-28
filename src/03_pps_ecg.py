import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ==================================================
# 1. LOAD ECG
# ==================================================

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", len(ecg))
print("Duration:", len(ecg) / fs, "seconds")


# ==================================================
# 2. CREATE BAND-PASS FILTER FUNCTION
# ==================================================

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order=4):

    nyquist = fs / 2

    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


# ==================================================
# 3. FILTER A: 0.5–40 Hz
# ==================================================

filtered_40 = bandpass_filter(
    ecg,
    low_cutoff=0.5,
    high_cutoff=40,
    fs=fs
)


# ==================================================
# 4. FILTER B: 0.5–20 Hz
# ==================================================

filtered_20 = bandpass_filter(
    ecg,
    low_cutoff=0.5,
    high_cutoff=20,
    fs=fs
)


# ==================================================
# 5. DIFFERENCE BETWEEN THE TWO FILTERS
# ==================================================

difference = filtered_40 - filtered_20


# ==================================================
# 6. CREATE TIME AXIS
# ==================================================

time = np.arange(len(ecg)) / fs


# ==================================================
# 7. PLOT EVERYTHING
# ==================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 11),
    sharex=True
)


# ---------- Raw ECG ----------

axes[0].plot(time, ecg)

axes[0].set_title("Raw ECG")
axes[0].set_ylabel("Amplitude")
axes[0].grid()


# ---------- Filter A ----------

axes[1].plot(time, filtered_40)

axes[1].set_title("Filter A: 0.5–40 Hz")
axes[1].set_ylabel("Amplitude")
axes[1].grid()


# ---------- Filter B ----------

axes[2].plot(time, filtered_20)

axes[2].set_title("Filter B: 0.5–20 Hz")
axes[2].set_ylabel("Amplitude")
axes[2].grid()


# ---------- Difference ----------

axes[3].plot(time, difference)

axes[3].set_title(
    "Difference: 0.5–40 Hz − 0.5–20 Hz"
)

axes[3].set_ylabel("Amplitude")
axes[3].set_xlabel("Time (seconds)")
axes[3].grid()


# ==================================================
# 8. DISPLAY FIRST 20 SECONDS
# ==================================================

axes[0].set_xlim(0, 20)


# ==================================================
# 9. FINAL DISPLAY
# ==================================================

plt.suptitle(
    "High-Frequency Filter Comparison - Record a01"
)

plt.tight_layout()

plt.show()