import wfdb
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. LOAD ECG
# ==================================================

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling frequency:", fs, "Hz")
print("Total samples:", len(ecg))


# ==================================================
# 2. TAKE FIRST 20 SECONDS
# ==================================================

duration = 20

N = int(duration * fs)

ecg_segment = ecg[:N]

print("Samples analyzed:", len(ecg_segment))
print("Analysis duration:", len(ecg_segment) / fs, "seconds")


# ==================================================
# 3. REMOVE DC COMPONENT
# ==================================================

ecg_segment = ecg_segment - np.mean(ecg_segment)


# ==================================================
# 4. CALCULATE FFT
# ==================================================

fft_result = np.fft.rfft(ecg_segment)

frequency = np.fft.rfftfreq(
    N,
    d=1/fs
)


# ==================================================
# 5. CALCULATE MAGNITUDE
# ==================================================

magnitude = np.abs(fft_result) / N

magnitude[1:-1] = 2 * magnitude[1:-1]


# ==================================================
# 6. PLOT TIME-DOMAIN ECG
# ==================================================

time = np.arange(N) / fs

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 8)
)

axes[0].plot(time, ecg_segment)

axes[0].set_title(
    "ECG Signal - First 20 Seconds"
)

axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")
axes[0].grid()


# ==================================================
# 7. PLOT FREQUENCY-DOMAIN ECG
# ==================================================

axes[1].plot(frequency, magnitude)

axes[1].set_title(
    "FFT Magnitude Spectrum - ECG"
)

axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid()

axes[1].set_xlim(0, 50)


# ==================================================
# 8. DISPLAY
# ==================================================

plt.suptitle(
    "Time Domain and Frequency Domain Analysis - Record a01"
)

plt.tight_layout()

plt.show()