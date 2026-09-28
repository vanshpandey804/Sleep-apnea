import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks


# ============================================================
# 1. LOAD ECG
# ============================================================

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling frequency:", fs, "Hz")


# ============================================================
# 2. 0.5–40 Hz ZERO-PHASE BANDPASS FILTER
# ============================================================

low_cutoff = 0.5
high_cutoff = 40

sos = butter(
    4,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, ecg)


# ============================================================
# 3. USE FIRST 20 SECONDS FOR VISUAL CHECK
# ============================================================

duration = 20

samples = int(duration * fs)

signal = filtered_ecg[:samples]


# ============================================================
# 4. DIFFERENTIATION
# ============================================================

differentiated = np.diff(signal)

# Keep same length
differentiated = np.append(
    differentiated,
    differentiated[-1]
)


# ============================================================
# 5. SQUARING
# ============================================================

squared = differentiated ** 2


# ============================================================
# 6. MOVING-WINDOW INTEGRATION
# ============================================================

window_seconds = 0.15

window_samples = int(
    window_seconds * fs
)

window = np.ones(window_samples) / window_samples

integrated = np.convolve(
    squared,
    window,
    mode="same"
)


# ============================================================
# 7. FIND QRS CANDIDATES
# ============================================================

threshold = 0.30 * np.max(integrated)

min_distance = int(0.25 * fs)

candidate_peaks, properties = find_peaks(
    integrated,
    height=threshold,
    distance=min_distance
)


# ============================================================
# 8. FIND ACTUAL R-PEAKS
# ============================================================

search_window = int(0.08 * fs)

r_peaks = []

for candidate in candidate_peaks:

    start = max(
        0,
        candidate - search_window
    )

    end = min(
        len(signal),
        candidate + search_window
    )

    local_peak = start + np.argmax(
        signal[start:end]
    )

    r_peaks.append(local_peak)


r_peaks = np.array(r_peaks)


# Remove duplicate R-peaks
r_peaks = np.unique(r_peaks)


# ============================================================
# 9. CONVERT SAMPLE NUMBERS TO TIME
# ============================================================

r_peak_times = r_peaks / fs


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("Number of detected R-peaks:", len(r_peaks))

print("\nDetected R-peak times:")

for time in r_peak_times:

    print(f"{time:.3f} seconds")


# ============================================================
# 11. PLOT RESULTS
# ============================================================

time = np.arange(len(signal)) / fs

plt.figure(figsize=(14, 9))


# ECG
plt.subplot(3, 1, 1)

plt.plot(
    time,
    signal,
    label="Filtered ECG"
)

plt.plot(
    r_peak_times,
    signal[r_peaks],
    "ro",
    label="Detected R-peaks"
)

plt.ylabel("Amplitude")
plt.title("QRS Detection - Filtered ECG")
plt.legend()
plt.grid()


# Differentiated + squared
plt.subplot(3, 1, 2)

plt.plot(
    time,
    squared
)

plt.ylabel("Squared derivative")
plt.title("Differentiated and Squared ECG")
plt.grid()


# Integrated signal
plt.subplot(3, 1, 3)

plt.plot(
    time,
    integrated,
    label="Integrated signal"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Detection threshold"
)

plt.plot(
    candidate_peaks / fs,
    integrated[candidate_peaks],
    "ro",
    label="QRS candidates"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Integrated amplitude")
plt.title("Moving-Window Integration")
plt.legend()
plt.grid()


plt.tight_layout()

plt.show()