
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================================================
# 1. LOAD ECG AND MATCHED PAIRS
# ============================================================

record_path = "data/ao1/a01"
pairs_path = "a01_detector4_matched_pairs.npy"

record = wfdb.rdrecord(record_path)
ecg = record.p_signal[:, 0].astype(float)
fs = float(record.fs)

pairs = np.load(pairs_path)

# Columns:
# 0 = detected sample
# 1 = reference sample
# 2 = signed error in samples
detected = pairs[:, 0].astype(int)
reference = pairs[:, 1].astype(int)
error_samples = pairs[:, 2].astype(float)

# Positive error means detected peak occurs AFTER reference.
error_ms = error_samples / fs * 1000

print("=" * 55)
print("DETECTOR 4 - TIMING ERROR ANALYSIS")
print("=" * 55)
print("Matched beats:", len(pairs))
print("Sampling frequency:", fs, "Hz")

# ============================================================
# 2. NUMERICAL SUMMARY
# ============================================================

print("\nSIGNED TIMING ERROR")
print("Mean:", round(float(np.mean(error_ms)), 3), "ms")
print("Median:", round(float(np.median(error_ms)), 3), "ms")
print("Standard deviation:", round(float(np.std(error_ms)), 3), "ms")

print("\nABSOLUTE TIMING ERROR")
print("Mean:", round(float(np.mean(np.abs(error_ms))), 3), "ms")
print("Median:", round(float(np.median(np.abs(error_ms))), 3), "ms")
print("90th percentile:", round(float(np.percentile(np.abs(error_ms), 90)), 3), "ms")
print("95th percentile:", round(float(np.percentile(np.abs(error_ms), 95)), 3), "ms")
print("Maximum:", round(float(np.max(np.abs(error_ms))), 3), "ms")

print("\nERROR COUNTS")
print("Detections after reference:", int(np.sum(error_samples > 0)))
print("Detections before reference:", int(np.sum(error_samples < 0)))
print("Exact sample matches:", int(np.sum(error_samples == 0)))

# ============================================================
# 3. SAVE ERROR DATA
# ============================================================

error_data = np.column_stack([
    reference / fs,
    detected / fs,
    error_samples,
    error_ms
])

np.savetxt(
    "a01_detector4_timing_errors.csv",
    error_data,
    delimiter=",",
    header="reference_time_s,detected_time_s,error_samples,error_ms",
    comments="",
    fmt="%.5f"
)

print("\nSaved: a01_detector4_timing_errors.csv")

# ============================================================
# 4. FILTER ECG FOR VISUAL INSPECTION
# ============================================================

sos = butter(
    4,
    [0.5, 40.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, ecg)

# ============================================================
# 5. PLOT ERROR DISTRIBUTION AND ERROR OVER TIME
# ============================================================

reference_times = reference / fs

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# Histogram of signed errors
axes[0].hist(error_ms, bins=np.arange(-55, 61, 10), edgecolor="black")
axes[0].axvline(0, linestyle="--", label="Zero error")
axes[0].axvline(
    np.median(error_ms),
    linestyle="--",
    label=f"Median = {np.median(error_ms):.1f} ms"
)
axes[0].set_xlabel("Signed timing error (ms)")
axes[0].set_ylabel("Number of beats")
axes[0].set_title("Detector 4: Timing-error distribution")
axes[0].legend()
axes[0].grid(True)

# Error across the recording
axes[1].scatter(
    reference_times / 3600,
    error_ms,
    s=3,
    alpha=0.5
)
axes[1].axhline(0, linestyle="--")
axes[1].axhline(
    np.median(error_ms),
    linestyle="--",
    label=f"Median = {np.median(error_ms):.1f} ms"
)
axes[1].set_xlabel("Recording time (hours)")
axes[1].set_ylabel("Signed timing error (ms)")
axes[1].set_title("Timing error throughout the overnight recording")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 6. SELECT REPRESENTATIVE BEATS FOR ECG OVERLAYS
# ============================================================

# Show examples near the beginning, middle, and end.
example_indices = np.unique(
    np.linspace(0, len(pairs) - 1, min(6, len(pairs)), dtype=int)
)

fig, axes = plt.subplots(
    len(example_indices),
    1,
    figsize=(12, 2.8 * len(example_indices)),
    squeeze=False
)

half_window = int(round(0.30 * fs))

for ax, idx in zip(axes[:, 0], example_indices):
    ref = int(reference[idx])
    det = int(detected[idx])

    start = max(0, min(ref, det) - half_window)
    end = min(len(filtered_ecg), max(ref, det) + half_window + 1)

    sample_indices = np.arange(start, end)
    times = (sample_indices - ref) / fs

    ax.plot(times, filtered_ecg[start:end], label="Filtered ECG")
    ax.axvline(0, linestyle="--", label="Reference annotation")
    ax.axvline(
        (det - ref) / fs,
        linestyle=":",
        label="Detected R-peak"
    )

    ax.set_xlabel("Time relative to reference (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Beat {idx + 1}: reference={ref}, detected={det}, "
        f"error={error_ms[idx]:+.1f} ms"
    )
    ax.legend(loc="upper right")
    ax.grid(True)

plt.tight_layout()
plt.show()

print("\nTiming analysis complete.")
