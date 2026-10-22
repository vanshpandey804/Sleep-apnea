
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# Load ECG and matched pairs
record = wfdb.rdrecord("data/ao1/a01")
ecg = record.p_signal[:, 0].astype(float)
fs = float(record.fs)

pairs = np.load("a01_detector4_matched_pairs.npy")
detected = pairs[:, 0].astype(int)
reference = pairs[:, 1].astype(int)

# Apply the same filter used by Detector 4
sos = butter(
    4, [0.5, 40.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)
filtered = sosfiltfilt(sos, ecg)

# Select six beats distributed through the recording
indices = np.unique(
    np.linspace(0, len(pairs) - 1, 6, dtype=int)
)

half_window = int(0.25 * fs)

fig, axes = plt.subplots(
    len(indices), 1,
    figsize=(13, 15),
    squeeze=False
)

for ax, idx in zip(axes[:, 0], indices):
    ref = reference[idx]
    det = detected[idx]

    start = max(0, min(ref, det) - half_window)
    end = min(len(filtered), max(ref, det) + half_window + 1)

    samples = np.arange(start, end)
    time_ms = (samples - ref) / fs * 1000

    ax.plot(time_ms, filtered[start:end], label="Filtered ECG")
    ax.axvline(0, color="green", linestyle="--",
               label="Reference annotation")
    ax.axvline((det - ref) / fs * 1000, color="red",
               linestyle=":", label="Detected R-peak")

    # Show the actual maximum in this displayed window
    segment = filtered[start:end]
    maximum_index = start + int(np.argmax(segment))
    maximum_time_ms = (maximum_index - ref) / fs * 1000

    ax.axvline(
        maximum_time_ms,
        color="purple",
        linestyle="-.",
        label="Window maximum"
    )

    ax.set_title(
        f"Beat {idx + 1}: offset={(det-ref)/fs*1000:.0f} ms"
    )
    ax.set_xlabel("Time from reference (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
