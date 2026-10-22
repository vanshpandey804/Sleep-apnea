
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ============================================================
# 1. LOAD ECG, REFERENCE ANNOTATIONS, AND DETECTOR 4 RESULTS
# ============================================================

record_path = "data/ao1/a01"

record = wfdb.rdrecord(record_path)
raw_ecg = record.p_signal[:, 0].astype(float)
fs = float(record.fs)

annotations = wfdb.rdann(record_path, "qrs")
reference = np.asarray(annotations.sample, dtype=int)

pairs = np.load("a01_detector4_matched_pairs.npy")

# Column 0: detected sample
# Column 1: corresponding reference sample
detected = pairs[:, 0].astype(int)
matched_reference = pairs[:, 1].astype(int)

print("=" * 65)
print("RAW ECG VS FILTERED ECG VS DETECTOR 4")
print("=" * 65)
print(f"Sampling frequency: {fs} Hz")
print(f"Time per sample: {1000 / fs:.1f} ms")
print(f"Raw ECG samples: {len(raw_ecg)}")
print(f"Reference annotations: {len(reference)}")
print(f"Matched pairs: {len(pairs)}")

if len(pairs) == 0:
    raise ValueError("No matched pairs found.")

if np.any(detected < 0) or np.any(detected >= len(raw_ecg)):
    raise ValueError("Some detected samples are outside the ECG record.")

if np.any(matched_reference < 0) or np.any(
    matched_reference >= len(raw_ecg)
):
    raise ValueError("Some reference samples are outside the ECG record.")

# ============================================================
# 2. APPLY THE SAME ZERO-PHASE FILTER AS DETECTOR 4
# ============================================================

sos = butter(
    4,
    [0.5, 40.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, raw_ecg)

# ============================================================
# 3. DEFINE A FUNCTION TO FIND LOCAL POSITIVE MAXIMA
# ============================================================

# Search around the reference and detector locations.
# The window includes both locations and extends by 250 ms.
# We use the same window for raw and filtered signals.

half_window = int(round(0.25 * fs))


def get_local_maxima(ref, det):
    start = max(0, min(ref, det) - half_window)
    end = min(len(raw_ecg), max(ref, det) + half_window + 1)

    raw_segment = raw_ecg[start:end]
    filtered_segment = filtered_ecg[start:end]

    raw_max_sample = start + int(np.argmax(raw_segment))
    filtered_max_sample = start + int(np.argmax(filtered_segment))

    return raw_max_sample, filtered_max_sample


# ============================================================
# 4. CALCULATE OFFSETS FOR EVERY MATCHED BEAT
# ============================================================

raw_max_samples = np.empty(len(pairs), dtype=int)
filtered_max_samples = np.empty(len(pairs), dtype=int)

for i, (ref, det) in enumerate(
    zip(matched_reference, detected)
):
    raw_max_samples[i], filtered_max_samples[i] = (
        get_local_maxima(int(ref), int(det))
    )

# Positive offset means the first sample is later than the reference.
detector_offsets_ms = (
    detected - matched_reference
) / fs * 1000

raw_offsets_ms = (
    raw_max_samples - matched_reference
) / fs * 1000

filtered_offsets_ms = (
    filtered_max_samples - matched_reference
) / fs * 1000

# Positive means Detector 4 occurs after the filtered maximum.
detector_vs_filtered_ms = (
    detected - filtered_max_samples
) / fs * 1000

raw_vs_filtered_ms = (
    raw_max_samples - filtered_max_samples
) / fs * 1000


# ============================================================
# 5. PRINT STATISTICS ACROSS THE FULL RECORD
# ============================================================

def report_offsets(name, values):
    values = np.asarray(values, dtype=float)

    print(f"\n{name}")
    print("-" * 65)
    print(f"Mean:   {np.mean(values):+.2f} ms")
    print(f"Median: {np.median(values):+.2f} ms")
    print(f"Std:    {np.std(values):.2f} ms")
    print(f"Minimum: {np.min(values):+.2f} ms")
    print(f"Maximum: {np.max(values):+.2f} ms")

    print(
        "Within +/-10 ms of zero: "
        f"{np.mean(np.abs(values) <= 10) * 100:.2f}%"
    )


print("\n" + "=" * 65)
print("FULL-RECORD TIMING STATISTICS")
print("=" * 65)

report_offsets(
    "1. Detector 4 relative to reference",
    detector_offsets_ms
)

report_offsets(
    "2. Raw ECG maximum relative to reference",
    raw_offsets_ms
)

report_offsets(
    "3. Filtered ECG maximum relative to reference",
    filtered_offsets_ms
)

report_offsets(
    "4. Detector 4 relative to filtered ECG maximum",
    detector_vs_filtered_ms
)

report_offsets(
    "5. Raw ECG maximum relative to filtered ECG maximum",
    raw_vs_filtered_ms
)

print("\n" + "=" * 65)
print("DETECTOR 4 PEAK-ALIGNMENT CHECK")
print("=" * 65)

for tolerance_ms in [0, 10, 20, 30, 40, 50]:
    percentage = (
        np.mean(np.abs(detector_vs_filtered_ms) <= tolerance_ms)
        * 100
    )

    print(
        f"Detector 4 within +/-{tolerance_ms:2d} ms "
        f"of filtered maximum: {percentage:.2f}%"
    )

print("\nOffset distribution: Detector 4 vs reference")
unique_offsets, counts = np.unique(
    np.round(detector_offsets_ms, 1),
    return_counts=True
)

order = np.argsort(counts)[::-1]

for i in order[:10]:
    percentage = counts[i] / len(detector_offsets_ms) * 100

    print(
        f"{unique_offsets[i]:+6.1f} ms: "
        f"{counts[i]:6d} beats ({percentage:.2f}%)"
    )


# ============================================================
# 6. PLOT SIX BEATS ACROSS THE RECORDING
# ============================================================

example_indices = np.unique(
    np.linspace(0, len(pairs) - 1, 6, dtype=int)
)

fig, axes = plt.subplots(
    len(example_indices),
    1,
    figsize=(14, 17),
    squeeze=False
)

print("\n" + "=" * 65)
print("SELECTED BEATS FOR VISUAL INSPECTION")
print("=" * 65)

for ax, idx in zip(axes[:, 0], example_indices):

    ref = int(matched_reference[idx])
    det = int(detected[idx])

    start = max(0, min(ref, det) - half_window)
    end = min(len(raw_ecg), max(ref, det) + half_window + 1)

    sample_indices = np.arange(start, end)
    time_ms = (sample_indices - ref) / fs * 1000

    raw_segment = raw_ecg[start:end]
    filtered_segment = filtered_ecg[start:end]

    raw_max_sample = int(raw_max_samples[idx])
    filtered_max_sample = int(filtered_max_samples[idx])

    raw_max_offset = (raw_max_sample - ref) / fs * 1000
    filtered_max_offset = (
        filtered_max_sample - ref
    ) / fs * 1000
    detected_offset = (det - ref) / fs * 1000

    print(f"\nMatched beat index: {idx}")
    print(f"Reference sample: {ref}")
    print(f"Detected sample: {det}")
    print(f"Detector 4 offset: {detected_offset:+.1f} ms")
    print(f"Raw maximum offset: {raw_max_offset:+.1f} ms")
    print(f"Filtered maximum offset: {filtered_max_offset:+.1f} ms")
    print(
        "Detector 4 minus filtered maximum: "
        f"{detected_offset - filtered_max_offset:+.1f} ms"
    )

    # Normalize for visual comparison only.
    # Timing calculations use the original unnormalized signals.
    raw_scale = np.max(np.abs(raw_segment))
    filtered_scale = np.max(np.abs(filtered_segment))

    raw_plot = (
        raw_segment / raw_scale
        if raw_scale > 0
        else raw_segment
    )

    filtered_plot = (
        filtered_segment / filtered_scale
        if filtered_scale > 0
        else filtered_segment
    )

    ax.plot(
        time_ms,
        raw_plot,
        label="Raw ECG (normalized)",
        alpha=0.75
    )

    ax.plot(
        time_ms,
        filtered_plot,
        label="Filtered ECG (normalized)"
    )

    ax.axvline(
        0,
        color="green",
        linestyle="--",
        label="Reference annotation"
    )

    ax.axvline(
        detected_offset,
        color="red",
        linestyle=":",
        label="Detector 4 peak"
    )

    ax.axvline(
        raw_max_offset,
        color="blue",
        linestyle="-.",
        label="Raw ECG maximum"
    )

    ax.axvline(
        filtered_max_offset,
        color="purple",
        linestyle="-.",
        label="Filtered ECG maximum"
    )

    ax.set_title(
        f"Beat {idx + 1}: Detector 4 offset "
        f"{detected_offset:+.0f} ms"
    )
    ax.set_xlabel("Time relative to reference (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()


# ============================================================
# 7. SAVE FULL-RECORD TIMING RESULTS
# ============================================================

results = np.column_stack(
    (
        matched_reference,
        detected,
        raw_max_samples,
        filtered_max_samples,
        detector_offsets_ms,
        raw_offsets_ms,
        filtered_offsets_ms,
        detector_vs_filtered_ms
    )
)

np.save(
    "a01_detector4_timing_diagnostics.npy",
    results
)

print("\nSaved detailed timing results to:")
print("a01_detector4_timing_diagnostics.npy")
print("\nInspection complete.")
