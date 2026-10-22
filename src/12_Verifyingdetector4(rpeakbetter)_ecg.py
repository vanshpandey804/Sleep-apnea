
import wfdb
import numpy as np

# ============================================================
# 1. LOAD DETECTED R-PEAKS AND REFERENCE ANNOTATIONS
# ============================================================

record_path = "data/ao1/a01"
detected_path = "a01_detector4_r_peaks.npy"

try:
    detected = np.load(detected_path).astype(int)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Cannot find {detected_path}. "
        "Run 09_Rpeakdetection_better_ecg.py first."
    )

annotation = wfdb.rdann(record_path, "qrs")
reference = np.asarray(annotation.sample, dtype=int)

record = wfdb.rdrecord(record_path)
fs = float(record.fs)

# Sort and remove duplicate indices for safe one-to-one matching.
detected = np.unique(detected)
reference = np.unique(reference)

# ============================================================
# 2. VALIDATION SETTINGS
# ============================================================

tolerance_ms = 50
tolerance_samples = int(round(tolerance_ms * fs / 1000))

# ============================================================
# 3. ONE-TO-ONE MATCHING
# ============================================================

i = 0
j = 0

tp = 0
fp = 0
fn = 0

timing_errors_samples = []
matched_pairs = []

while i < len(detected) and j < len(reference):
    difference = int(detected[i] - reference[j])

    if abs(difference) <= tolerance_samples:
        # A detected peak matches one reference annotation.
        tp += 1
        timing_errors_samples.append(difference)

        matched_pairs.append([
            detected[i],
            reference[j],
            difference
        ])

        i += 1
        j += 1

    elif difference < -tolerance_samples:
        # This detection is too early to match the current
        # reference annotation.
        fp += 1
        i += 1

    else:
        # This reference annotation is too early to match
        # the current detected peak.
        fn += 1
        j += 1

# Count unmatched items remaining at the end.
fp += len(detected) - i
fn += len(reference) - j

# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0

f1_score = (
    2 * sensitivity * precision / (sensitivity + precision)
    if (sensitivity + precision)
    else 0.0
)

timing_errors_samples = np.asarray(
    timing_errors_samples,
    dtype=float
)

timing_errors_ms = timing_errors_samples / fs * 1000

# ============================================================
# 5. SAVE MATCHED PAIRS
# ============================================================

matched_pairs = np.asarray(matched_pairs, dtype=int).reshape(-1, 3)

np.save(
    "a01_detector4_matched_pairs.npy",
    matched_pairs
)

# Columns: detected sample, reference sample, signed error in samples.
# Positive error means the detection occurred after the reference.
# Negative error means the detection occurred before the reference.

# ============================================================
# 6. PRINT REPORT
# ============================================================

print("\n==============================================")
print("DETECTOR 4 - VALIDATION RESULTS")
print("==============================================")

print("Sampling frequency:", fs, "Hz")
print("Matching tolerance:", tolerance_ms, "ms")
print("Tolerance in samples:", tolerance_samples)

print("\nDetected R-peaks:", len(detected))
print("Reference QRS annotations:", len(reference))

print("\nTrue Positives (TP):", tp)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)

print(f"\nSensitivity: {sensitivity * 100:.2f}%")
print(f"Precision:   {precision * 100:.2f}%")
print(f"F1 score:    {f1_score * 100:.2f}%")

if len(timing_errors_ms):
    print("\nTiming error statistics for matched peaks:")

    print(
        "Mean absolute error:",
        f"{np.mean(np.abs(timing_errors_ms)):.2f} ms"
    )

    print(
        "Median absolute error:",
        f"{np.median(np.abs(timing_errors_ms)):.2f} ms"
    )

    print(
        "Maximum absolute error:",
        f"{np.max(np.abs(timing_errors_ms)):.2f} ms"
    )

    print(
        "Mean signed error:",
        f"{np.mean(timing_errors_ms):.2f} ms"
    )

else:
    print("\nNo matched peaks; timing errors cannot be calculated.")

print("\nMatched pairs saved to: a01_detector4_matched_pairs.npy")
print("Validation complete.")
