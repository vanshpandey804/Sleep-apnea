
import wfdb
import numpy as np
from pathlib import Path

# ============================================================
# 1. FILE PATHS
# ============================================================

original_path = Path(
    r"C:\Users\vansh\OneDrive\Desktop\Sleep apnea\data\ao1\a01"
)

fresh_path = Path(
    r"C:\msys64\tmp\sqrs125_test\a01"
)

# ============================================================
# 2. LOAD ANNOTATIONS
# ============================================================

original = wfdb.rdann(str(original_path), "qrs")
fresh = wfdb.rdann(str(fresh_path), "qrs")

ref = np.asarray(original.sample, dtype=np.int64)
fresh_raw = np.asarray(fresh.sample, dtype=np.int64)

if len(ref) == 0 or len(fresh_raw) == 0:
    raise ValueError("One or both annotation files contain no annotations.")

# Read the original ECG sampling frequency.
record = wfdb.rdrecord(
    str(original_path),
    sampfrom=0,
    sampto=10
)
original_fs = float(record.fs)

# sqrs125 internally resamples this 100 Hz signal to 125 Hz.
# Convert fresh annotation positions back to the original time base.
sqrs125_fs = 125.0

new = np.rint(
    fresh_raw * original_fs / sqrs125_fs
).astype(np.int64)

# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("=" * 60)
print("FRESH SQRS125 VS ORIGINAL REFERENCE")
print("=" * 60)

print(f"Original sampling frequency: {original_fs:.1f} Hz")
print(f"Assumed sqrs125 time base:    {sqrs125_fs:.1f} Hz")
print(f"Original annotations:         {len(ref)}")
print(f"Fresh annotations:            {len(fresh_raw)}")

print("\nFirst 10 original positions:", ref[:10].tolist())
print("First 10 fresh raw positions:", fresh_raw[:10].tolist())
print("First 10 converted positions:", new[:10].tolist())

print(f"\nOriginal last position: {ref[-1]}")
print(f"Fresh last raw position: {fresh_raw[-1]}")
print(f"Fresh last converted position: {new[-1]}")

# ============================================================
# 4. ONE-TO-ONE BEAT MATCHING
# ============================================================

# At 100 Hz, 50 ms corresponds to 5 samples.
tolerance_ms = 50.0
tolerance_samples = int(
    round(tolerance_ms * original_fs / 1000.0)
)

i = 0
j = 0

matches = []
matched_ref_indices = set()
matched_new_indices = set()

while i < len(ref) and j < len(new):
    difference = int(new[j]) - int(ref[i])

    if abs(difference) <= tolerance_samples:
        matches.append((int(ref[i]), int(new[j])))
        matched_ref_indices.add(i)
        matched_new_indices.add(j)
        i += 1
        j += 1

    elif new[j] < ref[i] - tolerance_samples:
        j += 1

    else:
        i += 1

# Find annotations not matched within the tolerance.
unmatched_ref = [
    int(ref[k])
    for k in range(len(ref))
    if k not in matched_ref_indices
]

unmatched_new = [
    int(new[k])
    for k in range(len(new))
    if k not in matched_new_indices
]

# ============================================================
# 5. PERFORMANCE METRICS
# ============================================================

tp = len(matches)
fp = len(new) - tp
fn = len(ref) - tp

sensitivity = (
    100.0 * tp / len(ref)
    if len(ref) else 0.0
)

precision = (
    100.0 * tp / len(new)
    if len(new) else 0.0
)

if sensitivity + precision > 0:
    f1 = (
        2.0 * sensitivity * precision
        / (sensitivity + precision)
    )
else:
    f1 = 0.0

# ============================================================
# 6. UNMATCHED ANNOTATIONS
# ============================================================

print("\nUNMATCHED ANNOTATIONS")
print("-" * 60)

print(f"Unmatched original annotations: {len(unmatched_ref)}")

for sample in unmatched_ref:
    nearest_index = int(np.argmin(np.abs(new - sample)))
    nearest_sample = int(new[nearest_index])

    difference_ms = (
        (nearest_sample - sample) * 1000.0 / original_fs
    )

    print(
        f"Original sample {sample}; "
        f"nearest fresh sample {nearest_sample}; "
        f"difference {difference_ms:+.2f} ms"
    )

print(f"\nUnmatched fresh annotations: {len(unmatched_new)}")

for sample in unmatched_new:
    nearest_index = int(np.argmin(np.abs(ref - sample)))
    nearest_sample = int(ref[nearest_index])

    difference_ms = (
        (sample - nearest_sample) * 1000.0 / original_fs
    )

    print(
        f"Fresh sample {sample}; "
        f"nearest original sample {nearest_sample}; "
        f"difference {difference_ms:+.2f} ms"
    )

# ============================================================
# 7. TIMING ERRORS FOR MATCHED BEATS
# ============================================================

if matches:
    errors_samples = np.array(
        [
            fresh_sample - ref_sample
            for ref_sample, fresh_sample in matches
        ],
        dtype=np.float64
    )

    errors_ms = errors_samples * 1000.0 / original_fs

    mean_signed_ms = float(np.mean(errors_ms))
    mean_absolute_ms = float(np.mean(np.abs(errors_ms)))
    median_absolute_ms = float(np.median(np.abs(errors_ms)))
    maximum_absolute_ms = float(np.max(np.abs(errors_ms)))

else:
    mean_signed_ms = float("nan")
    mean_absolute_ms = float("nan")
    median_absolute_ms = float("nan")
    maximum_absolute_ms = float("nan")

# ============================================================
# 8. PRINT FINAL RESULTS
# ============================================================

print("\n" + "-" * 60)
print("VALIDATION RESULTS")
print("-" * 60)

print(f"Matching tolerance:   {tolerance_ms:.0f} ms")
print(f"Tolerance samples:    {tolerance_samples}")
print(f"True positives (TP):  {tp}")
print(f"False positives (FP): {fp}")
print(f"False negatives (FN): {fn}")

print(f"\nSensitivity: {sensitivity:.4f}%")
print(f"Precision:   {precision:.4f}%")
print(f"F1 score:    {f1:.4f}%")

print("\nTIMING ERRORS")
print(f"Mean signed error:          {mean_signed_ms:.2f} ms")
print(f"Mean absolute error:        {mean_absolute_ms:.2f} ms")
print(f"Median absolute error:      {median_absolute_ms:.2f} ms")
print(f"Maximum absolute error:     {maximum_absolute_ms:.2f} ms")

print("=" * 60)
print("Comparison complete. Source annotation files unchanged.")
