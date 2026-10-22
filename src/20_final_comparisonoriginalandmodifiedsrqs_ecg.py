
from pathlib import Path

import numpy as np
import wfdb

# ============================================================
# 1. LOAD ORIGINAL AND MODIFIED SQRS125 ANNOTATIONS
# ============================================================

original_path = r"C:\msys64\tmp\sqrs125_test\a01"
modified_path = r"C:\msys64\tmp\sqrs125_no_minus4\a01"

# ECG sampling frequency and sqrs125 annotation time base
ecg_fs = 100.0
annotation_fs = 125.0

original_ann = wfdb.rdann(original_path, "qrs")
modified_ann = wfdb.rdann(modified_path, "qrs")

original_raw = np.asarray(original_ann.sample, dtype=np.int64)
modified_raw = np.asarray(modified_ann.sample, dtype=np.int64)

# Convert both annotation positions to the same 100 Hz grid.
original = np.rint(original_raw * ecg_fs / annotation_fs).astype(int)
modified = np.rint(modified_raw * ecg_fs / annotation_fs).astype(int)

print("=" * 65)
print("ORIGINAL VS MODIFIED SQRS125 ANNOTATIONS")
print("=" * 65)
print(f"ECG sampling frequency: {ecg_fs:.0f} Hz")
print(f"Annotation time base assumed: {annotation_fs:.0f} Hz")
print(f"Original annotation count: {len(original)}")
print(f"Modified annotation count: {len(modified)}")

# ============================================================
# 2. MATCH ANNOTATIONS ONE-TO-ONE
# ============================================================

# Match each original annotation to its nearest unused modified
# annotation within 100 ms. This is deliberately wider than the
# expected 32 ms shift and accommodates small detection differences.
tolerance_samples = int(round(0.100 * ecg_fs))

modified_order = np.argsort(modified)
modified_sorted = modified[modified_order]
used = np.zeros(len(modified_sorted), dtype=bool)

matched_original = []
matched_modified = []
unmatched_original = []

for original_sample in original:
    left = np.searchsorted(
        modified_sorted,
        original_sample - tolerance_samples,
        side="left",
    )
    right = np.searchsorted(
        modified_sorted,
        original_sample + tolerance_samples,
        side="right",
    )

    candidates = np.arange(left, right)
    candidates = candidates[~used[candidates]]

    if len(candidates) == 0:
        unmatched_original.append(original_sample)
        continue

    distances = np.abs(modified_sorted[candidates] - original_sample)
    best = candidates[np.argmin(distances)]

    used[best] = True
    matched_original.append(original_sample)
    matched_modified.append(modified_sorted[best])

matched_original = np.asarray(matched_original, dtype=int)
matched_modified = np.asarray(matched_modified, dtype=int)

unmatched_modified = modified_sorted[~used]

if len(matched_original) == 0:
    raise RuntimeError(
        "No annotations matched. Check the annotation time-base assumption."
    )

# Positive shift means removing -4 moved an annotation later.
shift_samples = matched_modified - matched_original
shift_ms = shift_samples * 1000.0 / ecg_fs

# ============================================================
# 3. REPORT THE DIRECT SHIFT
# ============================================================

print("\nONE-TO-ONE MATCHING")
print("Matched annotations:", len(matched_original))
print("Unmatched original annotations:", len(unmatched_original))
print("Unmatched modified annotations:", len(unmatched_modified))

print("\nSHIFT CAUSED BY REMOVING -4")
print(f"Mean shift: {np.mean(shift_ms):.3f} ms")
print(f"Median shift: {np.median(shift_ms):.3f} ms")
print(f"Standard deviation: {np.std(shift_ms):.3f} ms")
print(f"Minimum shift: {np.min(shift_ms):.3f} ms")
print(f"Maximum shift: {np.max(shift_ms):.3f} ms")

print("\nSHIFT DISTRIBUTION")
for shift in np.unique(shift_ms):
    count = int(np.sum(shift_ms == shift))
    print(f"{shift:+.0f} ms: {count} annotations")

print("\nFirst 20 matched comparisons:")
print("Original sample | Modified sample | Shift (ms)")
for old, new, shift in zip(
    matched_original[:20],
    matched_modified[:20],
    shift_ms[:20],
):
    print(f"{old:15d} | {new:15d} | {shift:+8.1f}")

# ============================================================
# 4. SAVE THE RESULTS
# ============================================================

output_path = (
    Path(__file__).resolve().parent
    / "a01_sqrs125_original_vs_no_minus4.csv"
)

comparison = np.column_stack(
    [
        matched_original,
        matched_modified,
        shift_samples,
        shift_ms,
    ]
)

np.savetxt(
    output_path,
    comparison,
    delimiter=",",
    header=(
        "original_sample_100Hz,"
        "modified_sample_100Hz,"
        "shift_samples_100Hz,"
        "shift_ms"
    ),
    comments="",
    fmt=["%d", "%d", "%d", "%.1f"],
)

print(f"\nSaved comparison: {output_path}")
