import wfdb
import numpy as np


# ============================================================
# 1. LOAD DETECTOR 2 R-PEAKS
# ============================================================

detected_peaks = np.load(
    "a01_detector2_adaptive_r_peaks.npy"
)

# Make sure peaks are sorted
detected_peaks = np.sort(
    detected_peaks.astype(int)
)


# ============================================================
# 2. LOAD REFERENCE QRS ANNOTATIONS
# ============================================================

annotation = wfdb.rdann(
    "data/ao1/a01",
    "qrs"
)

reference_peaks = np.array(
    annotation.sample,
    dtype=int
)

reference_peaks = np.sort(
    reference_peaks
)

fs = 100


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("==============================================")
print("DETECTOR 2 - ADAPTIVE THRESHOLD RESULTS")
print("==============================================")

print(
    "Detected R-peaks:",
    len(detected_peaks)
)

print(
    "Reference QRS annotations:",
    len(reference_peaks)
)


# ============================================================
# 4. MATCHING TOLERANCE
# ============================================================

tolerance_ms = 50

tolerance_samples = int(
    tolerance_ms / 1000 * fs
)

print(
    "Matching tolerance:",
    tolerance_ms,
    "ms"
)

print(
    "Tolerance in samples:",
    tolerance_samples
)


# ============================================================
# 5. ONE-TO-ONE PEAK MATCHING
# ============================================================

matched_pairs = []

i = 0
j = 0

while (
    i < len(detected_peaks)
    and j < len(reference_peaks)
):

    detected = detected_peaks[i]
    reference = reference_peaks[j]

    difference = detected - reference

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    if abs(difference) <= tolerance_samples:

        matched_pairs.append(
            (detected, reference)
        )

        i += 1
        j += 1

    # --------------------------------------------------------
    # Detected peak occurs before reference peak
    # --------------------------------------------------------

    elif detected < reference:

        i += 1

    # --------------------------------------------------------
    # Reference peak occurs before detected peak
    # --------------------------------------------------------

    else:

        j += 1


# ============================================================
# 6. CALCULATE TP, FP, FN
# ============================================================

TP = len(matched_pairs)

FP = len(detected_peaks) - TP

FN = len(reference_peaks) - TP


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

if TP + FN > 0:

    sensitivity = (
        TP / (TP + FN)
    )

else:

    sensitivity = 0


if TP + FP > 0:

    precision = (
        TP / (TP + FP)
    )

else:

    precision = 0


if sensitivity + precision > 0:

    f1_score = (
        2
        * sensitivity
        * precision
        / (
            sensitivity
            + precision
        )
    )

else:

    f1_score = 0


# ============================================================
# 8. TIMING ERROR
# ============================================================

if len(matched_pairs) > 0:

    timing_errors = np.array(
        [
            abs(
                detected - reference
            )
            for detected, reference
            in matched_pairs
        ],
        dtype=float
    )

    mean_error_ms = (
        np.mean(timing_errors)
        / fs
        * 1000
    )

    median_error_ms = (
        np.median(timing_errors)
        / fs
        * 1000
    )

    max_error_ms = (
        np.max(timing_errors)
        / fs
        * 1000
    )

else:

    mean_error_ms = 0
    median_error_ms = 0
    max_error_ms = 0


# ============================================================
# 9. PRINT VALIDATION RESULTS
# ============================================================

print("\n----------------------------------------------")
print("VALIDATION RESULTS")
print("----------------------------------------------")

print(
    "True Positives (TP):",
    TP
)

print(
    "False Positives (FP):",
    FP
)

print(
    "False Negatives (FN):",
    FN
)

print(
    "\nSensitivity:",
    round(
        sensitivity * 100,
        2
    ),
    "%"
)

print(
    "Precision (PPV):",
    round(
        precision * 100,
        2
    ),
    "%"
)

print(
    "F1 Score:",
    round(
        f1_score * 100,
        2
    ),
    "%"
)


# ============================================================
# 10. TIMING ERROR RESULTS
# ============================================================

print("\n----------------------------------------------")
print("R-PEAK TIMING ERROR")
print("----------------------------------------------")

print(
    "Mean absolute error:",
    round(
        mean_error_ms,
        2
    ),
    "ms"
)

print(
    "Median absolute error:",
    round(
        median_error_ms,
        2
    ),
    "ms"
)

print(
    "Maximum absolute error:",
    round(
        max_error_ms,
        2
    ),
    "ms"
)


# ============================================================
# 11. FIRST 10 MATCHED PEAKS
# ============================================================

print("\n----------------------------------------------")
print("FIRST 10 MATCHED R-PEAKS")
print("----------------------------------------------")

for detected, reference in matched_pairs[:10]:

    detected_time = detected / fs

    reference_time = reference / fs

    error_ms = (
        abs(
            detected - reference
        )
        / fs
        * 1000
    )

    print(
        f"Detected: {detected_time:.3f} s | "
        f"Reference: {reference_time:.3f} s | "
        f"Error: {error_ms:.2f} ms"
    )


# ============================================================
# 12. SAVE MATCHED PAIRS
# ============================================================

matched_pairs_array = np.array(
    matched_pairs,
    dtype=int
)

np.save(
    "a01_detector2_matched_pairs.npy",
    matched_pairs_array
)


print("\n----------------------------------------------")
print("OUTPUT")
print("----------------------------------------------")

print(
    "Matched pairs saved as:"
)

print(
    "a01_detector2_matched_pairs.npy"
)


# ============================================================
# 13. COMPLETE
# ============================================================

print("\n==============================================")
print("DETECTOR 2 VERIFICATION COMPLETE")
print("==============================================")