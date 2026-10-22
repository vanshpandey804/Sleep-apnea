import wfdb
import numpy as np


# ============================================================
# 1. LOAD DETECTOR 1 RESULTS
# ============================================================

detected_peaks = np.load(
    "a01_detector1_fixed_r_peaks.npy"
)

detected_peaks = np.sort(
    detected_peaks
)


# ============================================================
# 2. LOAD a01.qrs REFERENCE ANNOTATIONS
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


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

fs = 100

print("==============================================")
print("DETECTOR 1 VALIDATION")
print("==============================================")

print()
print("Sampling frequency:", fs, "Hz")

print(
    "Detected R-peaks:",
    len(detected_peaks)
)

print(
    "Reference qrs annotations:",
    len(reference_peaks)
)


# ============================================================
# 4. MATCHING TOLERANCE
# ============================================================

# 50 ms tolerance
tolerance_ms = 50

tolerance_samples = int(
    tolerance_ms * fs / 1000
)

print()
print(
    "Matching tolerance:",
    tolerance_ms,
    "ms"
)

print(
    "Matching tolerance:",
    tolerance_samples,
    "samples"
)


# ============================================================
# 5. MATCH DETECTED PEAKS WITH REFERENCE PEAKS
# ============================================================

matched_pairs = []

detected_index = 0
reference_index = 0

while (
    detected_index < len(detected_peaks)
    and reference_index < len(reference_peaks)
):

    detected = detected_peaks[
        detected_index
    ]

    reference = reference_peaks[
        reference_index
    ]

    difference = detected - reference


    # --------------------------------------------------------
    # Detection matches reference
    # --------------------------------------------------------

    if abs(difference) <= tolerance_samples:

        matched_pairs.append(
            (
                detected,
                reference,
                difference
            )
        )

        detected_index += 1
        reference_index += 1


    # --------------------------------------------------------
    # Detection occurs too early
    # --------------------------------------------------------

    elif detected < reference:

        detected_index += 1


    # --------------------------------------------------------
    # Reference occurs before detection
    # --------------------------------------------------------

    else:

        reference_index += 1


# ============================================================
# 6. CALCULATE TP, FP AND FN
# ============================================================

TP = len(matched_pairs)

FP = len(detected_peaks) - TP

FN = len(reference_peaks) - TP


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

if TP + FN > 0:

    sensitivity = TP / (
        TP + FN
    )

else:

    sensitivity = 0


if TP + FP > 0:

    precision = TP / (
        TP + FP
    )

else:

    precision = 0


if precision + sensitivity > 0:

    f1_score = (
        2
        * precision
        * sensitivity
        / (
            precision
            + sensitivity
        )
    )

else:

    f1_score = 0


# ============================================================
# 8. R-PEAK TIMING ERROR
# ============================================================

if len(matched_pairs) > 0:

    timing_errors_samples = np.array(
        [
            pair[2]
            for pair in matched_pairs
        ]
    )

    timing_errors_ms = (
        timing_errors_samples
        / fs
        * 1000
    )

    mean_abs_error = np.mean(
        np.abs(timing_errors_ms)
    )

    median_abs_error = np.median(
        np.abs(timing_errors_ms)
    )

    max_abs_error = np.max(
        np.abs(timing_errors_ms)
    )

else:

    mean_abs_error = 0
    median_abs_error = 0
    max_abs_error = 0


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("RESULTS")
print("==============================================")

print()
print("True Positives (TP):", TP)

print(
    "False Positives (FP):",
    FP
)

print(
    "False Negatives (FN):",
    FN
)

print()
print(
    f"Sensitivity: {sensitivity * 100:.2f}%"
)

print(
    f"Precision (PPV): {precision * 100:.2f}%"
)

print(
    f"F1-score: {f1_score * 100:.2f}%"
)

print()
print(
    f"Mean absolute R-peak timing error: "
    f"{mean_abs_error:.2f} ms"
)

print(
    f"Median absolute R-peak timing error: "
    f"{median_abs_error:.2f} ms"
)

print(
    f"Maximum absolute R-peak timing error: "
    f"{max_abs_error:.2f} ms"
)


# ============================================================
# 10. SHOW FIRST 10 MATCHES
# ============================================================

print()
print("==============================================")
print("FIRST 10 MATCHED R-PEAKS")
print("==============================================")

print(
    "Detected(s)    Reference(s)    Error(ms)"
)

for detected, reference, difference in matched_pairs[:10]:

    detected_time = detected / fs
    reference_time = reference / fs
    error_ms = difference / fs * 1000

    print(
        f"{detected_time:10.3f}"
        f"{reference_time:16.3f}"
        f"{error_ms:12.2f}"
    )


# ============================================================
# 11. SAVE VALIDATION RESULTS
# ============================================================

np.save(
    "a01_detector1_matched_pairs.npy",
    np.array(
        matched_pairs,
        dtype=int
    )
)

print()
print(
    "Matched results saved to:"
)

print(
    "a01_detector1_matched_pairs.npy"
)