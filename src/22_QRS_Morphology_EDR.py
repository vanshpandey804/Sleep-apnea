
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import PchipInterpolator


# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

RECORD_PATH = ROOT / "data" / "ao1" / "a01"
PEAKS_PATH = ROOT / "a01_detector4_matched_pairs.npy"

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDR_FS = 4.0

# Morphology measurement windows, in seconds
QRS_HALF_WINDOW = 0.080
BASELINE_START = 0.120
BASELINE_END = 0.200


# ==========================================================
# 2. LOAD ECG AND DETECTOR 4 R-PEAKS
# ==========================================================

record = wfdb.rdrecord(str(RECORD_PATH))

fs = float(record.fs)
signals = record.p_signal
signal_names = record.sig_name

# Prefer a channel whose name identifies it as ECG.
ecg_candidates = [
    i for i, name in enumerate(signal_names)
    if "ecg" in name.lower()
]

ecg_channel = ecg_candidates[0] if ecg_candidates else 0
ecg = signals[:, ecg_channel].astype(float)

print("=" * 60)
print("QRS-MORPHOLOGY EDR")
print("=" * 60)
print(f"Sampling frequency: {fs:.2f} Hz")
print(f"Available channels: {signal_names}")
print(f"Selected channel: {signal_names[ecg_channel]}")
print(f"ECG samples: {len(ecg)}")

matched_pairs = np.load(PEAKS_PATH, allow_pickle=False)

if matched_pairs.ndim != 2 or matched_pairs.shape[1] < 1:
    raise ValueError(
        "The matched-pairs file must contain R-peak positions "
        "in its first column."
    )

r_peaks = np.rint(matched_pairs[:, 0]).astype(np.int64)
r_peaks = np.unique(np.sort(r_peaks))

# Keep only peaks with enough surrounding ECG samples.
margin = int(round(BASELINE_END * fs)) + 1

valid = (
    (r_peaks >= margin)
    & (r_peaks < len(ecg) - margin)
)

r_peaks = r_peaks[valid]

if len(r_peaks) < 3:
    raise ValueError("Too few valid R-peaks for morphology analysis.")

peak_times = r_peaks / fs

print(f"Valid Detector 4 R-peaks: {len(r_peaks)}")


# ==========================================================
# 3. FILTER ECG
# ==========================================================

# Use the same 0.5–40 Hz passband as the finalized detector.
sos = butter(
    4,
    [0.5, 40.0],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, ecg)


# ==========================================================
# 4. DEFINE MORPHOLOGY WINDOWS
# ==========================================================

qrs_half = int(round(QRS_HALF_WINDOW * fs))

baseline_inner = int(round(BASELINE_START * fs))
baseline_outer = int(round(BASELINE_END * fs))

r_amplitudes = []
qrs_peak_to_peak = []
qrs_areas = []
kept_times = []

for peak, time_s in zip(r_peaks, peak_times):

    # QRS region: approximately 80 ms before to 80 ms after R.
    qrs_start = peak - qrs_half
    qrs_end = peak + qrs_half + 1

    qrs = filtered_ecg[qrs_start:qrs_end]

    # Estimate the local baseline from regions on either side
    # of the QRS complex, avoiding the immediate QRS region.
    left_baseline = filtered_ecg[
        peak - baseline_outer : peak - baseline_inner
    ]

    right_baseline = filtered_ecg[
        peak + baseline_inner : peak + baseline_outer
    ]

    baseline = np.median(
        np.concatenate([left_baseline, right_baseline])
    )

    # Feature 1: signed R-wave amplitude relative to baseline.
    r_amplitude = filtered_ecg[peak] - baseline

    # Baseline-correct the QRS waveform.
    qrs_corrected = qrs - baseline

    # Feature 2: QRS peak-to-peak amplitude.
    peak_to_peak = (
        np.max(qrs_corrected) - np.min(qrs_corrected)
    )

    # Feature 3: approximate absolute QRS area.
    area = np.sum(np.abs(qrs_corrected)) / fs

    r_amplitudes.append(r_amplitude)
    qrs_peak_to_peak.append(peak_to_peak)
    qrs_areas.append(area)
    kept_times.append(time_s)


# ==========================================================
# 5. SAVE BEAT-LEVEL MORPHOLOGY FEATURES
# ==========================================================

beat_df = pd.DataFrame({
    "time_s": kept_times,
    "r_amplitude": r_amplitudes,
    "qrs_peak_to_peak": qrs_peak_to_peak,
    "qrs_area": qrs_areas
})

beat_output = OUTPUT_DIR / "a01_edr_morphology_beats.csv"
beat_df.to_csv(beat_output, index=False)


# ==========================================================
# 6. INTERPOLATE FEATURES TO A UNIFORM 4 Hz GRID
# ==========================================================

beat_times = beat_df["time_s"].to_numpy()

dt = 1.0 / EDR_FS

start_time = np.ceil(beat_times[0] / dt) * dt
end_time = np.floor(beat_times[-1] / dt) * dt

uniform_times = np.arange(
    start_time,
    end_time + dt / 2.0,
    dt
)

uniform_df = pd.DataFrame({"time_s": uniform_times})

feature_columns = [
    "r_amplitude",
    "qrs_peak_to_peak",
    "qrs_area"
]

for column in feature_columns:
    values = beat_df[column].to_numpy()

    interpolator = PchipInterpolator(
        beat_times,
        values,
        extrapolate=False
    )

    uniform_df[column] = interpolator(uniform_times)

uniform_output = OUTPUT_DIR / "a01_edr_morphology_uniform.csv"
uniform_df.to_csv(uniform_output, index=False)


# ==========================================================
# 7. REPORT RESULTS
# ==========================================================

print("\nRESULTS")
print("-" * 60)
print(f"Beat-level feature rows: {len(beat_df)}")
print(f"Uniform EDR sampling rate: {EDR_FS:.1f} Hz")
print(f"Uniform EDR samples: {len(uniform_df)}")
print(
    f"Duration: "
    f"{(uniform_times[-1] - uniform_times[0]):.2f} s"
)

print("\nFeature summary:")
print(beat_df[feature_columns].describe().loc[["mean", "std", "min", "max"]])

print("\nSaved files:")
print(beat_output)
print(uniform_output)

print("\nMorphology-based EDR generation completed.")
