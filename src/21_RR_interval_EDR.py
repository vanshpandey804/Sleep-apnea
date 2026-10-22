
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.interpolate import PchipInterpolator


# --------------------------------------------------
# 1. PROJECT PATHS AND PARAMETERS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

RECORD_PATH = ROOT / "data" / "ao1" / "a01"
PEAKS_PATH = ROOT / "a01_detector4_matched_pairs.npy"

OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Uniform sampling rate for the interpolated RR signal.
# We will use this for later spectral analysis.
EDR_FS = 4.0


# --------------------------------------------------
# 2. LOAD ECG SAMPLING FREQUENCY
# --------------------------------------------------

header = wfdb.rdheader(str(RECORD_PATH))
fs = float(header.fs)

print("=" * 55)
print("RR-INTERVAL EDR")
print("=" * 55)

print(f"ECG sampling frequency: {fs:.2f} Hz")


# --------------------------------------------------
# 3. LOAD FINALIZED DETECTOR 4 R-PEAKS
# --------------------------------------------------

matched_pairs = np.load(PEAKS_PATH, allow_pickle=False)

if matched_pairs.ndim != 2 or matched_pairs.shape[1] < 1:
    raise ValueError(
        "Unexpected matched-pairs file format. "
        "Expected a 2D array with Detector 4 peaks in column 0."
    )

# Column 0 contains Detector 4 peak locations.
r_peaks = np.rint(matched_pairs[:, 0]).astype(np.int64)

# Sort and remove any duplicate sample positions.
r_peaks = np.unique(np.sort(r_peaks))

if len(r_peaks) < 3:
    raise ValueError("Not enough R-peaks to calculate RR intervals.")

if np.any(np.diff(r_peaks) <= 0):
    raise ValueError("R-peak positions must be strictly increasing.")

print(f"Detector 4 R-peaks loaded: {len(r_peaks)}")


# --------------------------------------------------
# 4. CALCULATE RR INTERVALS
# --------------------------------------------------

# Convert sample positions into seconds.
peak_times = r_peaks / fs

# Difference between consecutive R-peak times.
rr_intervals = np.diff(peak_times)

# Assign each RR interval to the midpoint between its beats.
rr_times = (peak_times[:-1] + peak_times[1:]) / 2.0

# Save the original beat-to-beat measurements.
beat_df = pd.DataFrame({
    "time_s": rr_times,
    "rr_interval_s": rr_intervals
})

beat_output = OUTPUT_DIR / "a01_edr_rr_beats.csv"
beat_df.to_csv(beat_output, index=False)

print(f"RR intervals calculated: {len(rr_intervals)}")


# --------------------------------------------------
# 5. INTERPOLATE ONTO A UNIFORM TIME GRID
# --------------------------------------------------

# RR intervals occur at irregular beat times.
# Create a uniform time grid sampled at 4 Hz.

dt = 1.0 / EDR_FS

start_time = np.ceil(rr_times[0] / dt) * dt
end_time = np.floor(rr_times[-1] / dt) * dt

uniform_times = np.arange(
    start_time,
    end_time + dt / 2.0,
    dt
)

# Shape-preserving interpolation avoids some overshoot
# that can occur with ordinary cubic spline interpolation.
interpolator = PchipInterpolator(
    rr_times,
    rr_intervals,
    extrapolate=False
)

rr_uniform = interpolator(uniform_times)


# --------------------------------------------------
# 6. SAVE THE RR-DERIVED EDR SIGNAL
# --------------------------------------------------

uniform_df = pd.DataFrame({
    "time_s": uniform_times,
    "rr_edr_s": rr_uniform
})

uniform_output = OUTPUT_DIR / "a01_edr_rr_uniform.csv"
uniform_df.to_csv(uniform_output, index=False)


# --------------------------------------------------
# 7. DISPLAY SUMMARY
# --------------------------------------------------

print("\nRESULTS")
print("-" * 55)

print(f"Mean RR interval: {np.mean(rr_intervals):.4f} s")
print(f"Median RR interval: {np.median(rr_intervals):.4f} s")
print(f"Uniform EDR sampling rate: {EDR_FS:.1f} Hz")
print(f"Uniform EDR samples: {len(rr_uniform)}")
print(f"Duration: {uniform_times[-1] - uniform_times[0]:.2f} s")

print("\nSaved files:")
print(beat_output)
print(uniform_output)

print("\nRR-interval EDR generation completed.")
