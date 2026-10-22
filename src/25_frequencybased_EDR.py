
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.interpolate import interp1d
import wfdb


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ao1"
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"

TABLES.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

FS_EDR = 4.0
WINDOW_SEC = 60
STEP_SEC = 30

# Plausible respiratory frequency range:
# 6 to 42 breaths/minute
RESP_LOW = 0.10
RESP_HIGH = 0.70


# --------------------------------------------------
# 2. Load beat-level R-amplitude EDR
# --------------------------------------------------

beats_path = TABLES / "a01_edr_morphology_beats.csv"

if not beats_path.exists():
    raise FileNotFoundError(f"Missing file: {beats_path}")

beats = pd.read_csv(beats_path)
print("Beat-level columns:", beats.columns.tolist())

# Identify time and R-amplitude columns robustly.
time_candidates = ["time_sec", "time_s", "time", "beat_time_sec"]
amp_candidates = ["r_amplitude", "R_amplitude"]

time_col = next(
    (c for c in time_candidates if c in beats.columns), None
)
amp_col = next(
    (c for c in amp_candidates if c in beats.columns), None
)

if time_col is None or amp_col is None:
    raise ValueError(
        "Could not identify beat time and R-amplitude columns. "
        f"Available columns: {beats.columns.tolist()}"
    )

beat_time = beats[time_col].to_numpy(dtype=float)
beat_amp = beats[amp_col].to_numpy(dtype=float)

valid = np.isfinite(beat_time) & np.isfinite(beat_amp)
beat_time = beat_time[valid]
beat_amp = beat_amp[valid]

order = np.argsort(beat_time)
beat_time = beat_time[order]
beat_amp = beat_amp[order]

# Remove duplicate timestamps, if any.
beat_time, unique_idx = np.unique(beat_time, return_index=True)
beat_amp = beat_amp[unique_idx]

# Interpolate the irregular beat series onto a uniform 4 Hz grid.
uniform_time = np.arange(
    beat_time[0], beat_time[-1], 1.0 / FS_EDR
)
interpolator = interp1d(
    beat_time,
    beat_amp,
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
)
edr = interpolator(uniform_time)

# Fill occasional edge/interpolation gaps safely.
valid_edr = np.isfinite(edr)
edr = np.interp(
    uniform_time,
    uniform_time[valid_edr],
    edr[valid_edr],
)

# Remove slow baseline and isolate respiratory frequencies.
sos = butter(
    3, [RESP_LOW, RESP_HIGH],
    btype="bandpass",
    fs=FS_EDR,
    output="sos",
)
edr_filtered = sosfiltfilt(sos, edr)


# --------------------------------------------------
# 3. Load measured nasal airflow reference
# --------------------------------------------------

resp_record = wfdb.rdrecord(str(DATA / "a01r"))
channel_names = list(resp_record.sig_name)

if "Resp N" not in channel_names:
    raise ValueError(
        f"'Resp N' not found. Available channels: {channel_names}"
    )

resp_idx = channel_names.index("Resp N")
resp = resp_record.p_signal[:, resp_idx]
resp_fs = float(resp_record.fs)

resp_time = np.arange(len(resp)) / resp_fs
resp_valid = np.isfinite(resp)

# Common timeline, sampled at 4 Hz.
common_end = min(uniform_time[-1], resp_time[-1])
common_time = np.arange(0, common_end, 1.0 / FS_EDR)

edr_common = np.interp(
    common_time, uniform_time, edr_filtered
)
resp_common = np.interp(
    common_time, resp_time[resp_valid], resp[resp_valid]
)

resp_common = sosfiltfilt(sos, resp_common)


# --------------------------------------------------
# 4. Estimate breathing rate using Welch PSD
# --------------------------------------------------

def dominant_breathing_rate(signal, fs):
    """Return dominant respiratory frequency in breaths/min."""
    signal = np.asarray(signal, dtype=float)

    if len(signal) < int(30 * fs):
        return np.nan

    if not np.all(np.isfinite(signal)):
        return np.nan

    signal = signal - np.mean(signal)

    frequencies, power = welch(
        signal,
        fs=fs,
        nperseg=min(len(signal), int(60 * fs)),
        detrend="linear",
    )

    mask = (
        (frequencies >= RESP_LOW)
        & (frequencies <= RESP_HIGH)
    )

    if not np.any(mask):
        return np.nan

    peak_frequency = frequencies[mask][
        np.argmax(power[mask])
    ]

    return float(peak_frequency * 60.0)


# --------------------------------------------------
# 5. Evaluate in matching 60-second windows
# --------------------------------------------------

window_samples = int(WINDOW_SEC * FS_EDR)
step_samples = int(STEP_SEC * FS_EDR)

rows = []

for start in range(
    0,
    len(common_time) - window_samples + 1,
    step_samples,
):
    end = start + window_samples

    edr_segment = edr_common[start:end]
    resp_segment = resp_common[start:end]

    edr_rate = dominant_breathing_rate(
        edr_segment, FS_EDR
    )
    resp_rate = dominant_breathing_rate(
        resp_segment, FS_EDR
    )

    if not np.isfinite(edr_rate) or not np.isfinite(resp_rate):
        continue

    rows.append({
        "start_time_sec": common_time[start],
        "edr_rate_bpm": edr_rate,
        "airflow_rate_bpm": resp_rate,
        "absolute_rate_error_bpm": abs(edr_rate - resp_rate),
    })

results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No valid windows were evaluated. Check signal files and timing."
    )

out_csv = TABLES / "a01_frequency_EDR_validation.csv"
results.to_csv(out_csv, index=False)

# --------------------------------------------------
# 6. Report and plot
# --------------------------------------------------

print("\nFREQUENCY-DOMAIN EDR RESULTS")
print("=" * 45)
print("Evaluated windows:", len(results))
print(
    "Median absolute rate error:",
    round(results["absolute_rate_error_bpm"].median(), 3),
    "breaths/min",
)
print(
    "Mean absolute rate error:",
    round(results["absolute_rate_error_bpm"].mean(), 3),
    "breaths/min",
)
print("Saved:", out_csv)

plt.figure(figsize=(12, 5))
plt.plot(
    results["start_time_sec"] / 60,
    results["airflow_rate_bpm"],
    label="Nasal airflow reference",
    linewidth=1,
)
plt.plot(
    results["start_time_sec"] / 60,
    results["edr_rate_bpm"],
    label="Frequency-domain EDR",
    linewidth=1,
)
plt.xlabel("Time (minutes)")
plt.ylabel("Estimated respiratory rate (breaths/min)")
plt.title("Frequency-domain EDR vs nasal airflow — a01")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_plot = PLOTS / "a01_frequency_EDR_validation.png"
plt.savefig(out_plot, dpi=200)
plt.close()

print("Saved plot:", out_plot)
