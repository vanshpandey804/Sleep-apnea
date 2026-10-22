
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, welch


# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ao1"
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"

PLOTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

FS = 4.0
WINDOW_SECONDS = 60
STEP_SECONDS = 30
MAX_LAG_SECONDS = 5

rr_path = TABLES / "a01_edr_rr_uniform.csv"
morph_path = TABLES / "a01_edr_morphology_uniform.csv"


# ==========================================================
# 2. LOAD EDR SIGNALS AND MEASURED AIRFLOW
# ==========================================================

rr = pd.read_csv(rr_path)
morph = pd.read_csv(morph_path)

resp_record = wfdb.rdrecord(str(DATA / "a01r"))
names = [name.lower() for name in resp_record.sig_name]

if "resp n" not in names:
    raise ValueError(
        f"Nasal airflow channel 'Resp N' not found. "
        f"Available channels: {resp_record.sig_name}"
    )

airflow_index = names.index("resp n")
airflow = resp_record.p_signal[:, airflow_index].astype(float)
airflow_fs = float(resp_record.fs)

# Common time grid: use only times present in both EDR files
# and within the measured respiratory recording.
start = max(rr.time_s.min(), morph.time_s.min(), 0.0)
end = min(
    rr.time_s.max(),
    morph.time_s.max(),
    (len(airflow) - 1) / airflow_fs
)

times = np.arange(
    np.ceil(start * FS) / FS,
    end,
    1.0 / FS
)

rr_signal = np.interp(
    times, rr.time_s, rr.rr_edr_s
)

resp_features = {}
for column in ["r_amplitude", "qrs_peak_to_peak", "qrs_area"]:
    resp_features[column] = np.interp(
        times, morph.time_s, morph[column]
    )

airflow_times = np.arange(len(airflow)) / airflow_fs
airflow_uniform = np.interp(times, airflow_times, airflow)

print("=" * 60)
print("EDR VALIDATION")
print("=" * 60)
print(f"Common duration: {(end - start) / 60:.1f} minutes")
print(f"Comparison sampling rate: {FS:.1f} Hz")
print(f"Reference channel: {resp_record.sig_name[airflow_index]}")


# ==========================================================
# 3. FILTER TO THE RESPIRATORY FREQUENCY BAND
# ==========================================================

# Approximate 6–42 breaths/minute.
sos = butter(
    3, [0.1, 0.7],
    btype="bandpass",
    fs=FS,
    output="sos"
)


def respiratory_filter(signal):
    signal = np.asarray(signal, dtype=float)

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains missing or infinite values.")

    return sosfiltfilt(sos, signal - np.mean(signal))


reference = respiratory_filter(airflow_uniform)
signals = {
    "RR interval": respiratory_filter(rr_signal),
    "R amplitude": respiratory_filter(resp_features["r_amplitude"]),
    "QRS peak-to-peak": respiratory_filter(
        resp_features["qrs_peak_to_peak"]
    ),
    "QRS area": respiratory_filter(resp_features["qrs_area"]),
}


# ==========================================================
# 4. WINDOWED CORRELATION AND RESPIRATORY-RATE ERROR
# ==========================================================

window = int(WINDOW_SECONDS * FS)
step = int(STEP_SECONDS * FS)
max_lag = int(MAX_LAG_SECONDS * FS)

# Minimum reference variation avoids evaluating nearly flat
# airflow segments as if they were reliable breathing signals.
airflow_std_min = 1e-8

rows = []

for begin in range(0, len(times) - window + 1, step):
    finish = begin + window

    ref = reference[begin:finish]

    if np.std(ref) < airflow_std_min:
        continue

    # Reference respiratory rate from Welch PSD.
    freq, power = welch(ref, fs=FS, nperseg=window)

    band = (freq >= 0.1) & (freq <= 0.7)
    if not np.any(band):
        continue

    ref_freq = freq[band][np.argmax(power[band])]
    ref_bpm = ref_freq * 60.0

    for method, full_signal in signals.items():
        candidate = full_signal[begin:finish]

        if np.std(candidate) < 1e-10:
            best_corr = np.nan
            estimated_bpm = np.nan
            lag_seconds = np.nan
        else:
            # Search small positive and negative lags.
            best_corr = -1.0
            best_lag = 0

            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    a, b = ref[-lag:], candidate[:lag]
                elif lag > 0:
                    a, b = ref[:-lag], candidate[lag:]
                else:
                    a, b = ref, candidate

                if len(a) < 10 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    continue

                corr = np.corrcoef(a, b)[0, 1]

                # Absolute correlation allows inverted polarity.
                if np.isfinite(corr) and abs(corr) > best_corr:
                    best_corr = abs(corr)
                    best_lag = lag

            lag_seconds = best_lag / FS if best_corr >= 0 else np.nan

            f, p = welch(candidate, fs=FS, nperseg=window)
            valid_band = (f >= 0.1) & (f <= 0.7)

            if np.any(valid_band):
                estimated_freq = f[valid_band][
                    np.argmax(p[valid_band])
                ]
                estimated_bpm = estimated_freq * 60.0
            else:
                estimated_bpm = np.nan

        rows.append({
            "time_s": times[begin + window // 2],
            "method": method,
            "reference_rate_bpm": ref_bpm,
            "estimated_rate_bpm": estimated_bpm,
            "rate_abs_error_bpm": (
                abs(estimated_bpm - ref_bpm)
                if np.isfinite(estimated_bpm) else np.nan
            ),
            "best_abs_correlation": (
                best_corr if best_corr >= 0 else np.nan
            ),
            "lag_s": lag_seconds,
        })


# ==========================================================
# 5. SAVE WINDOW RESULTS AND SUMMARY
# ==========================================================

results = pd.DataFrame(rows)

if results.empty:
    raise RuntimeError(
        "No usable windows were evaluated. Check airflow values "
        "and signal units."
    )

window_path = TABLES / "a01_edr_validation_windows.csv"
results.to_csv(window_path, index=False)

summary = results.groupby("method").agg(
    evaluated_windows=("time_s", "count"),
    median_abs_correlation=("best_abs_correlation", "median"),
    median_rate_error_bpm=("rate_abs_error_bpm", "median"),
    mean_rate_error_bpm=("rate_abs_error_bpm", "mean"),
).reset_index()

summary_path = TABLES / "a01_edr_validation_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSUMMARY")
print("-" * 60)
print(summary.to_string(index=False))
print(f"\nWindow results: {window_path}")
print(f"Summary results: {summary_path}")


# ==========================================================
# 6. PLOT THE COMPARISON
# ==========================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for method, group in results.groupby("method"):
    axes[0].plot(
        group["time_s"] / 3600,
        group["best_abs_correlation"],
        label=method,
        alpha=0.8
    )

    axes[1].plot(
        group["time_s"] / 3600,
        group["rate_abs_error_bpm"],
        label=method,
        alpha=0.8
    )

axes[0].set_title("EDR Agreement with Nasal Airflow")
axes[0].set_ylabel("Best absolute correlation")
axes[0].set_ylim(0, 1)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Respiratory-Rate Estimation Error")
axes[1].set_xlabel("Recording time (hours)")
axes[1].set_ylabel("Absolute error (breaths/min)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plot_path = PLOTS / "a01_edr_validation.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Plot: {plot_path}")
print("\nEDR validation completed.")
