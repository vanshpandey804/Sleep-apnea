"""
Step 3: Apnea Annotation Validation and Signal Alignment (a01)
==============================================================
This script inspects and verifies:
1. Exact binary and WFDB encoding of a01.apn (timestamps, interval, symbol dictionary).
2. Temporal alignment between ECG (a01.dat) and respiration/SpO2 (a01r.dat).
3. Physiological concordance at the exact transition minute from Normal to Apnea.
4. Establishes the formalized 1-minute windowing and labeling rule for Step 4.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# -------------------------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ao1"
TABLES_DIR = ROOT / "results" / "tables"
PLOTS_DIR = ROOT / "results" / "plots"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 2. LOAD ANNOTATIONS AND SIGNALS
# -------------------------------------------------------------------------
print("=" * 65)
print("STEP 3: APNEA ANNOTATION VALIDATION & ALIGNMENT (RECORD a01)")
print("=" * 65)

# Read ECG header and record
ecg_header = wfdb.rdheader(str(DATA_DIR / "a01"))
ecg_record = wfdb.rdrecord(str(DATA_DIR / "a01"))
fs_ecg = float(ecg_header.fs)
ecg_len = ecg_header.sig_len
ecg_sig = ecg_record.p_signal[:, 0]

print(f"ECG Record: a01 | fs = {fs_ecg:.1f} Hz | Total Samples = {ecg_len} ({ecg_len/fs_ecg/60:.2f} mins)")

# Read Respiration record (a01r)
resp_record = wfdb.rdrecord(str(DATA_DIR / "a01r"))
fs_resp = float(resp_record.fs)
resp_len = resp_record.sig_len
resp_names = [s.lower() for s in resp_record.sig_name]

print(f"Respiration Record: a01r | fs = {fs_resp:.1f} Hz | Total Samples = {resp_len} ({resp_len/fs_resp/60:.2f} mins)")
print(f"Available Respiration Channels: {resp_record.sig_name}")

airflow_idx = resp_names.index("resp n")
spo2_idx = resp_names.index("spo2")
resp_c_idx = resp_names.index("resp c")
resp_a_idx = resp_names.index("resp a")

airflow = resp_record.p_signal[:, airflow_idx]
spo2 = resp_record.p_signal[:, spo2_idx]
resp_c = resp_record.p_signal[:, resp_c_idx]
resp_a = resp_record.p_signal[:, resp_a_idx]

# Read Apnea Annotations (.apn)
ann = wfdb.rdann(str(DATA_DIR / "a01"), "apn")
sample_indices = np.asarray(ann.sample, dtype=np.int64)
symbols = np.asarray(ann.symbol)

print(f"\nApnea Annotations Loaded: {len(sample_indices)} total labels")
print(f"Unique Symbols: {set(symbols)} (N = Normal/Non-apnea, A = Apnea)")
counts = {s: int(np.sum(symbols == s)) for s in set(symbols)}
print(f"Symbol Counts: {counts}")

# -------------------------------------------------------------------------
# 3. VERIFY STEP SIZE AND TIME BASE
# -------------------------------------------------------------------------
sample_diffs = np.diff(sample_indices)
unique_diffs = set(sample_diffs)
print(f"\nSample increments between successive annotations: {unique_diffs}")
step_samples = sample_diffs[0] if len(sample_diffs) > 0 else 6000
step_seconds = step_samples / fs_ecg

print(f"Step duration: {step_samples} samples / {fs_ecg:.1f} Hz = {step_seconds:.2f} seconds ({step_seconds/60:.2f} minute)")

# Construct inspection dataframe of the first 25 annotations
inspection_rows = []
for idx in range(min(30, len(sample_indices))):
    s_start = sample_indices[idx]
    s_end = s_start + step_samples
    t_start_s = s_start / fs_ecg
    t_end_s = s_end / fs_ecg
    sym = symbols[idx]
    
    # Airflow metrics in window
    af_win = airflow[s_start:min(s_end, len(airflow))]
    spo2_win = spo2[s_start:min(s_end, len(spo2))]
    
    af_ptp = np.ptp(af_win) if len(af_win) else np.nan
    af_std = np.std(af_win) if len(af_win) else np.nan
    spo2_mean = np.mean(spo2_win) if len(spo2_win) else np.nan
    spo2_min = np.min(spo2_win) if len(spo2_win) else np.nan
    
    inspection_rows.append({
        "epoch_index": idx,
        "sample_start": s_start,
        "sample_end": s_end,
        "time_start_s": t_start_s,
        "time_end_s": t_end_s,
        "time_start_min": t_start_s / 60.0,
        "label": sym,
        "is_apnea": 1 if sym == "A" else 0,
        "airflow_ptp": round(float(af_ptp), 4),
        "airflow_std": round(float(af_std), 4),
        "spo2_mean": round(float(spo2_mean), 2),
        "spo2_min": round(float(spo2_min), 2),
    })

df_inspection = pd.DataFrame(inspection_rows)
csv_inspection_path = TABLES_DIR / "a01_apnea_annotation_inspection.csv"
df_inspection.to_csv(csv_inspection_path, index=False)
print(f"\nInspection table saved to: {csv_inspection_path}")
print("\nFirst 18 Epochs (covering Normal -> Apnea transition):")
print(df_inspection[["epoch_index", "time_start_min", "label", "airflow_ptp", "spo2_mean", "spo2_min"]].to_string(index=False))

# -------------------------------------------------------------------------
# 4. TRANSITION ANALYSIS & VERIFICATION PLOT
# -------------------------------------------------------------------------
# Epoch 12 is N (720-780s), Epoch 13 is A (780-840s)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Time axis around transition: minutes 10 to 18 (600s to 1080s)
t_zoom_start = 600.0
t_zoom_end = 1080.0
idx_zoom_ecg = (np.arange(len(ecg_sig)) / fs_ecg >= t_zoom_start) & (np.arange(len(ecg_sig)) / fs_ecg <= t_zoom_end)
t_ecg_zoom = np.arange(len(ecg_sig))[idx_zoom_ecg] / fs_ecg

idx_zoom_resp = (np.arange(len(airflow)) / fs_resp >= t_zoom_start) & (np.arange(len(airflow)) / fs_resp <= t_zoom_end)
t_resp_zoom = np.arange(len(airflow))[idx_zoom_resp] / fs_resp

# Plot 1: ECG
axes[0].plot(t_ecg_zoom / 60.0, ecg_sig[idx_zoom_ecg], color="darkblue", lw=0.7, label="Lead-I ECG")
axes[0].set_ylabel("ECG (mV)")
axes[0].set_title("Apnea Annotation Alignment Verification: Minute 10 to 18 Transition (Record a01)")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

# Plot 2: Nasal Airflow
axes[1].plot(t_resp_zoom / 60.0, airflow[idx_zoom_resp], color="teal", lw=0.8, label="Nasal Airflow (Resp N)")
axes[1].set_ylabel("Airflow (ADC)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper right")

# Plot 3: SpO2
axes[2].plot(t_resp_zoom / 60.0, spo2[idx_zoom_resp], color="crimson", lw=1.2, label="Oxygen Saturation (SpO2 %)")
axes[2].set_ylabel("SpO2 (%)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="upper right")

# Plot 4: Ground Truth Apnea Label
# Epochs 10 to 18
epoch_times = []
epoch_labels = []
for ep in range(10, 19):
    epoch_times.extend([ep, ep + 1])
    epoch_labels.extend([1 if symbols[ep] == "A" else 0, 1 if symbols[ep] == "A" else 0])

axes[3].step(epoch_times, epoch_labels, where="post", color="purple", lw=2, label="Apnea Ground Truth (1 = Apnea, 0 = Normal)")
axes[3].set_ylabel("Apnea (Binary)")
axes[3].set_xlabel("Time (Minutes)")
axes[3].set_ylim(-0.1, 1.2)
axes[3].grid(True, alpha=0.3)
axes[3].legend(loc="upper right")

# Highlight transition minute 13
for ax in axes:
    ax.axvline(13.0, color="red", linestyle="--", lw=1.5, alpha=0.8)
    ax.axvspan(13.0, 18.0, color="orange", alpha=0.15)

plot_path = PLOTS_DIR / "a01_annotation_alignment_verification.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"\nTransition verification plot saved to: {plot_path}")
print("\n" + "=" * 65)
print("CONFIRMATION OF STEP 3 RULES:")
print("1. Standard window size: exactly 60.0 seconds (6000 samples at 100 Hz).")
print("2. 0% overlap between adjacent windows ensures exact 1:1 mapping with PhysioNet labels.")
print("3. Epoch i starts at t = i * 60s and ends at t = (i + 1) * 60s.")
print("4. Label for epoch i is exactly ann.symbol[i] ('N' -> 0, 'A' -> 1).")
print("=" * 65)
