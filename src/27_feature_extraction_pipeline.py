"""
Step 4: Unified Feature Extraction Pipeline (5 EDR Modalities + HRV)
===================================================================
Processes overnight ECG recordings:
1. Applies 0.5-40 Hz zero-phase Butterworth bandpass filter.
2. Detects QRS/R-peaks using validated Detector 4 (Pan-Tompkins adaptive).
3. Computes all 5 EDR representations:
   - EDR 1: RR-Interval / HRV (RSA)
   - EDR 2: R-wave amplitude
   - EDR 3: QRS peak-to-peak amplitude
   - EDR 4: QRS area
   - EDR 5: Frequency-based Welch PSD respiratory rate and power
4. Partitions signal into 1-minute non-overlapping epochs aligned with .apn annotations.
5. Computes patient-specific baseline normalization (Constraint 3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from scipy.interpolate import PchipInterpolator

# -------------------------------------------------------------------------
# CONSTANTS AND PARAMETERS
# -------------------------------------------------------------------------
FS_EDR = 4.0               # Uniform resampling rate for EDR signals
WINDOW_SEC = 60.0          # Non-overlapping window duration
RESP_LOW = 0.10            # 6 breaths/min
RESP_HIGH = 0.70           # 42 breaths/min
QRS_HALF_WINDOW = 0.080    # 80 ms around R-peak
BASELINE_START = 0.120     # 120 ms
BASELINE_END = 0.200       # 200 ms


def bandpass_filter_ecg(ecg, fs):
    """0.5 - 40 Hz 4th-order zero-phase Butterworth bandpass filter."""
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, ecg)


def detect_r_peaks_detector4(filtered_ecg, fs):
    """
    Detector 4: Pan-Tompkins with moving integration, dual adaptive threshold,
    refractory period, search-back, and T-wave discrimination.
    """
    # 1. Derivative and squaring
    differentiated = np.diff(filtered_ecg, prepend=filtered_ecg[0])
    squared = differentiated ** 2

    # 2. Moving-window integration (150 ms)
    integration_samples = max(1, int(round(0.150 * fs)))
    window = np.ones(integration_samples) / integration_samples
    integrated = np.convolve(squared, window, mode="same")

    # Robust outlier protection against electrode detachment spikes
    cap_val = float(np.percentile(integrated, 99.5)) * 3.0
    integrated = np.clip(integrated, 0, cap_val)

    # 3. Local maxima in integrated signal
    local_peaks, _ = find_peaks(integrated)
    if len(local_peaks) < 3:
        raise RuntimeError("Too few local maxima in integrated signal.")

    # 4. Robust initialization using samples 2s - 12s
    init_start = int(round(1.0 * fs))
    init_end = min(len(integrated), int(round(11.0 * fs)))
    initial_peaks = local_peaks[(local_peaks >= init_start) & (local_peaks < init_end)]

    if len(initial_peaks) < 4:
        initial_peaks = local_peaks[:min(50, len(local_peaks))]

    initial_values = integrated[initial_peaks]
    noise_level = float(np.median(initial_values))
    signal_level = float(np.percentile(initial_values, 75))
    if signal_level <= noise_level:
        signal_level = float(np.max(initial_values)) + 1e-6

    threshold = noise_level + 0.25 * (signal_level - noise_level)

    refractory_period = max(1, int(round(0.250 * fs)))
    t_wave_window = max(1, int(round(0.360 * fs)))
    searchback_factor = 1.66

    accepted_qrs = []
    t_wave_rejected = set()
    threshold_at_peak = np.zeros(len(integrated), dtype=float)

    last_qrs = None
    last_qrs_slope = None

    def local_slope(idx):
        start = max(0, idx - 5)
        end = min(len(differentiated), idx + 6)
        return float(np.max(np.abs(differentiated[start:end]))) if end > start else 0.0

    # 5. Adaptive detection pass
    for peak in local_peaks:
        peak_value = float(integrated[peak])
        threshold_at_peak[peak] = threshold

        if peak_value > threshold:
            if last_qrs is not None and (peak - last_qrs) < refractory_period:
                if peak_value > integrated[last_qrs]:
                    accepted_qrs[-1] = int(peak)
                    last_qrs = int(peak)
                    last_qrs_slope = local_slope(peak)
                    signal_level = 0.875 * signal_level + 0.125 * min(peak_value, 2.5 * signal_level)
                else:
                    noise_level = 0.875 * noise_level + 0.125 * peak_value
            else:
                is_t_wave = False
                current_slope = local_slope(peak)
                if last_qrs is not None and last_qrs_slope is not None:
                    if (peak - last_qrs) < t_wave_window and current_slope < 0.5 * last_qrs_slope:
                        is_t_wave = True

                if is_t_wave:
                    t_wave_rejected.add(int(peak))
                    noise_level = 0.875 * noise_level + 0.125 * peak_value
                else:
                    accepted_qrs.append(int(peak))
                    last_qrs = int(peak)
                    last_qrs_slope = current_slope
                    signal_level = 0.875 * signal_level + 0.125 * min(peak_value, 2.5 * signal_level)
        else:
            noise_level = 0.875 * noise_level + 0.125 * peak_value

        threshold = noise_level + 0.25 * (signal_level - noise_level)

    accepted_qrs = np.array(sorted(set(accepted_qrs)), dtype=int)

    # 6. Search-back pass
    searchback_peaks = []
    for i in range(1, len(accepted_qrs)):
        prev = int(accepted_qrs[i - 1])
        curr = int(accepted_qrs[i])
        rr = curr - prev
        prior_rrs = np.diff(accepted_qrs[:i])
        if len(prior_rrs) < 2:
            continue
        avg_rr = float(np.mean(prior_rrs[-8:]))
        if avg_rr > 0 and rr > searchback_factor * avg_rr:
            s_start = prev + refractory_period
            s_end = curr - refractory_period
            if s_end > s_start:
                possible = local_peaks[(local_peaks > s_start) & (local_peaks < s_end)]
                eligible = [p for p in possible if p not in t_wave_rejected and threshold_at_peak[p] > 0
                            and integrated[p] <= threshold_at_peak[p] and integrated[p] > 0.5 * threshold_at_peak[p]]
                if eligible:
                    best_peak = max(eligible, key=lambda p: integrated[p])
                    searchback_peaks.append(best_peak)

    accepted_qrs = np.array(sorted(set(accepted_qrs).union(searchback_peaks)), dtype=int)

    # 7. Localization in filtered ECG (search 80 ms around candidate)
    search_samples = max(1, int(round(0.080 * fs)))
    raw_r_peaks = []
    for candidate in accepted_qrs:
        st = max(0, int(candidate) - search_samples)
        en = min(len(filtered_ecg), int(candidate) + search_samples + 1)
        if en > st:
            raw_r_peaks.append(st + int(np.argmax(filtered_ecg[st:en])))

    raw_r_peaks = np.array(sorted(set(raw_r_peaks)), dtype=int)

    # Re-enforce refractory interval
    r_peaks = []
    for p in raw_r_peaks:
        if not r_peaks:
            r_peaks.append(int(p))
        elif p - r_peaks[-1] >= refractory_period:
            r_peaks.append(int(p))
        elif filtered_ecg[p] > filtered_ecg[r_peaks[-1]]:
            r_peaks[-1] = int(p)

    return np.array(r_peaks, dtype=np.int64)


def extract_morphology_and_edr(filtered_ecg, r_peaks, fs):
    """
    Extracts beat-level and 4Hz interpolated:
    - RR intervals
    - R-wave amplitudes
    - QRS peak-to-peak amplitudes
    - QRS areas
    """
    qrs_half = int(round(QRS_HALF_WINDOW * fs))
    baseline_inner = int(round(BASELINE_START * fs))
    baseline_outer = int(round(BASELINE_END * fs))
    margin = baseline_outer + 2

    # Filter out boundary peaks
    valid_mask = (r_peaks >= margin) & (r_peaks < len(filtered_ecg) - margin)
    valid_peaks = r_peaks[valid_mask]
    peak_times = valid_peaks / fs

    r_amplitudes = []
    qrs_ptps = []
    qrs_areas = []

    for peak in valid_peaks:
        left_bl = filtered_ecg[peak - baseline_outer : peak - baseline_inner]
        right_bl = filtered_ecg[peak + baseline_inner : peak + baseline_outer]
        baseline = np.median(np.concatenate([left_bl, right_bl]))

        qrs = filtered_ecg[peak - qrs_half : peak + qrs_half + 1]
        qrs_corr = qrs - baseline

        r_amp = filtered_ecg[peak] - baseline
        ptp = np.max(qrs_corr) - np.min(qrs_corr)
        area = np.sum(np.abs(qrs_corr)) / fs

        r_amplitudes.append(r_amp)
        qrs_ptps.append(ptp)
        qrs_areas.append(area)

    r_amplitudes = np.array(r_amplitudes)
    qrs_ptps = np.array(qrs_ptps)
    qrs_areas = np.array(qrs_areas)

    # RR intervals
    rr_intervals = np.diff(peak_times)
    rr_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    # Uniform 4 Hz interpolation for all continuous EDR signals
    dt = 1.0 / FS_EDR
    start_time = np.ceil(max(peak_times[0], rr_times[0]) / dt) * dt
    end_time = np.floor(min(peak_times[-1], rr_times[-1]) / dt) * dt
    uniform_times = np.arange(start_time, end_time + dt / 2.0, dt)

    # Shape-preserving Pchip interpolation
    rr_pchip = PchipInterpolator(rr_times, rr_intervals, extrapolate=False)(uniform_times)
    r_amp_pchip = PchipInterpolator(peak_times, r_amplitudes, extrapolate=False)(uniform_times)
    ptp_pchip = PchipInterpolator(peak_times, qrs_ptps, extrapolate=False)(uniform_times)
    area_pchip = PchipInterpolator(peak_times, qrs_areas, extrapolate=False)(uniform_times)

    return {
        "valid_peaks": valid_peaks,
        "peak_times": peak_times,
        "rr_times": rr_times,
        "rr_intervals": rr_intervals,
        "r_amplitudes": r_amplitudes,
        "qrs_ptps": qrs_ptps,
        "qrs_areas": qrs_areas,
        "uniform_times": uniform_times,
        "edr_rr": rr_pchip,
        "edr_r_amp": r_amp_pchip,
        "edr_ptp": ptp_pchip,
        "edr_area": area_pchip,
    }


def compute_epoch_features(epoch_start_s, epoch_end_s, edr_dict):
    """
    Extracts window-level statistical and spectral features for a single 60s epoch
    across all 5 EDR representations.
    """
    t_u = edr_dict["uniform_times"]
    mask_u = (t_u >= epoch_start_s) & (t_u < epoch_end_s)

    # 1. RR / HRV features (from irregular beats falling inside epoch)
    rr_t = edr_dict["rr_times"]
    mask_rr = (rr_t >= epoch_start_s) & (rr_t < epoch_end_s)
    rr_in_epoch = edr_dict["rr_intervals"][mask_rr]

    if len(rr_in_epoch) >= 4:
        rr_mean = float(np.mean(rr_in_epoch))
        rr_std = float(np.std(rr_in_epoch))
        rr_median = float(np.median(rr_in_epoch))
        rr_iqr = float(np.percentile(rr_in_epoch, 75) - np.percentile(rr_in_epoch, 25))
        rr_min = float(np.min(rr_in_epoch))
        rr_max = float(np.max(rr_in_epoch))
        rr_rmssd = float(np.sqrt(np.mean(np.diff(rr_in_epoch) ** 2))) if len(rr_in_epoch) > 1 else 0.0
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
        hr_mean = 60.0 / rr_mean if rr_mean > 0 else 0.0
        hr_std = float(np.std(60.0 / rr_in_epoch))
    else:
        rr_mean = rr_std = rr_median = rr_iqr = rr_min = rr_max = rr_rmssd = rr_cv = hr_mean = hr_std = np.nan

    # 2. R-wave amplitude morphology features
    pk_t = edr_dict["peak_times"]
    mask_pk = (pk_t >= epoch_start_s) & (pk_t < epoch_end_s)
    r_amp_in = edr_dict["r_amplitudes"][mask_pk]
    ptp_in = edr_dict["qrs_ptps"][mask_pk]
    area_in = edr_dict["qrs_areas"][mask_pk]

    if len(r_amp_in) >= 4:
        r_amp_mean = float(np.mean(r_amp_in))
        r_amp_std = float(np.std(r_amp_in))
        r_amp_median = float(np.median(r_amp_in))
        r_amp_iqr = float(np.percentile(r_amp_in, 75) - np.percentile(r_amp_in, 25))
        r_amp_ptp = float(np.ptp(r_amp_in))

        qrs_ptp_mean = float(np.mean(ptp_in))
        qrs_ptp_std = float(np.std(ptp_in))
        qrs_ptp_median = float(np.median(ptp_in))
        qrs_ptp_iqr = float(np.percentile(ptp_in, 75) - np.percentile(ptp_in, 25))
        qrs_ptp_range = float(np.ptp(ptp_in))

        qrs_area_mean = float(np.mean(area_in))
        qrs_area_std = float(np.std(area_in))
        qrs_area_median = float(np.median(area_in))
        qrs_area_iqr = float(np.percentile(area_in, 75) - np.percentile(area_in, 25))
        qrs_area_range = float(np.ptp(area_in))
    else:
        r_amp_mean = r_amp_std = r_amp_median = r_amp_iqr = r_amp_ptp = np.nan
        qrs_ptp_mean = qrs_ptp_std = qrs_ptp_median = qrs_ptp_iqr = qrs_ptp_range = np.nan
        qrs_area_mean = qrs_area_std = qrs_area_median = qrs_area_iqr = qrs_area_range = np.nan

    # 3. Frequency-based Welch PSD respiratory rate and power (EDR 5)
    # Applied to filtered R-amplitude uniform signal in window
    if np.sum(mask_u) >= int(30 * FS_EDR):
        sig_edr = edr_dict["edr_r_amp"][mask_u]
        sig_edr_detrend = sig_edr - np.mean(sig_edr)
        nperseg = min(len(sig_edr_detrend), int(WINDOW_SEC * FS_EDR))
        freqs, psd = welch(sig_edr_detrend, fs=FS_EDR, nperseg=nperseg)
        resp_band = (freqs >= RESP_LOW) & (freqs <= RESP_HIGH)

        if np.any(resp_band):
            peak_f = freqs[resp_band][np.argmax(psd[resp_band])]
            resp_rate_bpm = float(peak_f * 60.0)
            resp_power = float(np.trapezoid(psd[resp_band], freqs[resp_band])) if hasattr(np, 'trapezoid') else float(np.trapz(psd[resp_band], freqs[resp_band]))
            total_power = float(np.trapezoid(psd, freqs)) if hasattr(np, 'trapezoid') else float(np.trapz(psd, freqs))
            resp_power_ratio = resp_power / (total_power + 1e-12)
        else:
            resp_rate_bpm = resp_power = resp_power_ratio = np.nan
    else:
        resp_rate_bpm = resp_power = resp_power_ratio = np.nan

    return {
        # EDR 1: RR / HRV Features
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_median": rr_median,
        "rr_iqr": rr_iqr,
        "rr_min": rr_min,
        "rr_max": rr_max,
        "rr_rmssd": rr_rmssd,
        "rr_cv": rr_cv,
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        # EDR 2: R-wave amplitude
        "r_amp_mean": r_amp_mean,
        "r_amp_std": r_amp_std,
        "r_amp_median": r_amp_median,
        "r_amp_iqr": r_amp_iqr,
        "r_amp_ptp": r_amp_ptp,
        # EDR 3: QRS peak-to-peak
        "qrs_ptp_mean": qrs_ptp_mean,
        "qrs_ptp_std": qrs_ptp_std,
        "qrs_ptp_median": qrs_ptp_median,
        "qrs_ptp_iqr": qrs_ptp_iqr,
        "qrs_ptp_range": qrs_ptp_range,
        # EDR 4: QRS area
        "qrs_area_mean": qrs_area_mean,
        "qrs_area_std": qrs_area_std,
        "qrs_area_median": qrs_area_median,
        "qrs_area_iqr": qrs_area_iqr,
        "qrs_area_range": qrs_area_range,
        # EDR 5: Welch PSD frequency features
        "resp_rate_welch_bpm": resp_rate_bpm,
        "resp_spectral_power": resp_power,
        "resp_power_ratio": resp_power_ratio,
    }


def process_patient_record(record_dir, record_name):
    """
    Full pipeline execution for a single patient record:
    Filtering -> Detector 4 -> 5 EDRs -> 60s Windowing -> Alignment with .apn
    """
    rec_path = str(record_dir / record_name)
    header = wfdb.rdheader(rec_path)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "apn")

    fs = float(header.fs)
    ecg = record.p_signal[:, 0].astype(float)
    duration_s = len(ecg) / fs

    print(f"[{record_name}] Filtering ECG ({len(ecg)} samples, {duration_s/60:.1f} mins)...")
    filtered_ecg = bandpass_filter_ecg(ecg, fs)

    print(f"[{record_name}] Running Detector 4 R-peak detection...")
    r_peaks = detect_r_peaks_detector4(filtered_ecg, fs)
    print(f"[{record_name}] Detected {len(r_peaks)} R-peaks.")

    print(f"[{record_name}] Extracting 5 EDR modalities...")
    edr_dict = extract_morphology_and_edr(filtered_ecg, r_peaks, fs)

    # Align with 1-minute ground-truth annotations
    sample_indices = np.asarray(ann.sample, dtype=np.int64)
    symbols = np.asarray(ann.symbol)

    rows = []
    for idx in range(len(sample_indices)):
        s_start = sample_indices[idx]
        t_start = s_start / fs
        t_end = t_start + WINDOW_SEC

        if t_end > duration_s:
            break

        sym = symbols[idx]
        label_binary = 1 if sym == "A" else 0

        feat = compute_epoch_features(t_start, t_end, edr_dict)
        feat["patient_id"] = record_name
        feat["window_idx"] = idx
        feat["start_time_s"] = t_start
        feat["end_time_s"] = t_end
        feat["label"] = sym
        feat["is_apnea"] = label_binary
        rows.append(feat)

    df_patient = pd.DataFrame(rows)
    print(f"[{record_name}] Extracted {len(df_patient)} labeled 1-minute feature rows.")

    # Apply Patient-Specific Baseline Normalization (Constraint 3)
    feature_cols = [c for c in df_patient.columns if c not in ["patient_id", "window_idx", "start_time_s", "end_time_s", "label", "is_apnea"]]
    for col in feature_cols:
        med = df_patient[col].median()
        iqr = df_patient[col].quantile(0.75) - df_patient[col].quantile(0.25)
        scale = iqr if (iqr is not None and iqr > 1e-8) else (df_patient[col].std() + 1e-8)
        df_patient[f"{col}_norm"] = (df_patient[col] - med) / scale

    return df_patient


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    data_dir = ROOT / "data" / "ao1"
    output_dir = ROOT / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_a01 = process_patient_record(data_dir, "a01")
    out_file = output_dir / "a01_feature_dataset.csv"
    df_a01.to_csv(out_file, index=False)
    print(f"Feature dataset for a01 saved to: {out_file}")
    print(f"Dataset shape: {df_a01.shape}")
    print("\nSample Columns:")
    print(df_a01.columns.tolist()[:15])
