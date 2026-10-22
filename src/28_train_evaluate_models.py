"""
Step 5: Patient-Independent Model Training and 5-EDR Benchmark
=============================================================
This module evaluates Machine Learning models for sleep apnea detection across:
1. All 5 individual EDR feature groups + Combined:
   - EDR 1: RR-Interval / HRV features alone
   - EDR 2: R-wave amplitude features alone
   - EDR 3: QRS peak-to-peak amplitude features alone
   - EDR 4: QRS area features alone
   - EDR 5: Welch frequency-based spectral features alone
   - Combined: All 5 EDR feature modalities
2. ML Classifiers:
   - Logistic Regression (with class weighting / regularization)
   - Random Forest Classifier (interpretable ensemble)
3. Evaluates:
   - Sensitivity (Recall)
   - Specificity
   - Precision
   - F1-Score
   - ROC-AUC
   - PR-AUC
4. Enforces:
   - Strict Patient-Independent evaluation (GroupKFold by patient_id)
   - Patient-specific baseline normalization (using normalized feature set)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# PATHS AND CONFIGURATION
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "results" / "tables"
PLOTS_DIR = ROOT / "results" / "plots"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# DEFINE 5 EDR FEATURE GROUPS
# -------------------------------------------------------------------------
FEATURE_GROUPS = {
    "EDR 1 (RR / HRV)": [
        "rr_mean_norm", "rr_std_norm", "rr_median_norm", "rr_iqr_norm",
        "rr_min_norm", "rr_max_norm", "rr_rmssd_norm", "rr_cv_norm",
        "hr_mean_norm", "hr_std_norm"
    ],
    "EDR 2 (R-Amplitude)": [
        "r_amp_mean_norm", "r_amp_std_norm", "r_amp_median_norm",
        "r_amp_iqr_norm", "r_amp_ptp_norm"
    ],
    "EDR 3 (QRS Peak-to-Peak)": [
        "qrs_ptp_mean_norm", "qrs_ptp_std_norm", "qrs_ptp_median_norm",
        "qrs_ptp_iqr_norm", "qrs_ptp_range_norm"
    ],
    "EDR 4 (QRS Area)": [
        "qrs_area_mean_norm", "qrs_area_std_norm", "qrs_area_median_norm",
        "qrs_area_iqr_norm", "qrs_area_range_norm"
    ],
    "EDR 5 (Frequency / Spectral)": [
        "resp_rate_welch_bpm_norm", "resp_spectral_power_norm",
        "resp_power_ratio_norm"
    ],
}

# Add Combined feature set
FEATURE_GROUPS["Combined (All 5 EDRs)"] = (
    FEATURE_GROUPS["EDR 1 (RR / HRV)"]
    + FEATURE_GROUPS["EDR 2 (R-Amplitude)"]
    + FEATURE_GROUPS["EDR 3 (QRS Peak-to-Peak)"]
    + FEATURE_GROUPS["EDR 4 (QRS Area)"]
    + FEATURE_GROUPS["EDR 5 (Frequency / Spectral)"]
)


def evaluate_predictions(y_true, y_pred, y_prob):
    """Computes all required performance metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan

    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def run_benchmark(df):
    """
    Evaluates Logistic Regression and Random Forest on every EDR feature set.
    Uses GroupKFold if multiple patients are present, or StratifiedKFold for single patient.
    """
    patients = df["patient_id"].unique()
    is_multi_patient = len(patients) > 1

    print("=" * 70)
    print("STEP 5: PATIENT EVALUATION BENCHMARK (5 EDRs + COMBINED)")
    print(f"Total Epochs: {len(df)} | Patients ({len(patients)}): {patients.tolist()[:10]}")
    print(f"Apnea Epochs: {int(np.sum(df['is_apnea'] == 1))} | Normal: {int(np.sum(df['is_apnea'] == 0))}")
    print(f"Evaluation Mode: {'Patient-Independent (GroupKFold)' if is_multi_patient else 'Cross-Validation (Stratified)'}")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    n_splits = min(5, len(patients)) if is_multi_patient else 5
    if is_multi_patient and n_splits >= 2:
        cv = GroupKFold(n_splits=n_splits)
        splitter = cv.split(df, df["is_apnea"], groups=df["patient_id"])
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splitter = cv.split(df, df["is_apnea"])

    # Pre-generate CV folds
    cv_folds = list(splitter)

    results_list = []

    for model_name, model_template in models.items():
        print(f"\n--- Model: {model_name} ---")

        for edr_name, feat_cols in FEATURE_GROUPS.items():
            # Check availability of columns
            available_cols = [c for c in feat_cols if c in df.columns]
            if len(available_cols) == 0:
                print(f"Skipping {edr_name} (columns not found)")
                continue

            X = df[available_cols].values
            y = df["is_apnea"].values

            all_y_true = []
            all_y_pred = []
            all_y_prob = []

            for train_idx, test_idx in cv_folds:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Impute NaNs and scale
                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()

                X_train_clean = scaler.fit_transform(imputer.fit_transform(X_train))
                X_test_clean = scaler.transform(imputer.transform(X_test))

                from sklearn.base import clone
                clf = clone(model_template)
                clf.fit(X_train_clean, y_train)

                y_pred = clf.predict(X_test_clean)
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test_clean)[:, 1]
                else:
                    y_prob = clf.decision_function(X_test_clean)

                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
                all_y_prob.extend(y_prob)

            metrics = evaluate_predictions(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob))
            metrics["Model"] = model_name
            metrics["EDR Approach"] = edr_name
            metrics["Num Features"] = len(available_cols)
            results_list.append(metrics)

            print(f"[{model_name}] {edr_name:28s} | Sens: {metrics['Sensitivity']:.3f} | Spec: {metrics['Specificity']:.3f} | F1: {metrics['F1-Score']:.3f} | ROC-AUC: {metrics['ROC-AUC']:.3f}")

    results_df = pd.DataFrame(results_list)
    results_path = TABLES_DIR / "edr_model_benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nAll benchmark results saved to: {results_path}")

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, m_name in enumerate(models.keys()):
        sub_df = results_df[results_df["Model"] == m_name]
        x_indices = np.arange(len(sub_df))
        bar_width = 0.22

        axes[i].bar(x_indices - bar_width, sub_df["Sensitivity"], width=bar_width, label="Sensitivity (Recall)", color="#2b5c8f")
        axes[i].bar(x_indices, sub_df["Specificity"], width=bar_width, label="Specificity", color="#5fa052")
        axes[i].bar(x_indices + bar_width, sub_df["F1-Score"], width=bar_width, label="F1-Score", color="#d95f02")

        axes[i].set_xticks(x_indices)
        axes[i].set_xticklabels([e.replace("EDR ", "E").replace(" (All 5 EDRs)", "") for e in sub_df["EDR Approach"]], rotation=25, ha="right", fontsize=9)
        axes[i].set_ylabel("Metric Value (0 - 1)")
        axes[i].set_title(f"Performance Across 5 EDRs — {m_name}")
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="lower right")

    plot_path = PLOTS_DIR / "edr_model_comparison_benchmark.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Benchmark comparison plot saved to: {plot_path}")

    return results_df


if __name__ == "__main__":
    dataset_file = TABLES_DIR / "a01_feature_dataset.csv"
    if not dataset_file.exists():
        print(f"Dataset file {dataset_file} not found. Run 27_feature_extraction_pipeline.py first.")
    else:
        df = pd.read_csv(dataset_file)
        run_benchmark(df)
