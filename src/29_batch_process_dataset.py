"""
Step 4 & 5 Multi-Patient Batch Engine (with Incremental Caching)
===============================================================
1. Discovers all curated patient records in data/curated/ (a01-a20, b01-b05, c01-c10).
2. For each patient:
   - Checks if results/tables/patient_features/<id>.csv already exists (caching).
   - If not, runs 0.5-40 Hz filtering, Detector 4, and extracts all 5 EDR modalities.
   - Computes patient-specific baseline normalization.
   - Saves individual CSV.
3. Compiles the master dataset: results/tables/all_patients_edr_dataset.csv
4. Executes Step 5: Strict Patient-Independent GroupKFold ML Benchmark across
   all 5 individual EDRs + Combined.
"""

from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = ROOT / "data" / "curated"
TABLES_DIR = ROOT / "results" / "tables"
FEATURES_DIR = TABLES_DIR / "patient_features"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Helper to import numeric prefix scripts
def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

step4_module = import_from_path("step4_feature_pipeline", ROOT / "src" / "27_feature_extraction_pipeline.py")
step5_module = import_from_path("step5_model_training", ROOT / "src" / "28_train_evaluate_models.py")

process_patient_record = step4_module.process_patient_record
run_benchmark = step5_module.run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Process curated patients and run EDR benchmark.")
    parser.add_argument("--max_patients", type=int, default=None, help="Maximum number of patients to process.")
    parser.add_argument("--force_recompute", action="store_true", help="Recompute existing patient feature files.")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-PATIENT EDR FEATURE EXTRACTION & BENCHMARK")
    print("=" * 70)

    # Discover curated records
    record_dirs = [d for d in CURATED_DIR.iterdir() if d.is_dir() and (d / f"{d.name}.hea").exists()]
    record_dirs.sort(key=lambda d: d.name)

    if args.max_patients:
        record_dirs = record_dirs[:args.max_patients]

    print(f"Total curated patient records to process: {len(record_dirs)}")
    print(f"Patient IDs: {[d.name for d in record_dirs]}")

    all_dfs = []
    for idx, r_dir in enumerate(record_dirs, 1):
        rec_name = r_dir.name
        cached_file = FEATURES_DIR / f"{rec_name}_features.csv"

        if cached_file.exists() and not args.force_recompute:
            print(f"[{idx}/{len(record_dirs)}] Loading cached features for {rec_name}...")
            df_patient = pd.read_csv(cached_file)
        else:
            print(f"[{idx}/{len(record_dirs)}] Processing record: {rec_name}...")
            try:
                df_patient = process_patient_record(r_dir, rec_name)
                df_patient.to_csv(cached_file, index=False)
                print(f"  Saved: {cached_file}")
            except Exception as e:
                print(f"  Error processing {rec_name}: {e}")
                continue

        all_dfs.append(df_patient)

    if not all_dfs:
        print("No patient data processed.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_path = TABLES_DIR / "all_patients_edr_dataset.csv"
    master_df.to_csv(master_path, index=False)

    print("\n" + "=" * 70)
    print(f"MASTER DATASET GENERATED: {master_path}")
    print(f"Total Evaluated Patients: {master_df['patient_id'].nunique()}")
    print(f"Total 1-Minute Epochs: {len(master_df)}")
    apnea_cnt = int(np.sum(master_df['is_apnea'] == 1))
    normal_cnt = int(np.sum(master_df['is_apnea'] == 0))
    print(f"Apnea Epochs: {apnea_cnt} ({apnea_cnt/len(master_df)*100:.1f}%) | Normal Epochs: {normal_cnt} ({normal_cnt/len(master_df)*100:.1f}%)")
    print("=" * 70)

    # Execute Step 5: Patient-Independent GroupKFold ML Benchmark
    results_df = run_benchmark(master_df)
    return results_df


if __name__ == "__main__":
    main()
