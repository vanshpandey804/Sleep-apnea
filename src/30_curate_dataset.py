"""
Dataset Curation and Organization Module
========================================
1. Parses additional-information.txt into a structured demographics table:
   results/tables/patient_demographics.csv
2. Organizes and validates the 35 complete learning records:
   data/curated/<record_id>/ (containing .hea, .dat, .apn, .qrs)
3. Organizes the 8 multi-signal respiration records:
   data/curated_respiration/<record_id>/ (containing .hea, .dat, .apn)
4. Organizes the 35 challenge test records:
   data/test_records/<record_id>/
"""

from pathlib import Path
import shutil
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "apnea-ecg"
CURATED_DIR = ROOT / "data" / "curated"
RESP_DIR = ROOT / "data" / "curated_respiration"
TEST_DIR = ROOT / "data" / "test_records"
TABLES_DIR = ROOT / "results" / "tables"

CURATED_DIR.mkdir(parents=True, exist_ok=True)
RESP_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def parse_demographics(info_file):
    """Parses additional-information.txt into a pandas DataFrame."""
    rows = []
    with open(info_file, "r") as f:
        lines = f.readlines()

    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Record") and "AHI" in line:
            header_idx = i
            break

    if header_idx == -1:
        print("Could not find table header in additional-information.txt")
        return pd.DataFrame()

    data_lines = lines[header_idx + 2 :]
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 12:
            rec = parts[0]
            try:
                length_min = float(parts[1])
                non_apn = float(parts[2])
                apn = float(parts[3])
                hours_apn = float(parts[4])
                ai = float(parts[5])
                hi = float(parts[6])
                ahi = float(parts[7])
                age = float(parts[8])
                sex = parts[9]
                height_cm = float(parts[10])
                weight_kg = float(parts[11])
                bmi = weight_kg / ((height_cm / 100.0) ** 2) if height_cm > 0 else np.nan

                group = rec[0].upper()

                rows.append({
                    "patient_id": rec,
                    "group": group,
                    "length_minutes": length_min,
                    "non_apnea_minutes": non_apn,
                    "apnea_minutes": apn,
                    "hours_with_apnea": hours_apn,
                    "AI": ai,
                    "HI": hi,
                    "AHI": ahi,
                    "age": age,
                    "sex": sex,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "BMI": round(bmi, 2),
                })
            except Exception as e:
                print(f"Skipping line: {line} ({e})")

    df_demo = pd.DataFrame(rows)
    return df_demo


def curate_files():
    print("=" * 65)
    print("DATASET CURATION & ORGANIZATION")
    print("=" * 65)

    info_path = RAW_DIR / "additional-information.txt"
    if info_path.exists():
        df_demo = parse_demographics(info_path)
        demo_out = TABLES_DIR / "patient_demographics.csv"
        df_demo.to_csv(demo_out, index=False)
        print(f"Parsed demographics for {len(df_demo)} patients -> {demo_out}")
        print("\nDemographics Summary by Group:")
        print(df_demo.groupby("group").agg(
            patients=("patient_id", "count"),
            mean_age=("age", "mean"),
            mean_BMI=("BMI", "mean"),
            mean_AHI=("AHI", "mean"),
            total_apnea_min=("apnea_minutes", "sum"),
            total_normal_min=("non_apnea_minutes", "sum"),
        ).round(2))

    # 1. Curate Learning Records (a01-a20, b01-b05, c01-c10)
    learning_ids = (
        [f"a{i:02d}" for i in range(1, 21)]
        + [f"b{i:02d}" for i in range(1, 6)]
        + [f"c{i:02d}" for i in range(1, 11)]
    )

    curated_count = 0
    for rec_id in learning_ids:
        target_folder = CURATED_DIR / rec_id
        target_folder.mkdir(parents=True, exist_ok=True)
        for ext in [".hea", ".dat", ".apn", ".qrs"]:
            src = RAW_DIR / f"{rec_id}{ext}"
            dst = target_folder / f"{rec_id}{ext}"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        curated_count += 1

    print(f"\nCurated {curated_count} learning records in: {CURATED_DIR}")

    # 2. Curate Multi-Signal Respiration Records (a01r-a04r, b01r, c01r-c03r)
    resp_ids = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]
    resp_count = 0
    for rec_id in resp_ids:
        target_folder = RESP_DIR / rec_id
        target_folder.mkdir(parents=True, exist_ok=True)
        for ext in [".hea", ".dat", ".apn"]:
            src = RAW_DIR / f"{rec_id}{ext}"
            dst = target_folder / f"{rec_id}{ext}"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        resp_count += 1
    print(f"Curated {resp_count} multi-channel respiration records in: {RESP_DIR}")

    # 3. Curate Test Records (x01-x35)
    test_ids = [f"x{i:02d}" for i in range(1, 36)]
    test_count = 0
    for rec_id in test_ids:
        target_folder = TEST_DIR / rec_id
        target_folder.mkdir(parents=True, exist_ok=True)
        for ext in [".hea", ".dat", ".qrs"]:
            src = RAW_DIR / f"{rec_id}{ext}"
            dst = target_folder / f"{rec_id}{ext}"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        test_count += 1
    print(f"Curated {test_count} test records in: {TEST_DIR}")
    print("\nDataset curation completed successfully.")


if __name__ == "__main__":
    curate_files()
