
from pathlib import Path
import wfdb

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ao1"

print("=" * 60)
print("RESPIRATION DATASET INSPECTION")
print("=" * 60)
print(f"Data directory: {DATA_DIR}")

for record_name in ["a01", "a01r", "a01er"]:
    header_path = DATA_DIR / f"{record_name}.hea"

    print(f"\n--- {record_name} ---")

    if not header_path.exists():
        print("Header not found.")
        continue

    try:
        record = wfdb.rdheader(str(DATA_DIR / record_name))

        print(f"Sampling frequency: {record.fs} Hz")
        print(f"Number of signals: {record.n_sig}")
        print(f"Signal names: {record.sig_name}")
        print(f"Number of samples: {record.sig_len}")

        if record.fs:
            print(f"Duration: {record.sig_len / record.fs:.2f} seconds")

    except Exception as exc:
        print(f"Could not read header: {exc}")

print("\nInspection completed.")
