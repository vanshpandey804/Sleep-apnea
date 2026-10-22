
from pathlib import Path
import numpy as np
import wfdb

# These are the two independent annotation files.
modified_dir = Path(r"C:\msys64\tmp\sqrs125_no_minus4")
original_dir = Path(r"C:\msys64\tmp\sqrs125_test")

modified = wfdb.rdann(str(modified_dir / "a01"), "qrs")
original = wfdb.rdann(str(original_dir / "a01"), "qrs")

# The modified and original files are read on their WFDB time bases.
# Convert the 125 Hz annotation positions to the 100 Hz ECG sample grid.
modified_100 = np.rint(modified.sample * 100 / 125).astype(int)
original_100 = np.rint(original.sample * 100 / 125).astype(int)

print("Modified annotation count:", len(modified.sample))
print("Original annotation count:", len(original.sample))

# Pair corresponding annotations by order for an initial diagnostic.
n = min(len(modified_100), len(original_100))
difference_samples = modified_100[:n] - original_100[:n]
difference_ms = difference_samples * 10.0

print("\nPaired annotations:", n)
print("Median shift (ms):", np.median(difference_ms))
print("Mean shift (ms):", np.mean(difference_ms))
print("Minimum shift (ms):", np.min(difference_ms))
print("Maximum shift (ms):", np.max(difference_ms))
print("First 20 shifts (ms):", difference_ms[:20])
print("Count not shifted by 30-40 ms:",
      np.sum((difference_ms < 30) | (difference_ms > 40)))
