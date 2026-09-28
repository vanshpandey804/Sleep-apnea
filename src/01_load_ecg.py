import wfdb
import matplotlib.pyplot as plt

# Load record
record = wfdb.rdrecord("data/ao1/a01")

# Extract ECG
ecg = record.p_signal[:, 0]

# Sampling frequency
fs = record.fs

print("Sampling frequency:", fs, "Hz")
print("Number of samples:", len(ecg))
print("Duration:", len(ecg) / fs, "seconds")

# Time axis
time = [i / fs for i in range(len(ecg))]

# Plot first 20 seconds
plt.figure(figsize=(12, 4))
plt.plot(time, ecg)

plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude")
plt.title("Raw ECG - Record a01")

plt.xlim(0, 20)
plt.grid()
plt.show()