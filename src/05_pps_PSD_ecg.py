import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# ============================================================
# 1. LOAD THE ENTIRE ECG RECORDING
# ============================================================

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("==========================================")
print("ECG PSD QUANTITATIVE ANALYSIS")
print("==========================================")

print("Sampling frequency:", fs, "Hz")
print("Total samples:", len(ecg))
print("Total duration:", len(ecg) / 3600, "hours")


# ============================================================
# 2. FUNCTION TO CALCULATE PSD
# ============================================================

def calculate_psd(signal, fs):

    # Remove DC component
    signal = signal - np.mean(signal)

    frequency, power = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=1024,
        noverlap=512
    )

    return frequency, power


# ============================================================
# 3. FUNCTION TO CALCULATE POWER IN A FREQUENCY BAND
# ============================================================

def band_power(frequency, power, low, high):

    mask = (frequency >= low) & (frequency <= high)

    return np.trapezoid(
        power[mask],
        frequency[mask]
    )


# ============================================================
# 4. FIND FREQUENCY CONTAINING A GIVEN % OF POWER
# ============================================================

def cumulative_power_frequency(
        frequency,
        power,
        low_frequency=0.5,
        high_frequency=50,
        percentage=0.99):

    mask = (
        (frequency >= low_frequency) &
        (frequency <= high_frequency)
    )

    f = frequency[mask]
    p = power[mask]

    # Cumulative integral of PSD
    cumulative = np.zeros(len(f))

    for i in range(1, len(f)):

        cumulative[i] = (
            cumulative[i - 1]
            + 0.5 * (p[i] + p[i - 1])
            * (f[i] - f[i - 1])
        )

    total_power = cumulative[-1]

    normalized = cumulative / total_power

    # Find first frequency reaching desired percentage
    index = np.where(
        normalized >= percentage
    )[0][0]

    # Linear interpolation for better estimate
    if index == 0:

        cutoff = f[0]

    else:

        f1 = f[index - 1]
        f2 = f[index]

        p1 = normalized[index - 1]
        p2 = normalized[index]

        cutoff = (
            f1
            + (percentage - p1)
            * (f2 - f1)
            / (p2 - p1)
        )

    return cutoff, total_power


# ============================================================
# 5. ANALYZE THREE PARTS OF OVERNIGHT RECORD
# ============================================================

total_samples = len(ecg)

# Beginning: first 5 minutes
beginning = ecg[:int(5 * 60 * fs)]

# Middle: 5 minutes around the middle
middle_start = total_samples // 2

middle = ecg[
    middle_start:
    middle_start + int(5 * 60 * fs)
]

# End: last 5 minutes
end = ecg[-int(5 * 60 * fs):]


segments = {
    "Beginning": beginning,
    "Middle": middle,
    "End": end
}


# ============================================================
# 6. ANALYZE EACH SEGMENT
# ============================================================

results = {}

for name, signal in segments.items():

    print("\n------------------------------------------")
    print(name)
    print("------------------------------------------")

    frequency, power = calculate_psd(
        signal,
        fs
    )

    results[name] = (
        frequency,
        power
    )

    # Calculate power bands
    total_power = band_power(
        frequency,
        power,
        0.5,
        50
    )

    power_0_5_5 = band_power(
        frequency,
        power,
        0.5,
        5
    )

    power_5_10 = band_power(
        frequency,
        power,
        5,
        10
    )

    power_10_20 = band_power(
        frequency,
        power,
        10,
        20
    )

    power_20_40 = band_power(
        frequency,
        power,
        20,
        40
    )

    power_40_50 = band_power(
        frequency,
        power,
        40,
        50
    )

    print(
        "Total power (0.5-50 Hz):",
        total_power
    )

    print(
        "0.5-5 Hz:",
        100 * power_0_5_5 / total_power,
        "%"
    )

    print(
        "5-10 Hz:",
        100 * power_5_10 / total_power,
        "%"
    )

    print(
        "10-20 Hz:",
        100 * power_10_20 / total_power,
        "%"
    )

    print(
        "20-40 Hz:",
        100 * power_20_40 / total_power,
        "%"
    )

    print(
        "40-50 Hz:",
        100 * power_40_50 / total_power,
        "%"
    )

    # Find cumulative-power frequencies
    for percentage in [0.90, 0.95, 0.99, 0.995]:

        cutoff, _ = cumulative_power_frequency(
            frequency,
            power,
            0.5,
            50,
            percentage
        )

        print(
            f"{percentage * 100:.1f}% power below: "
            f"{cutoff:.2f} Hz"
        )


# ============================================================
# 7. PLOT PSD OF BEGINNING, MIDDLE AND END
# ============================================================

plt.figure(figsize=(14, 7))

for name, (frequency, power) in results.items():

    plt.plot(
        frequency,
        power,
        label=name
    )

plt.xlim(0, 50)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power / Hz")

plt.title(
    "Welch PSD Comparison - Entire Overnight Record"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL INTERPRETATION")
print("==========================================")

print(
    "The values above show the actual frequencies "
    "containing 90%, 95%, 99% and 99.5% of the "
    "measured ECG power between 0.5 and 50 Hz."
)

print(
    "\nWe will use these results together with "
    "QRS preservation to select the final "
    "high-frequency cutoff."
)