import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ============================================================
# 1. LOAD ECG
# ============================================================

record = wfdb.rdrecord("data/ao1/a01")

ecg = record.p_signal[:, 0]
fs = record.fs

print("Sampling frequency:", fs)
print("Total samples:", len(ecg))


# ============================================================
# 2. BAND-PASS FILTER: 0.5 - 40 Hz
# ============================================================

low_cutoff = 0.5
high_cutoff = 40

sos = butter(
    4,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs,
    output="sos"
)

filtered_ecg = sosfiltfilt(sos, ecg)


# ============================================================
# 3. SELECT 20 SECONDS FOR TESTING
# ============================================================

duration = 20
samples = int(duration * fs)

signal = filtered_ecg[:samples]

time = np.arange(len(signal)) / fs


# ============================================================
# 4. DIFFERENTIATION
# ============================================================

differentiated = np.diff(signal)

# Keep the signal length unchanged
differentiated = np.append(
    differentiated,
    differentiated[-1]
)


# ============================================================
# 5. SQUARING
# ============================================================

squared = differentiated ** 2


# ============================================================
# 6. MOVING-WINDOW INTEGRATION
# ============================================================

# 150 ms integration window
window_seconds = 0.15
window_samples = int(window_seconds * fs)

window = np.ones(window_samples) / window_samples

integrated = np.convolve(
    squared,
    window,
    mode="same"
)


# ============================================================
# 7. INITIAL PEAK ESTIMATION
# ============================================================

# Use first 2 seconds to initialize the detector
initial_samples = int(2 * fs)

initial_signal = integrated[:initial_samples]

# Find local maxima in the initial region
initial_peaks = []

for i in range(1, len(initial_signal) - 1):

    if (
        initial_signal[i] > initial_signal[i - 1]
        and initial_signal[i] >= initial_signal[i + 1]
    ):
        initial_peaks.append(initial_signal[i])

initial_peaks = np.array(initial_peaks)

# Safety check
if len(initial_peaks) == 0:
    raise RuntimeError(
        "No initial peaks found. Check the ECG signal."
    )


# ============================================================
# 8. INITIAL SIGNAL AND NOISE LEVELS
# ============================================================

# Pan-Tompkins-style initialization
signal_level = 0.25 * np.max(initial_peaks)

noise_level = 0.5 * np.mean(initial_peaks)


# Initial adaptive threshold
threshold = noise_level + 0.25 * (
    signal_level - noise_level
)


# ============================================================
# 9. ADAPTIVE QRS DETECTION
# ============================================================

candidate_peaks = []

# 250 ms refractory period
refractory_period = int(0.25 * fs)

last_peak = -refractory_period


# Store threshold at every sample
threshold_history = np.zeros(len(integrated))


for i in range(1, len(integrated) - 1):

    # Store current threshold
    threshold_history[i] = threshold


    # --------------------------------------------------------
    # Check for local maximum
    # --------------------------------------------------------

    is_local_peak = (
        integrated[i] > integrated[i - 1]
        and integrated[i] >= integrated[i + 1]
    )

    if not is_local_peak:
        continue


    peak_value = integrated[i]


    # --------------------------------------------------------
    # Peak above adaptive threshold
    # --------------------------------------------------------

    if peak_value > threshold:

        # Check refractory period
        if i - last_peak >= refractory_period:

            candidate_peaks.append(i)

            last_peak = i

            # Update signal estimate
            signal_level = (
                0.875 * signal_level
                + 0.125 * peak_value
            )


        else:

            # Two peaks are too close together.
            # Keep the stronger one.

            if len(candidate_peaks) > 0:

                previous_peak = candidate_peaks[-1]

                if peak_value > integrated[previous_peak]:

                    candidate_peaks[-1] = i

                    last_peak = i

                    signal_level = (
                        0.875 * signal_level
                        + 0.125 * peak_value
                    )


    # --------------------------------------------------------
    # Peak below threshold -> noise
    # --------------------------------------------------------

    else:

        noise_level = (
            0.875 * noise_level
            + 0.125 * peak_value
        )


    # --------------------------------------------------------
    # Update adaptive threshold
    # --------------------------------------------------------

    threshold = noise_level + 0.25 * (
        signal_level - noise_level
    )


# Fill final threshold value
threshold_history[-1] = threshold


# ============================================================
# 10. LOCATE ACTUAL R-PEAKS
# ============================================================

candidate_peaks = np.array(candidate_peaks, dtype=int)

r_peaks = []

# Search ±80 ms around QRS candidate
search_window = int(0.08 * fs)


for candidate in candidate_peaks:

    start = max(
        0,
        candidate - search_window
    )

    end = min(
        len(signal),
        candidate + search_window + 1
    )

    # Find maximum ECG amplitude
    local_peak = start + np.argmax(
        signal[start:end]
    )

    r_peaks.append(local_peak)


r_peaks = np.array(r_peaks, dtype=int)

# Remove duplicate positions
r_peaks = np.unique(r_peaks)

r_peak_times = r_peaks / fs


# ============================================================
# 11. PRINT DETECTION RESULTS
# ============================================================

print("\n----------------------------------------")
print("QRS DETECTION RESULTS")
print("----------------------------------------")

print("Number of detected R-peaks:",
      len(r_peaks))

print("\nR-peak times (seconds):")

print(
    np.round(r_peak_times, 3)
)


# ============================================================
# 12. RR INTERVALS AND HEART RATE
# ============================================================

if len(r_peaks) > 1:

    rr_intervals = np.diff(r_peak_times)

    heart_rate = 60 / rr_intervals

    print("\nRR intervals (seconds):")

    print(
        np.round(rr_intervals, 3)
    )

    print("\nInstantaneous heart rate (BPM):")

    print(
        np.round(heart_rate, 2)
    )

    print(
        "\nAverage heart rate:",
        round(np.mean(heart_rate), 2),
        "BPM"
    )


# ============================================================
# 13. PLOT RESULTS
# ============================================================

plt.figure(figsize=(16, 10))


# ============================================================
# TOP: FILTERED ECG + R-PEAKS
# ============================================================

plt.subplot(3, 1, 1)

plt.plot(
    time,
    signal,
    label="Filtered ECG"
)

plt.scatter(
    r_peak_times,
    signal[r_peaks],
    color="red",
    s=35,
    label="Detected R-peaks",
    zorder=3
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "Adaptive QRS Detection - Filtered ECG"
)

plt.legend()
plt.grid()


# ============================================================
# MIDDLE: DIFFERENTIATED + SQUARED
# ============================================================

plt.subplot(3, 1, 2)

plt.plot(
    time,
    squared,
    label="Squared derivative"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "Differentiated and Squared ECG"
)

plt.legend()
plt.grid()


# ============================================================
# BOTTOM: INTEGRATED SIGNAL + ADAPTIVE THRESHOLD
# ============================================================

plt.subplot(3, 1, 3)

plt.plot(
    time,
    integrated,
    label="Integrated signal"
)

plt.scatter(
    candidate_peaks / fs,
    integrated[candidate_peaks],
    color="red",
    s=35,
    label="QRS candidates",
    zorder=3
)

plt.plot(
    time,
    threshold_history,
    linestyle="--",
    label="Adaptive threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Integrated amplitude")

plt.title(
    "Moving-Window Integration and Adaptive Threshold"
)

plt.legend()
plt.grid()


plt.tight_layout()
plt.show()