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
# 3. SELECT 20 SECONDS
# ============================================================

duration = 20
samples = int(duration * fs)

signal = filtered_ecg[:samples]

time = np.arange(len(signal)) / fs


# ============================================================
# 4. DIFFERENTIATION
# ============================================================

differentiated = np.diff(signal)

differentiated = np.append(
    differentiated,
    differentiated[-1]
)


# ============================================================
# 5. SQUARING
# ============================================================

squared = differentiated ** 2


# ============================================================
# 6. 150 ms MOVING-WINDOW INTEGRATION
# ============================================================

window_seconds = 0.15
window_samples = int(window_seconds * fs)

window = np.ones(window_samples) / window_samples

integrated = np.convolve(
    squared,
    window,
    mode="same"
)


# ============================================================
# 7. INITIAL SIGNAL / NOISE ESTIMATION
# ============================================================

initial_samples = int(2 * fs)

initial_signal = integrated[:initial_samples]

initial_peaks = []

for i in range(1, len(initial_signal) - 1):

    if (
        initial_signal[i] > initial_signal[i - 1]
        and initial_signal[i] >= initial_signal[i + 1]
    ):
        initial_peaks.append(initial_signal[i])

initial_peaks = np.array(initial_peaks)

if len(initial_peaks) == 0:
    raise RuntimeError(
        "No initial peaks found."
    )


# ============================================================
# 8. INITIAL ADAPTIVE THRESHOLD
# ============================================================

signal_level = 0.25 * np.max(initial_peaks)

noise_level = 0.5 * np.mean(initial_peaks)

threshold = noise_level + 0.25 * (
    signal_level - noise_level
)


# ============================================================
# 9. ADAPTIVE QRS CANDIDATE DETECTION
# ============================================================

candidate_peaks = []

refractory_period = int(0.25 * fs)

last_peak = -refractory_period

threshold_history = np.zeros(len(integrated))


for i in range(1, len(integrated) - 1):

    threshold_history[i] = threshold

    # Check local maximum
    if not (
        integrated[i] > integrated[i - 1]
        and integrated[i] >= integrated[i + 1]
    ):
        continue

    peak_value = integrated[i]


    # --------------------------------------------------------
    # ABOVE THRESHOLD
    # --------------------------------------------------------

    if peak_value > threshold:

        if i - last_peak >= refractory_period:

            candidate_peaks.append(i)

            last_peak = i

            signal_level = (
                0.875 * signal_level
                + 0.125 * peak_value
            )

        else:

            # Two candidates too close together
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
    # BELOW THRESHOLD = NOISE
    # --------------------------------------------------------

    else:

        noise_level = (
            0.875 * noise_level
            + 0.125 * peak_value
        )


    # --------------------------------------------------------
    # UPDATE THRESHOLD
    # --------------------------------------------------------

    threshold = noise_level + 0.25 * (
        signal_level - noise_level
    )


threshold_history[-1] = threshold


# ============================================================
# 10. R-PEAK LOCALIZATION
# ============================================================

candidate_peaks = np.array(
    candidate_peaks,
    dtype=int
)

# ±80 ms search region
search_time = 0.08

search_samples = int(
    search_time * fs
)

print(
    "\nR-peak search window: ±",
    search_time * 1000,
    "ms"
)

print(
    "At", fs,
    "Hz this corresponds to ±",
    search_samples,
    "samples"
)


r_peaks = []


for candidate in candidate_peaks:

    # --------------------------------------------------------
    # Define ±80 ms search region
    # --------------------------------------------------------

    start = max(
        0,
        candidate - search_samples
    )

    end = min(
        len(signal),
        candidate + search_samples + 1
    )


    # --------------------------------------------------------
    # Find maximum amplitude in filtered ECG
    # --------------------------------------------------------

    local_peak = start + np.argmax(
        signal[start:end]
    )

    r_peaks.append(local_peak)


r_peaks = np.array(
    r_peaks,
    dtype=int
)

# Remove duplicate locations
r_peaks = np.unique(r_peaks)

r_peak_times = r_peaks / fs


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n========================================")
print("R-PEAK DETECTION RESULTS")
print("========================================")

print(
    "Number of QRS candidates:",
    len(candidate_peaks)
)

print(
    "Number of detected R-peaks:",
    len(r_peaks)
)

print("\nR-peak locations:")

for i, peak in enumerate(r_peaks):

    print(
        f"R-peak {i + 1:2d}: "
        f"{peak / fs:.3f} s"
    )


# ============================================================
# 12. RR INTERVALS
# ============================================================

if len(r_peaks) > 1:

    rr_intervals = np.diff(
        r_peak_times
    )

    heart_rate = 60 / rr_intervals

    print("\nRR intervals:")

    print(
        np.round(
            rr_intervals,
            3
        )
    )

    print("\nInstantaneous heart rate:")

    print(
        np.round(
            heart_rate,
            2
        ),
        "BPM"
    )


# ============================================================
# 13. PLOT
# ============================================================

plt.figure(figsize=(16, 10))


# ============================================================
# TOP: FILTERED ECG + QRS CANDIDATES + R-PEAKS
# ============================================================

plt.subplot(2, 1, 1)

plt.plot(
    time,
    signal,
    label="Filtered ECG"
)

# QRS candidates
plt.scatter(
    candidate_peaks / fs,
    signal[candidate_peaks],
    color="orange",
    s=45,
    marker="x",
    label="QRS candidates"
)

# Final R-peaks
plt.scatter(
    r_peak_times,
    signal[r_peaks],
    color="red",
    s=40,
    label="Final R-peaks",
    zorder=3
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.title(
    "R-Peak Localization on Filtered ECG"
)

plt.legend()
plt.grid()


# ============================================================
# BOTTOM: INTEGRATED SIGNAL + ADAPTIVE THRESHOLD
# ============================================================

plt.subplot(2, 1, 2)

plt.plot(
    time,
    integrated,
    label="150 ms integrated signal"
)

plt.plot(
    time,
    threshold_history,
    linestyle="--",
    label="Adaptive threshold"
)

plt.scatter(
    candidate_peaks / fs,
    integrated[candidate_peaks],
    color="red",
    s=35,
    label="QRS candidates"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Integrated amplitude")

plt.title(
    "QRS Candidate Detection"
)

plt.legend()
plt.grid()

plt.tight_layout()

plt.show()