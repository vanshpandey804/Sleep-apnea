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
# 2. BAND-PASS FILTER
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
# 3. USE FIRST 20 SECONDS FOR TESTING
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
# 6. MOVING-WINDOW INTEGRATION
# ============================================================

integration_time = 0.15

integration_samples = int(
    integration_time * fs
)

window = (
    np.ones(integration_samples)
    / integration_samples
)

integrated = np.convolve(
    squared,
    window,
    mode="same"
)


# ============================================================
# 7. FIND ALL LOCAL PEAKS
# ============================================================

local_peaks = []

for i in range(1, len(integrated) - 1):

    if (
        integrated[i] > integrated[i - 1]
        and integrated[i] >= integrated[i + 1]
    ):

        local_peaks.append(i)

local_peaks = np.array(
    local_peaks,
    dtype=int
)


# ============================================================
# 8. INITIAL SIGNAL / NOISE ESTIMATION
# ============================================================

initial_samples = int(2 * fs)

initial_peaks = local_peaks[
    local_peaks < initial_samples
]

initial_values = integrated[
    initial_peaks
]

if len(initial_values) < 2:

    raise RuntimeError(
        "Not enough initial peaks for adaptive initialization."
    )


# ------------------------------------------------------------
# Robust initialization
# ------------------------------------------------------------

# Median gives a robust estimate of the typical
# background peak level.
noise_level = np.median(
    initial_values
)

# Upper percentile represents stronger peaks.
signal_level = np.percentile(
    initial_values,
    90
)


# Initial threshold
threshold = noise_level + 0.25 * (
    signal_level - noise_level
)


print("\nInitial noise level:",
      round(noise_level, 5))

print("Initial signal level:",
      round(signal_level, 5))

print("Initial threshold:",
      round(threshold, 5))


# ============================================================
# 9. DETECTOR PARAMETERS
# ============================================================

# Minimum separation between two QRS complexes
refractory_period = int(
    0.25 * fs
)

# T-wave discrimination region
t_wave_window = int(
    0.36 * fs
)

# Search-back condition
searchback_factor = 1.66


# ============================================================
# 10. STORAGE
# ============================================================

accepted_qrs = []

noise_peaks = []

threshold_history = np.zeros(
    len(integrated)
)

signal_history = np.zeros(
    len(integrated)
)

noise_history = np.zeros(
    len(integrated)
)


last_qrs = None

last_qrs_slope = None

rr_intervals_samples = []


# ============================================================
# 11. ADAPTIVE QRS DETECTION
# ============================================================

for peak in local_peaks:

    


    peak_value = integrated[peak]


    # --------------------------------------------------------
    # Store current adaptive values
    # --------------------------------------------------------

    threshold_history[peak] = threshold

    signal_history[peak] = signal_level

    noise_history[peak] = noise_level


    # --------------------------------------------------------
    # STEP A: PEAK ABOVE THRESHOLD
    # --------------------------------------------------------

    if peak_value > threshold:


        # ----------------------------------------------------
        # Check refractory period
        # ----------------------------------------------------

        if (
            last_qrs is not None
            and peak - last_qrs < refractory_period
        ):

            # Two peaks are too close.
            # Keep the stronger one.

            if peak_value > integrated[last_qrs]:

                accepted_qrs[-1] = peak

                last_qrs = peak

            continue


        # ----------------------------------------------------
        # T-WAVE DISCRIMINATION
        # ----------------------------------------------------

        is_t_wave = False


        if last_qrs is not None:

            time_since_qrs = (
                peak - last_qrs
            )


            if time_since_qrs < t_wave_window:

                # Calculate local slope around candidate
                current_slope = np.max(
                    np.abs(
                        differentiated[
                            max(0, peak - 5):
                            min(
                                len(differentiated),
                                peak + 6
                            )
                        ]
                    )
                )


                # Compare with previous QRS slope
                if last_qrs_slope is not None:

                    if current_slope < (
                        0.5 * last_qrs_slope
                    ):

                        is_t_wave = True


        # ----------------------------------------------------
        # Reject T-wave
        # ----------------------------------------------------

        if is_t_wave:

            noise_peaks.append(peak)

            noise_level = (
                0.875 * noise_level
                + 0.125 * peak_value
            )

        else:

            # ------------------------------------------------
            # Accept QRS
            # ------------------------------------------------

            accepted_qrs.append(peak)

            last_qrs = peak

            # Calculate slope of accepted QRS
            last_qrs_slope = np.max(
                np.abs(
                    differentiated[
                        max(0, peak - 5):
                        min(
                            len(differentiated),
                            peak + 6
                        )
                    ]
                )
            )

            # Update signal estimate
            signal_level = (
                0.875 * signal_level
                + 0.125 * peak_value
            )


    # --------------------------------------------------------
    # STEP B: PEAK BELOW THRESHOLD
    # --------------------------------------------------------

    else:

        noise_peaks.append(peak)

        noise_level = (
            0.875 * noise_level
            + 0.125 * peak_value
        )


    # --------------------------------------------------------
    # UPDATE ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    threshold = noise_level + 0.25 * (
        signal_level - noise_level
    )


# ============================================================
# 12. SEARCH-BACK FOR MISSED QRS
# ============================================================

accepted_qrs = np.array(
    accepted_qrs,
    dtype=int
)

accepted_qrs = np.unique(
    accepted_qrs
)


# We look at the intervals between accepted QRS peaks.
# If an interval is unusually long, search for a
# previously rejected peak.

searchback_peaks = []


for i in range(1, len(accepted_qrs)):

    previous = accepted_qrs[i - 1]

    current = accepted_qrs[i]

    rr = current - previous


    # Average RR from previously accepted beats
    if len(accepted_qrs[:i]) >= 2:

        previous_rrs = np.diff(
            accepted_qrs[:i]
        )

        average_rr = np.mean(
            previous_rrs
        )

    else:

        continue


    # --------------------------------------------------------
    # Search-back condition
    # --------------------------------------------------------

    if rr > searchback_factor * average_rr:

        search_start = previous + refractory_period

        search_end = current - refractory_period

        possible_peaks = local_peaks[
            (local_peaks > search_start)
            & (local_peaks < search_end)
        ]


        # Only consider peaks that were below
        # the normal threshold.

        possible_peaks = [
            p for p in possible_peaks
            if integrated[p] > (
                0.5 * threshold
            )
        ]


        if len(possible_peaks) > 0:

            # Select strongest missed candidate
            best_peak = max(
                possible_peaks,
                key=lambda p: integrated[p]
            )

            searchback_peaks.append(
                best_peak
            )


# Add search-back detections
if len(searchback_peaks) > 0:

    accepted_qrs = np.concatenate(
        [
            accepted_qrs,
            np.array(
                searchback_peaks,
                dtype=int
            )
        ]
    )

    accepted_qrs = np.sort(
        np.unique(accepted_qrs)
    )


# ============================================================
# 13. R-PEAK LOCALIZATION
# ============================================================

# ±80 ms search around QRS candidate
search_time = 0.08

search_samples = int(
    search_time * fs
)


r_peaks = []


for candidate in accepted_qrs:

    start = max(
        0,
        candidate - search_samples
    )

    end = min(
        len(signal),
        candidate + search_samples + 1
    )


    # Find actual maximum in filtered ECG
    local_peak = start + np.argmax(
        signal[start:end]
    )

    r_peaks.append(
        local_peak
    )


r_peaks = np.array(
    r_peaks,
    dtype=int
)

r_peaks = np.unique(
    r_peaks
)

r_peak_times = r_peaks / fs


# ============================================================
# 14. RR INTERVALS
# ============================================================

if len(r_peaks) > 1:

    rr_intervals = np.diff(
        r_peak_times
    )

    heart_rate = (
        60 / rr_intervals
    )

else:

    rr_intervals = np.array([])

    heart_rate = np.array([])


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n========================================")
print("IMPROVED QRS / R-PEAK DETECTION")
print("========================================")

print(
    "Initial local peaks:",
    len(initial_peaks)
)

print(
    "Initial threshold:",
    round(
        threshold_history[
            initial_peaks[-1]
        ],
        5
    )
)

print(
    "Normal QRS candidates:",
    len(accepted_qrs)
)

print(
    "Search-back candidates:",
    len(searchback_peaks)
)

print(
    "Final R-peaks:",
    len(r_peaks)
)


print("\nR-peak times (seconds):")

print(
    np.round(
        r_peak_times,
        3
    )
)


if len(rr_intervals) > 0:

    print("\nRR intervals (seconds):")

    print(
        np.round(
            rr_intervals,
            3
        )
    )

    print("\nHeart rate (BPM):")

    print(
        np.round(
            heart_rate,
            2
        )
    )


# ============================================================
# 16. CREATE THRESHOLD CURVE
# ============================================================

# The threshold is only updated at peak locations.
# Interpolate it between updates for visualization.

valid_threshold = np.where(
    threshold_history > 0
)[0]


if len(valid_threshold) > 1:

    threshold_curve = np.interp(
        np.arange(len(integrated)),
        valid_threshold,
        threshold_history[
            valid_threshold
        ]
    )

else:

    threshold_curve = np.full(
        len(integrated),
        threshold
    )


# ============================================================
# 17. PLOT
# ============================================================

plt.figure(
    figsize=(16, 11)
)


# ============================================================
# TOP: ECG + R-PEAKS
# ============================================================

plt.subplot(3, 1, 1)

plt.plot(
    time,
    signal,
    label="Filtered ECG"
)

plt.scatter(
    accepted_qrs / fs,
    signal[accepted_qrs],
    color="orange",
    marker="x",
    s=45,
    label="Accepted QRS candidates"
)

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
    "Improved QRS Detection and R-Peak Localization"
)

plt.legend()
plt.grid()


# ============================================================
# MIDDLE: SQUARED DERIVATIVE
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
# BOTTOM: INTEGRATED SIGNAL + THRESHOLD
# ============================================================

plt.subplot(3, 1, 3)

plt.plot(
    time,
    integrated,
    label="150 ms integrated signal"
)

plt.plot(
    time,
    threshold_curve,
    linestyle="--",
    label="Adaptive threshold"
)

plt.scatter(
    accepted_qrs / fs,
    integrated[accepted_qrs],
    color="red",
    s=35,
    label="Accepted QRS"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Integrated amplitude")

plt.title(
    "Adaptive QRS Detection with Search-Back"
)

plt.legend()
plt.grid()


plt.tight_layout()

plt.show()