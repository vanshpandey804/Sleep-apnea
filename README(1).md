# Table of Contents

1. [Loading the ECG Dataset](#1-loading-the-ecg-dataset)
2. [Preprocessing the Dataset](#2-preprocessing-the-dataset)
   1. [Initial Butterworth Filter](#21-initial-butterworth-filter)
   2. Validating the Filter
      1. Lower-Frequency Analysis in the Time Domain
      2. Higher-Frequency Analysis in the Time Domain
      3. Higher-Frequency Analysis in the Frequency Domain
      4. Finding a Suitable Higher Frequency Using PSD Analysis
3. [QRS and R-Peak Detection](#3-qrs-and-r-peak-detection)
   1. Detector 1: Basic
   2. Detector 2: Adaptive
   3. Detector 3: Basic R-Peak Detection
   4. Detector 4: Improved R-Peak Detection
4. [Verifying QRS and R-Peak Detectors Using Supplied QRS Annotations](#4-verifying-qrs-and-r-peak-detectors-using-supplied-qrs-annotations)
   1. Ground Truth vs. Supplied Annotations: Initial Discussion and Objectives
   2. Detector 1 Performance
   3. Detector 2 Performance
   4. Why Detector 3 Is Skipped
   5. Detector 4 Performance
   6. Final Comparison and Selection of the Best-Performing Detector
   7. Timing Error
      1. Analysis Over 8 Hours
      2. Investigating Alignment as a Possible Cause
      3. Investigating Filtering as a Possible Cause
      4. SQRS125
      5. SQRS125 and Its `-4` Method as a Possible Cause
5. [ECG-Derived Respiration (EDR)](#5-ecg-derived-respiration-edr)
   1. RR-Interval-Based EDR
   2. QRS-Morphology-Based EDR
      1. QRS Peak-to-Peak Amplitude
      2. R-Wave Amplitude
      3. QRS Area
   3. Inspecting Respiration Channels in the Supplied Respiration Data
   4. Validation and Performance of Each EDR Method
   5. Frequency-Based EDR
   6. Final Discussion

---

# 1. Loading the ECG Dataset

## Overview

The first step in the project is to load the raw Electrocardiogram (ECG) signal for subsequent preprocessing, QRS complex detection, R-peak detection, and ECG-Derived Respiration (EDR) analysis.

The dataset is loaded using the **WFDB (Waveform Database)** Python package, which provides tools for reading physiological signals stored in WFDB-compatible formats.

## Implementation

The ECG record is loaded using `wfdb.rdrecord()`. The signal is then extracted from the first channel of the record, and its sampling frequency is retrieved from the record metadata.

```python
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
```

### Key Parameters

- **Record path:** `data/ao1/a01`
- **Signal extraction:** `record.p_signal[:, 0]` selects the first signal channel.
- **Sampling frequency (`fs`):** Retrieved from the record metadata in hertz (Hz).
- **Number of samples:** The total number of samples in the extracted ECG signal.
- **Signal duration:** Calculated as `T = N / f_s`, where:
  - `N` is the number of samples.
  - `f_s` is the sampling frequency in hertz (Hz).

## Visualizing the Raw ECG Signal

A time axis is constructed by assigning each sample a timestamp based on the sampling frequency:

`t[n] = n / f_s`

Here, `n` is the sample index, starting from zero, and `f_s` is the sampling frequency.

The complete signal is plotted against this time axis, with the horizontal axis restricted to the first 20 seconds for visualization.

```python
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
```

This initial visualization provides a view of the unprocessed ECG waveform before filtering or peak detection is performed.

## Outcome

At the end of this stage, the following are available for subsequent processing:

- The raw ECG signal extracted from the first channel.
- The sampling frequency required for time-domain and frequency-domain analysis.
- The total number of samples and corresponding signal duration.
- A visualization of the first 20 seconds of the raw ECG waveform.

The extracted signal serves as the input to the preprocessing stage.

### Raw ECG Signal — Record a01

The following figure shows the raw ECG waveform over the first 20 seconds of record `a01`.

![Raw ECG signal for record a01 over the first 20 seconds](Images/Screenshot-2026-09-07-042025.png)

**Figure 1:** Raw ECG signal from record `a01` over the first 20 seconds.

---

# 2. Preprocessing the Dataset

## Objective

The raw ECG signal contains the physiological information of interest along with unwanted noise components. Low-frequency disturbances, such as baseline wander, can obscure the underlying waveform, while high-frequency noise can interfere with the identification of QRS complexes and R-peaks.

To improve the signal quality before peak detection, a **Butterworth band-pass filter** is initially applied. Rather than treating the initial cutoff frequencies as final, the filtering process is evaluated through time-domain and frequency-domain analysis. The lower and upper frequency limits are then investigated separately to determine a suitable passband that suppresses noise while preserving the important characteristics of the ECG waveform.

## 2.1 Initial Butterworth Filter

A band-pass filter is selected to attenuate unwanted frequency components below and above the desired ECG frequency range.

The **Butterworth filter** is chosen because of its maximally flat passband response, which avoids passband ripple and provides a smooth frequency response. This makes it a reasonable starting point for ECG preprocessing, where unnecessary amplitude variations within the retained frequency range are undesirable.

An initial passband of **0.5–40 Hz** is selected, with a specified design order of **4**. This provides a practical starting point for evaluating noise attenuation and waveform preservation. Higher filter orders can produce sharper frequency transitions, but may also increase numerical sensitivity and the risk of waveform distortion.

The filter is implemented using `scipy.signal.butter()` and applied using `scipy.signal.filtfilt()`. Forward-backward filtering eliminates phase shift in the resulting signal, which is particularly useful when the timing of QRS complexes and R-peaks will be examined in subsequent stages.

The removed component is calculated as:

`x_removed[n] = x_raw[n] - x_filtered[n]`

This component represents the difference between the original and filtered signals. It helps visualize what the filter has removed, although it should not be interpreted as noise alone, since some physiological signal content may also be attenuated.

The raw ECG, filtered ECG, and removed component are plotted together over the first 20 seconds of record `a01` to provide an initial qualitative assessment.

### Initial Filter Parameters

| Parameter | Value |
|---|---|
| Filter type | Butterworth band-pass |
| Lower cutoff frequency | 0.5 Hz |
| Upper cutoff frequency | 40 Hz |
| Specified design order | 4 |
| Filtering method | Forward-backward (`filtfilt`) |
| Record | `a01` |
| Visualization interval | First 20 seconds |

### Next Steps

The initial filter is only a starting point. Its performance is subsequently investigated by examining the effects of the lower and upper cutoff frequencies separately.

- **Lower-frequency analysis in the time domain:** Examine how low-frequency components affect the baseline and overall ECG waveform.
- **Higher-frequency analysis in the time domain:** Evaluate how the upper cutoff affects waveform detail and sharp QRS features.
- **Higher-frequency analysis in the frequency domain:** Investigate the distribution of signal energy across frequencies.
- **PSD-based cutoff selection:** Use Power Spectral Density (PSD) analysis to help identify a suitable upper cutoff frequency.

These investigations guide the refinement of the initial passband before proceeding to QRS and R-peak detection.

### 2.2 Validating the Filter

The initial Butterworth band-pass filter uses a passband of 0.5–40 Hz. Before finalizing these cutoff frequencies, their effects on the ECG waveform are examined to assess whether unwanted components are attenuated while important signal characteristics are preserved.

The filter is evaluated by varying its lower and upper cutoff frequencies separately while keeping the remaining parameters unchanged. The filtered signals are compared with the raw ECG in the time domain, and the differences between filtered versions are examined to understand the effects of changing each cutoff frequency.

#### 2.2.1 Lower-Frequency Analysis in the Time Domain

To investigate the effect of low-frequency removal, two Butterworth band-pass filters are compared:

- **Filter A:** 0.5–40 Hz
- **Filter B:** 0.05–40 Hz

The upper cutoff frequency is kept at 40 Hz so that the comparison isolates the effect of changing the lower cutoff frequency.

The two filtered signals are compared alongside the raw ECG, and their difference is calculated as:

`D(t) = ECG_0.5–40(t) - ECG_0.05–40(t)`

The difference signal helps identify the waveform components affected by the change in cutoff frequency.

In the analysed 20-second segment, the two filtered signals appear largely similar around the QRS complexes. Their difference is predominantly a slow variation rather than sharp, QRS-shaped features. This indicates that changing the lower cutoff has relatively little effect on the major cardiac complexes in this segment, primarily affecting slower signal variations instead.

Since lowering the cutoff to 0.05 Hz does not produce a significant visible improvement in the preservation of the major cardiac complexes, **we proceed with 0.5 Hz as the lower cutoff frequency** for subsequent preprocessing.

#### 2.2.2 Higher-Frequency Analysis in the Time Domain

To investigate the effect of high-frequency removal, two Butterworth band-pass filters are compared:

- **Filter A:** 0.5–40 Hz
- **Filter B:** 0.5–20 Hz

The lower cutoff frequency is kept at 0.5 Hz so that the comparison primarily examines the effect of reducing the upper cutoff frequency from 40 Hz to 20 Hz.

The two filtered signals are compared alongside the raw ECG, and their difference is calculated as:

`D(t) = ECG_0.5–40(t) - ECG_0.5–20(t)`

The difference signal helps identify the waveform components affected by changing the upper cutoff frequency.

In the analysed segment, the difference signal contains prominent, repeated sharp spikes around the heartbeats. The QRS complexes in the 0.5–40 Hz filtered signal also appear sharper and more clearly defined than those in the 0.5–20 Hz filtered signal, where they appear smoother and shorter.

These observations indicate that changing the upper cutoff frequency can substantially alter the ECG waveform, particularly the QRS morphology. Unlike the lower-cutoff comparison, where the major QRS complexes remained largely similar, this comparison reveals a more pronounced effect. **Therefore, rather than immediately finalizing the upper cutoff at 40 Hz, we must investigate the ECG's frequency content to determine a suitable upper cutoff frequency.**

The next step is to perform frequency-domain analysis, including power spectral density (PSD) analysis, to examine how the signal's spectral content is distributed and help guide the choice of cutoff frequency.

#### 2.2.3 Higher-Frequency Analysis in the Frequency Domain (FFT)

Following the time-domain comparison, an initial frequency-domain analysis is performed to examine the distribution of frequency components in the raw ECG signal. The first 20 seconds of record `a01` are analysed using the Fast Fourier Transform (FFT).

Before computing the FFT, the mean of the ECG segment is subtracted to remove its DC component. The single-sided magnitude spectrum is then calculated and plotted over the frequency range of 0–50 Hz. Since the sampling frequency is 100 Hz, the Nyquist frequency is 50 Hz.

The magnitude spectrum provides an initial view of the frequency content of the ECG. The observed spectrum shows substantial content across the lower and intermediate frequencies, with the magnitude generally decreasing at higher frequencies. However, the magnitude spectrum alone does not provide a sufficiently robust basis for determining the final filter cutoff frequencies.

**Conclusion:** This FFT analysis serves as an initial investigation of the ECG's frequency content, rather than a final basis for selecting the preprocessing filter. To obtain a more reliable understanding of how signal power is distributed across frequency bands, the next step is to perform **Power Spectral Density (PSD) analysis using Welch's method**. This will help quantify the relative power present in different frequency ranges and support a more informed selection of the upper cutoff frequency.

#### 2.2.4 Finding a Suitable Higher Cutoff Frequency Using PSD Analysis

To make an informed decision about the upper cutoff frequency, Welch's Power Spectral Density (PSD) analysis is performed on the ECG recording. Unlike the initial FFT analysis, which provides a view of the frequency components present in the signal, PSD analysis helps quantify how the signal's power is distributed across different frequency bands.

Insert this paragraph after the introductory paragraph of Section 2.2.4, before describing the analysis of the three recording segments:

Welch's method estimates the Power Spectral Density (PSD) by dividing a signal into overlapping segments, applying a window function to each segment, calculating its periodogram, and averaging the resulting estimates. This averaging reduces fluctuations in the spectral estimate compared with analysing a single segment, providing a smoother representation of how signal power is distributed across frequency. In this analysis, Welch's method is used to quantify the ECG power across different frequency bands and identify the frequencies below which specified percentages of the measured power are concentrated.

To ensure that the observations are not limited to a short segment, the recording is analysed at three locations: the beginning, middle, and end of the overnight record `a01`. For each segment, the cumulative power distribution is used to identify the frequencies below which 90%, 95%, 99%, and 99.5% of the measured power lies.

The results are summarized below:

| Power criterion | Beginning (Hz) | Middle (Hz) | End (Hz) |
|---|---:|---:|---:|
| 90% | 20.88 | 21.04 | 21.56 |
| 95% | 23.79 | 24.11 | 24.43 |
| 99% | 33.96 | 34.56 | 33.99 |
| 99.5% | 39.95 | 40.34 | 40.02 |

The results are consistent across all three portions of the recording. The frequency containing 99% of the measured power lies at approximately 34 Hz, while the frequency containing 99.5% lies at approximately 40 Hz. This suggests that most of the measured ECG power is concentrated below 40 Hz, with relatively little power between 40 and 50 Hz.

The band-wise power distribution supports this observation. Across the three segments, the 20–40 Hz band accounts for approximately 11.5–13.3% of the measured power, whereas the 40–50 Hz band accounts for only about 0.5%. However, low power alone does not establish that a frequency component is noise, nor does high power guarantee that it is physiologically useful. The PSD results must therefore be interpreted alongside the time-domain filter comparisons.

The earlier time-domain analysis showed that reducing the upper cutoff from 40 Hz to 20 Hz noticeably altered the QRS complexes, making them smoother and less sharply defined. This indicates that higher-frequency components contribute to the observed QRS morphology. Consequently, selecting the cutoff solely from the 99% cumulative-power frequency of approximately 34 Hz could remove some of these components without establishing whether doing so would improve the signal for subsequent analysis.

**Conclusion:** The lower-cutoff comparison supports using 0.5 Hz, as changing it had relatively little effect on the major QRS complexes in the analysed segment. The upper-cutoff comparison shows that reducing the cutoff too far can noticeably alter QRS morphology. Furthermore, the PSD analysis across the beginning, middle, and end of record `a01` consistently places the 99.5% cumulative-power frequency at approximately 40 Hz.

Taken together, these findings provide sufficient justification to **finalize 0.5–40 Hz as the Butterworth band-pass filter for preprocessing**. The same filter will be applied consistently to every ECG recording in the dataset.

## 3. QRS and R-Peak Detection

The objective of this stage is to identify the QRS complexes in the filtered ECG and determine the corresponding R-peak locations. These locations provide the timing information needed to characterize the cardiac rhythm.

Simply identifying the largest peaks in the ECG is not sufficiently robust. Although the R-wave is often a prominent feature, the signal can also contain P and T waves, baseline variations, muscle noise, electrode artifacts, and abnormal beats. Some of these can produce local peaks that may be mistaken for R-peaks.

To make QRS complexes easier to distinguish, the detection process follows the main principles of the **Pan–Tompkins approach**. The implementation is developed through four detector versions, progressing from a fixed-threshold method to adaptive detection, explicit R-peak localization, and an improved detector incorporating additional decision logic.

### 3.1 Detector 1: Basic Pan–Tompkins-Style Detection

The first detector implements the main signal-processing stages of the Pan–Tompkins approach: differentiation, squaring, moving-window integration, thresholding, and R-peak localization.

#### 3.1.1 Differentiation

Differentiation emphasizes rapid changes in the ECG waveform. Since QRS complexes generally contain steeper slopes than the surrounding ECG waveform, differentiation makes these regions more prominent.

A first difference is calculated as:

$$
y[n] = x[n] - x[n-1]
$$

where `x[n]` is the filtered ECG signal and `y[n]` is its differentiated output.

#### 3.1.2 Squaring

The differentiated signal is squared:

$$
y_{\mathrm{sq}}[n] = \bigl(y[n]\bigr)^2
$$

Squaring makes all values non-negative and emphasizes larger differences more strongly than smaller ones. Consequently, regions with rapid changes in the ECG produce prominent peaks in the squared signal.

#### 3.1.3 Moving-Window Integration

The squared signal is passed through a moving-window integrator to smooth rapid fluctuations and emphasize regions containing sustained signal energy.

For a window of `N` samples, the integrated output can be expressed as:

$$
y_{\mathrm{int}}[n]
= \frac{1}{N}\sum_{k=0}^{N-1} y_{\mathrm{sq}}[n-k]
$$

The implementation uses a **150 ms integration window**. At the dataset's sampling frequency of 100 Hz, this corresponds to:

$$
N = 0.15 \times 100 = 15 \text{ samples}
$$

The integrated signal helps identify the approximate regions containing QRS complexes. Its peaks are not necessarily located at the exact R-wave maxima, so it is used for QRS candidate detection rather than final R-peak timing.

#### 3.1.4 Fixed-Threshold Detection

In the first implementation, the detection threshold is set to 30% of the maximum value of the integrated signal:

$$
THR = 0.30 \times \max_n\bigl(y_{\mathrm{int}}[n]\bigr)
$$

Local maxima in the integrated signal that exceed this threshold are treated as QRS candidates. A minimum separation of **250 ms** is enforced between candidates to reduce multiple detections within the same QRS complex.

This approach is straightforward, but its threshold depends on the maximum value across the analysed segment. An unusually large peak can therefore raise the threshold and make smaller genuine QRS complexes harder to detect. Conversely, a threshold based on the segment's maximum may not remain suitable when the signal amplitude or noise level changes.

#### 3.1.5 R-Peak Localization

After a QRS candidate is identified, the detector returns to the filtered ECG and searches for the maximum ECG amplitude within a local window around that candidate.

The initial implementation searches within **±80 ms** of each candidate. The sample with the largest ECG amplitude in this region is selected as the R-peak.

This distinction is important: the integrated signal identifies *where a QRS complex is likely to be*, whereas the filtered ECG is used to locate the actual R-peak.

The detector then converts the detected sample indices into time using:

$$
t_R = \frac{n_R}{f_s}
$$

where `n_R` is the R-peak sample index and `f_s` is the sampling frequency.

**Limitation:** The fixed threshold is not adaptive to changing signal conditions. This motivates the second detector.

### 3.2 Detector 2: Adaptive Threshold Detection

The second detector retains the same preprocessing and signal-transformation stages but replaces the fixed threshold with an adaptive threshold based on estimated signal and noise peak levels.

#### 3.2.1 Adaptive Threshold

Instead of using a fixed fraction of the maximum integrated value, the detector maintains two estimates:

- `SPKI`: the estimated level of signal peaks associated with QRS complexes.
- `NPKI`: the estimated level of noise peaks and other non-QRS peaks.

The threshold is calculated as:

$$
THR = NPKI + 0.25(SPKI - NPKI)
$$

This places the threshold 25% of the way from the estimated noise level toward the estimated signal level.

The estimates are updated as the detector processes successive peaks. The implementation uses the following exponential updates:

$$
SPKI_{\mathrm{new}}
= 0.875\,SPKI_{\mathrm{old}} + 0.125\,PEAK
$$

$$
NPKI_{\mathrm{new}}
= 0.875\,NPKI_{\mathrm{old}} + 0.125\,PEAK
$$

Here, `PEAK` is the current peak value in the integrated signal. Peaks accepted as QRS candidates update the signal-level estimate, while peaks below the threshold update the noise-level estimate. The threshold is recalculated after these updates.

The coefficients 0.875 and 0.125, together with the threshold fraction 0.25, follow the exponential-update and threshold structure used in the Pan–Tompkins method. However, the initialization procedure used in this implementation is a custom adaptation rather than a complete reproduction of the original algorithm.

#### 3.2.2 Initialization

The initial signal and noise levels are estimated from local maxima in the first two seconds of the integrated signal. The implementation initializes the estimates as:

$$
SPKI = 0.25 \times \max(PEAKS_{\mathrm{initial}})
$$

$$
NPKI = 0.5 \times \operatorname{mean}(PEAKS_{\mathrm{initial}})
$$

The initial threshold is then calculated using the adaptive-threshold equation above.

These initialization multipliers are implementation choices. They should not be confused with the subsequent adaptive update equations.

#### 3.2.3 Refractory Period

A **250 ms refractory period** is applied to prevent the detector from accepting multiple QRS candidates too close together. If another sufficiently strong candidate occurs within this interval, the implementation retains the stronger of the two candidates.

This helps reduce duplicate detections within a single QRS complex. However, a fixed refractory period also limits how closely two separate beats can be detected, so it is an important implementation parameter.

**Improvement over Detector 1:** The threshold now changes with the estimated signal and noise levels instead of depending exclusively on the largest peak in the segment. This makes the detector more responsive to changing signal conditions, although its initialization and detection decisions still require validation.

### 3.3 Detector 3: Dedicated R-Peak Localization

The third detector separates QRS candidate detection from the process of locating the R-peak. It retains the adaptive QRS detection stage and focuses explicitly on converting each detected QRS region into a sample-level R-peak location.

#### 3.3.1 Why Localization Is Necessary

The moving-window integration stage smooths and combines energy across a window of samples. As a result, the maximum of the integrated signal does not necessarily coincide with the maximum of the original filtered ECG.

The integrated signal therefore provides a candidate location, not necessarily the precise time of the R-wave. Using its peak directly as the R-peak timestamp could introduce a timing error.

#### 3.3.2 Local Maximum Search

For each QRS candidate at sample `n_c`, the implementation searches the filtered ECG within a window of ±80 ms:

$$
[n_c - N_s,\; n_c + N_s]
$$

where:

$$
N_s = \left\lfloor 0.08 f_s \right\rfloor
$$

At the sampling frequency of 100 Hz:

$$
N_s = \left\lfloor 0.08 \times 100 \right\rfloor
= 8 \text{ samples}
$$

Thus, the search covers approximately eight samples before and after the candidate. Within this interval, the sample with the maximum ECG amplitude is selected as the R-peak. The search boundaries are clipped to the available signal so that candidates near the beginning or end of the segment can be processed safely.

The selected sample indices are converted to timestamps by dividing by the sampling frequency. Duplicate sample positions are removed.

The ±80 ms window is an implementation choice rather than a universal Pan–Tompkins constant. It provides a local search region around the candidate, but its suitability depends on the timing accuracy of the QRS candidates and the morphology of the ECG. Its effect can be assessed by comparing alternative window sizes against reference annotations.

**Outcome:** Detector 3 establishes a clear distinction between identifying QRS candidates and locating their corresponding R-peaks. This separation is retained in the improved detector.

### 3.4 Detector 4: Improved QRS and R-Peak Detection

The fourth detector extends the adaptive implementation with additional logic intended to address initialization, possible T-wave detections, and missed beats. Its processing pipeline is:

$$
\begin{aligned}
\text{Filtered ECG}
&\rightarrow \text{Differentiation}\\
&\rightarrow \text{Squaring}\\
&\rightarrow \text{Moving-window integration}\\
&\rightarrow \text{Adaptive thresholding}\\
&\rightarrow \text{QRS candidate decisions}\\
&\rightarrow \text{R-peak localization}
\end{aligned}
$$

The detector retains the previously selected 0.5–40 Hz preprocessing filter, 150 ms integration window, and ±80 ms R-peak search window. The current implementation processes the first 20 seconds of the ECG for testing and visualization.

#### 3.4.1 Revised Signal and Noise Initialization

Rather than initializing the signal and noise levels using fixed fractions of the maximum and mean of the initial peaks, the improved implementation uses the distribution of local peak values during the first two seconds.

The initialization is:

$$
NPKI = \operatorname{median}(PEAKS_{\mathrm{initial}})
$$

$$
SPKI = P_{90}(PEAKS_{\mathrm{initial}})
$$

where `P90` is the 90th percentile of the initial peak values.

The median provides a robust measure of the typical peak level, while the upper percentile supplies an estimate of stronger peaks. If the signal estimate is not greater than the noise estimate, the maximum initial peak is used as a fallback.

The threshold is initialized using the same adaptive-threshold equation:

$$
THR = NPKI + 0.25(SPKI - NPKI)
$$

This is a **custom statistical initialization** intended to be less sensitive to extreme peak values. It is not the original Pan–Tompkins initialization procedure, and its performance must be established empirically.

The initial two-second segment is used to estimate these levels, but its candidates are also processed during detection.

**Limitation and proposed improvement:** All local maxima in the initialization interval may include small fluctuations that are not meaningful QRS candidates. Initialization could be improved by applying preliminary peak-selection criteria before estimating the signal and noise levels. The initial threshold should also be stored separately before detection begins so that it can be reported accurately rather than inferred from threshold values recorded later.

#### 3.4.2 T-Wave Discrimination

T-waves can sometimes produce peaks in the transformed signal that resemble QRS candidates. The improved detector therefore applies an additional check to candidates occurring within **360 ms** of the previous accepted QRS.

For these candidates, the implementation compares the local slope around the current candidate with the slope around the previous accepted QRS. The slope is estimated from the maximum absolute value of the differentiated signal in a small local region.

A candidate is classified as a possible T-wave when its estimated slope is less than half that of the previous accepted QRS:

$$
S_{\mathrm{candidate}} < 0.5\,S_{\mathrm{previous\ QRS}}
$$

Candidates classified as possible T-waves are rejected as QRS detections and contribute to the noise-level estimate. Other candidates that satisfy the detection conditions can be accepted as QRS complexes.

This is a heuristic decision rule and does not guarantee that every T-wave will be rejected or every genuine QRS will be accepted.

**Limitation and proposed improvement:** The integrated-signal peak may be shifted relative to the actual QRS complex. Consequently, a derivative window centred on the integrated peak may not capture the most representative QRS slope. The slope measurement should be aligned more carefully with the corresponding ECG event, and the 50% criterion should be validated against representative ECG segments.

#### 3.4.3 Search-Back for Potentially Missed Beats

A detector can miss a genuine QRS when its peak does not exceed the current threshold. The improved implementation adds a search-back stage to look for possible missed candidates when an interval between accepted QRS detections appears unusually long.

The search-back condition is based on the average of up to eight preceding accepted RR intervals. At least two preceding intervals must be available:

$$
RR_{\mathrm{current}} > 1.66\,RR_{\mathrm{average}}
$$

When this condition is met, the detector searches between the two accepted QRS candidates, excluding regions close to either candidate according to the refractory-period constraint.

A candidate is eligible for recovery if its integrated amplitude was below its original threshold but greater than half that threshold:

$$
0.5THR_P < P \leq THR_P
$$

Here, \(THR_P\) is the threshold recorded when the candidate was originally evaluated. Candidates previously rejected as T-waves are excluded. If eligible candidates are found, the one with the largest integrated-signal amplitude is added as a recovered QRS candidate.

The factor 1.66 is associated with classical Pan–Tompkins-style search-back logic. The specific implementation here is a simplified search-back heuristic rather than a complete reproduction of the original algorithm.

**Limitation and proposed improvement:** Search-back must use the threshold that was active when each candidate was evaluated, rather than the final threshold after the main detection pass. The threshold history should therefore be stored for every candidate. Candidates must also be verified as having fallen below their original thresholds. Since the current search-back stage runs only once, a more systematic implementation should re-evaluate the resulting beat sequence after inserting recovered candidates and check for any newly implausible RR intervals.

#### 3.4.4 Final R-Peak Localization

After the normal detection and search-back stages, the accepted QRS candidates are sorted and duplicate positions are removed. The detector then applies the same local-maximum search used in Detector 3.

For each accepted candidate, the maximum amplitude in the filtered ECG within ±80 ms is selected as the final R-peak. This assumes that the R waves have positive, upward deflections in the filtered ECG.

A final refractory-period check is applied after localization. If two localized peaks are too close together, the one with the greater filtered ECG amplitude is retained. The resulting sample indices are converted to timestamps using the sampling frequency.

#### 3.4.5 Implementation Checks and Limitations

The detector's intermediate and final outputs can be inspected using plots of:

1. The filtered ECG with accepted QRS candidates and localized R-peaks.
2. The differentiated and squared signal.
3. The moving-window integrated signal with the adaptive threshold.



An additional issue arises during refractory-period handling. If a stronger candidate replaces the previously accepted candidate, the signal-level estimate and the stored QRS slope must be updated consistently. Otherwise, subsequent thresholding and T-wave discrimination may use stale state information. The replacement logic should therefore update the relevant state before continuing to the next candidate.

Overall, Detector 4 adds percentile-based initialization, adaptive thresholding, T-wave discrimination, refractory-period handling, and search-back. These mechanisms remain heuristic and require evaluation against the reference QRS annotations to determine whether they improve detection accuracy over the earlier detectors.


## 4. Verifying QRS and R-Peak Detectors Using Supplied QRS Annotations

### 4.1 Ground Truth vs. Supplied Annotations: Initial Discussion and Objectives

The detectors are evaluated over the full `a01` ECG record, approximately **8.2 hours long**, rather than relying only on the initial 20-second plots. The supplied `a01.qrs` annotations are used as a reference for quantitative evaluation, not as absolute ground truth: they are machine-generated and unaudited, so disagreements do not automatically mean that the detector is wrong.

At a sampling frequency of 100 Hz, a matching tolerance of \(\pm 50\text{ ms}\) corresponds to \(\pm 5\) samples. Each detected peak is matched to at most one reference annotation within this tolerance. The resulting counts are used to calculate:

$$
\text{Sensitivity}=\frac{TP}{TP+FN}
$$

$$
\text{PPV}=\frac{TP}{TP+FP}
$$

$$
F1=\frac{2TP}{2TP+FP+FN}
$$

Here, \(TP\) denotes matched peaks, \(FP\) denotes unmatched detections, and \(FN\) denotes unmatched reference annotations. Timing error is calculated as the detected sample index minus the matched reference sample index; a positive value means the detected peak occurs later.

These metrics measure agreement with the supplied annotations. Disagreements, particularly apparent false positives and false negatives, should be examined against the ECG waveform before being treated as definite detector errors.

### 4.2 Detector 1 Performance

**Validation method**

The saved Detector 1 R-peak indices are loaded and matched chronologically against the `a01.qrs` reference annotations using a \(\pm 50\text{ ms}\) tolerance. Matched pairs are used to calculate TP, FP, FN, sensitivity, PPV, F1 score, and absolute timing-error statistics. The matched results are saved to `a01_detector1_matched_pairs.npy`.

**Boundary correction**

A small correction was made to the R-peak localization window:

```python
end = min(len(signal), candidate + search_window + 1)
```

Previously, the expression omitted `+ 1`. Since Python slices exclude their ending index, this change includes the intended right-hand sample in the search, unless the signal boundary is reached first.

**Validation results**

| Metric | Result |
|---|---:|
| Detected R-peaks | 16,231 |
| Reference annotations | 29,938 |
| True positives (TP) | 16,230 |
| False positives (FP) | 1 |
| False negatives (FN) | 13,708 |
| Sensitivity | 54.21% |
| PPV | 99.99% |
| F1 score | 70.31% |
| Median absolute timing error | 40 ms |
| Maximum absolute timing error | 50 ms |

Detector 1 has very high PPV but low sensitivity. Almost every detected peak matches a supplied annotation, but a large number of reference annotations remain unmatched. This is consistent with the limitation of its fixed threshold:

```python
threshold = 0.30 * np.max(integrated)
```

The threshold is determined by the maximum value of the integrated signal across the recording. It does not adapt to changes in ECG amplitude, so weaker QRS complexes may fall below it.

The matched peaks also show a tendency to occur later than the reference annotations. The median absolute timing error is 40 ms, while the maximum is 50 ms under the chosen matching tolerance. This timing behaviour should be investigated separately; Detector 1 is retained as the fixed-threshold baseline rather than tuned to improve its scores.

### 4.3 Detector 2 Performance

**Validation method**

The saved Detector 2 R-peak indices are matched to the nearest available reference annotation within \(\pm 50\text{ ms}\), ensuring that a reference annotation is not matched more than once. TP, FP, FN, sensitivity, PPV, F1 score, and timing-error statistics are calculated, and the matched pairs are saved to `a01_detector2_matched_pairs.npy`.

**Validation results**

| Metric | Result |
|---|---:|
| Detected R-peaks | 29,949 |
| Reference annotations | 29,938 |
| True positives (TP) | 29,938 |
| False positives (FP) | 11 |
| False negatives (FN) | 0 |
| Sensitivity | 100.00% |
| PPV | 99.96% |
| F1 score | 99.98% |
| Mean absolute timing error | 41.32 ms |

Against the supplied annotations, Detector 2 matches every reference annotation, with 11 additional detections. Its sensitivity is therefore 100% under the selected matching criterion, but this should not be interpreted as 100% clinical accuracy because the reference annotations are not independently verified ground truth.

A notable result is the timing error: matched detections are generally around 40–50 ms later than the reference annotations. This suggests a systematic localization offset that should be investigated before using the R-peak times for RR-interval, HRV, or EDR calculations. The timing difference should not be corrected by simply subtracting a constant without first examining the matched waveforms and signed-error distribution.

### 4.4 Why Detector 3 Is Skipped

Detector 3 was created separately to demonstrate R-peak localization by searching for the maximum ECG amplitude within a window of \(\pm 80\text{ ms}\) around a candidate. It uses the same underlying QRS detection logic as Detector 2 rather than introducing an independent detector.

Since the validation of the final R-peak indices already evaluates the results after localization, a separate Detector 3 validation would largely repeat the same evaluation. It is therefore omitted from the performance comparison.

### 4.5 Detector 4 Performance

**Validation method**

The saved Detector 4 indices and `a01.qrs` annotations are loaded, sorted, and deduplicated before chronological one-to-one matching within \(\pm 50\text{ ms}\). Unmatched detections and reference annotations are counted as FP and FN, respectively, and performance and signed timing-error statistics are calculated. Matched pairs are saved to `a01_detector4_matched_pairs.npy`.

**Validation results**

| Metric | Result |
|---|---:|
| Detected R-peaks | 29,942 |
| Reference annotations | 29,938 |
| True positives (TP) | 29,938 |
| False positives (FP) | 4 |
| False negatives (FN) | 0 |
| Sensitivity | 100.00% |
| PPV | 99.99% |
| F1 score | 99.99% |
| Mean absolute timing error | 41.32 ms |
| Mean signed timing error | +41.32 ms |

Detector 4 matches all supplied reference annotations, with four additional detections and no unmatched reference annotations. Its positive mean signed timing error indicates that detected peaks occur, on average, approximately 41 ms later than the reference annotations. At 100 Hz, this is approximately four samples.

The high matching scores do not resolve the timing offset. Possible explanations include differences in how the reference annotations and the localization rule define the QRS/R-peak location, shifts introduced by selecting the maximum amplitude within the \(\pm 80\text{ ms}\) window, or imprecision in the reference timestamps. The matched waveforms and signed timing-error distribution should be examined before applying any timing correction.

### 4.6 Final Comparison and Selection of the Best-Performing Detector

The validation results are summarized below using the same \(\pm 50\text{ ms}\) matching tolerance.

| Metric | Detector 1 | Detector 2 | Detector 4 |
|---|---:|---:|---:|
| Detected R-peaks | 16,231 | 29,949 | 29,942 |
| Reference annotations | 29,938 | 29,938 | 29,938 |
| True positives | 16,230 | 29,938 | 29,938 |
| False positives | 1 | 11 | 4 |
| False negatives | 13,708 | 0 | 0 |
| Sensitivity | 54.21% | 100.00% | 100.00% |
| PPV | 99.99% | 99.96% | 99.99% |
| F1 score | 70.31% | 99.98% | 99.99% |
| Mean absolute timing error | — | 41.32 ms | 41.32 ms |

*Note: The supplied Detector 1 results report median and maximum absolute timing error, rather than mean absolute timing error.*

**Detector 4 is selected for the subsequent ECG-derived respiration analysis.** It matches all 29,938 supplied annotations, has no unmatched reference annotations, and produces only four unmatched detections. Its sensitivity, PPV, and F1 score are all approximately 100% under the selected matching criterion.

This selection is based on agreement with the supplied annotations, not proof of clinical superiority. The remaining timing offset and the four unmatched detections still warrant waveform inspection. In addition, the individual contributions of Detector 4's added features, such as T-wave discrimination and search-back, have not been isolated through ablation tests. The results therefore support using Detector 4 as the current implementation for subsequent analysis while retaining these limitations in the interpretation.


### 4.7 Timing Error

Detector 4 achieved 100% sensitivity and 99.99% positive predictive value (PPV) against the supplied `a01.qrs` annotations. However, detection accuracy and timing accuracy are different measures. Although Detector 4 matched all 29,938 reference annotations, the detected peaks did not coincide with the reference sample positions.

The full-record analysis revealed a consistent positive timing offset. The detected peaks occurred approximately 41 ms after their corresponding reference annotations, with most errors concentrated at +40 ms. This section investigates whether the offset changes over the recording, whether it arises from the R-peak localization procedure, and whether filtering could explain the observed difference.

The analysis uses the complete `a01` ECG recording, containing 2,957,000 samples at 100 Hz, or approximately 8.214 hours. The supplied QRS annotations are machine-generated and unaudited; consequently, the results establish agreement or disagreement with those annotations, not independently verified clinical accuracy.

#### 4.7.1 Analysis Over 8 Hours

The signed timing error was calculated for every matched detection-reference pair:

$$
e_i = \frac{n_{\mathrm{detected},i}-n_{\mathrm{reference},i}}{f_s}\times1000
$$

where \(n_{\mathrm{detected},i}\) is the detected sample index, \(n_{\mathrm{reference},i}\) is the reference sample index, and \(f_s=100\text{ Hz}\).

A positive error indicates that the detected peak occurs after the reference annotation, while a negative error indicates that it occurs before the annotation.

**Full-record timing-error statistics**

| Metric | Result |
|---|---:|
| Matched beats | 29,938 |
| Mean signed error | +41.318 ms |
| Median signed error | +40 ms |
| Standard deviation | 3.477 ms |
| Mean absolute error | 41.318 ms |
| Median absolute error | 40 ms |
| 90th-percentile absolute error | 50 ms |
| 95th-percentile absolute error | 50 ms |
| Maximum absolute error | 50 ms |
| Detections after reference | 29,938 |
| Detections before reference | 0 |
| Exact sample matches | 0 |

Every matched detection occurred after its corresponding reference annotation. Since all errors were positive, the mean signed error and mean absolute error are identical.

**Quantization of the timing error**

At a sampling frequency of 100 Hz, the interval between consecutive samples is:

$$
\Delta t=\frac{1}{f_s}
=\frac{1}{100}\text{ s}
=10\text{ ms}
$$

Both the detected peaks and the reference annotations are represented by discrete sample indices. Their difference must therefore be an integer number of samples, making the timing error quantized in increments of 10 ms:

$$
e_i=k\Delta t=k(10\text{ ms}),\qquad k\in\mathbb{Z}
$$

The observed distribution is:

| Signed timing error | Number of beats | Percentage |
|---|---:|---:|
| +30 ms (3 samples) | 97 | 0.32% |
| +40 ms (4 samples) | 25,799 | 86.17% |
| +50 ms (5 samples) | 4,042 | 13.50% |
| **Total** | **29,938** | **100.00%** |

The dominant error is +40 ms, accounting for 86.17% of matched beats. Another 13.50% have an error of +50 ms, while only 0.32% have an error of +30 ms.

Quantization explains why the measured errors occur in 10 ms increments, but it does not explain why nearly all errors are positive or why the dominant difference is four samples. Those observations require investigation of the detector's localization method and the reference annotations.

**Timing Error Over the Full Recording**

The following figure shows the signed timing error between Detector 4 and the supplied QRS annotations over the complete recording.

![Timing error over the full eight-hour recording](Images/Screenshot-2026-10-22-001103.png)

**Figure 4.7.1:** Signed timing error between Detector 4 and the supplied QRS annotations over the full recording.

The timing error remains within approximately +30 to +50 ms, without an obvious progressive drift over the recording. This is consistent with a systematic offset rather than a discrepancy that accumulates with time.

A growing timing error could suggest a time-base mismatch or synchronization problem. The observed distribution does not show that pattern. Instead, the offset remains confined to a narrow range throughout the approximately 8.2-hour recording.

#### 4.7.2 Investigating Alignment as a Possible Cause

Detector 4 uses differentiation, squaring, and 150 ms moving-window integration to identify QRS candidates. These operations help identify regions containing QRS complexes, but the maximum of the integrated signal is not necessarily the location of the R-peak in the original ECG.

After accepting a QRS candidate, Detector 4 searches within ±80 ms of that candidate in the filtered ECG and selects the sample with the maximum positive amplitude. The final detected R-peak is therefore determined by the local maximum of the filtered waveform rather than by the maximum of the integrated signal.

To investigate the timing discrepancy visually, representative beats distributed across the recording were examined. The plots compare the reference annotation, Detector 4's detected peak, and the local ECG maximum.

The following figure presents the representative ECG alignment plots.

![Representative ECG alignment plots](Images/Screenshot-2026-10-22-001647.png)

**Figure 4.7.2:** Representative ECG segments comparing the supplied reference annotations with Detector 4's localized R-peaks.

In these examples, the detected peak lies near the local positive maximum of the ECG waveform, while the reference annotation appears approximately 40 ms earlier. The visualizations therefore suggest that the timing difference may be related to the locations selected by the reference annotations and the detector's maximum-amplitude localization rule.

However, the visual examples alone cannot establish that this relationship holds throughout the recording. The full-record timing analysis provides the broader evidence: all 29,938 matched detections occur after their corresponding reference annotations, and the observed errors range from +30 to +50 ms.

The distinction between a waveform maximum and a reference annotation is important. The largest positive deflection is not guaranteed to be the correct R-wave fiducial for every ECG morphology. Similarly, the supplied `a01.qrs` annotations are machine-generated and have not been independently audited. The plots therefore demonstrate a consistent difference in timing, but do not prove that the reference annotations are incorrect.

#### 4.7.3 Investigating Filtering as a Possible Cause

The next investigation examined whether filtering substantially shifts the ECG maxima, potentially explaining why Detector 4's detections occur later than the reference annotations.

Detector 4 applies a fourth-order Butterworth band-pass filter with cut-off frequencies of 0.5 Hz and 40 Hz. The filter is implemented using second-order sections and `sosfiltfilt`, which performs forward-backward filtering to avoid the phase delay associated with conventional one-directional filtering.

To assess the effect of filtering, the raw ECG and filtered ECG were compared using all 29,938 matched detection-reference pairs.

For each matched beat, the diagnostic script located the maximum positive amplitude in both the raw and filtered ECG. The search window extended 250 ms beyond the earlier and later of the reference and detected sample positions. The resulting maxima were compared with the reference annotations and with each other.

**Full-record raw-versus-filtered comparison**

| Measurement | Mean signed offset | Median offset | Standard deviation | Observed range |
|---|---:|---:|---:|---:|
| Detector 4 relative to reference | +41.32 ms | +40 ms | 3.48 ms | +30 to +50 ms |
| Raw ECG maximum relative to reference | +41.99 ms | +40 ms | 4.00 ms | +30 to +50 ms |
| Filtered ECG maximum relative to reference | +41.32 ms | +40 ms | 3.48 ms | +30 to +50 ms |
| Detector 4 relative to filtered ECG maximum | 0 ms | 0 ms | 0 ms | 0 ms |
| Raw maximum relative to filtered maximum | +0.67 ms | 0 ms | 2.50 ms | −10 to +10 ms |

The most significant result is that Detector 4 selected exactly the same sample as the local positive maximum of the filtered ECG for all 29,938 matched beats. The detector-to-filtered-maximum timing difference was zero for every matched pair.

The raw and filtered ECG maxima were also closely aligned. Their mean signed difference was +0.67 ms, the median difference was 0 ms, and the observed range was −10 to +10 ms. Thus, the raw and filtered maxima differed by no more than one sample, equivalent to 10 ms at the recording's sampling frequency.

The following figure compares the raw and filtered ECG waveforms and shows the reference annotation and detected peak.

![Raw versus filtered ECG comparison](Images/Screenshot-2026-10-22-002244.png)

**Figure 4.7.3:** Comparison of raw and filtered ECG waveforms, with the reference annotation and detected peak marked.

These measurements indicate that the approximately 40 ms timing discrepancy is not explained by a substantial displacement between the raw and filtered ECG maxima. Although filtering can alter ECG morphology, the measured raw-versus-filtered maximum displacement in this analysis is at most one sample, whereas the dominant detector-to-reference offset is four samples.

The results are consistent with Detector 4 selecting the local positive maximum of the filtered waveform while the reference annotations identify an earlier sample. The use of forward-backward filtering also makes a conventional filter phase delay an unlikely explanation for the systematic offset.

Nevertheless, these findings do not prove that the maximum-amplitude location is clinically correct or that the supplied annotations are wrong. They establish only that Detector 4 matches the filtered ECG's local maximum and that filtering does not account for the observed timing difference to any substantial degree.

**Conclusion**

The full-record investigation establishes three findings:

1. **The offset is systematic:** all 29,938 matched detections occur after their reference annotations, with a mean signed timing error of +41.318 ms.
2. **The errors are quantized:** at 100 Hz, the 10 ms sampling interval produces discrete timing errors. The dominant offset is +40 ms, followed by +50 ms and +30 ms.
3. **Filtering does not explain the offset:** Detector 4 matches the filtered ECG's local positive maximum at exactly the same sample for every matched beat, while raw and filtered maxima differ by no more than one sample.

No fixed −40 ms correction has been applied, and Detector 4 has not been modified as a result of this investigation. Applying such a correction solely to improve agreement with unaudited annotations would assume that the reference timing is correct without independently verifying that assumption.

#### 4.7.4 SQRS125

The preceding analysis showed that Detector 4 consistently identifies the local positive maximum of the filtered ECG, whereas the supplied reference annotations occur earlier. The next step was to investigate the origin of the reference annotations themselves. In particular, it was necessary to determine whether the supplied `a01.qrs` file could be reproduced by rerunning the algorithm used to generate it.

The supplied annotations were generated using `sqrs125`, a QRS-detection utility from the WFDB software suite. The PhysioNet documentation identifies these annotations as machine-generated and not hand-edited. A fresh execution of `sqrs125` therefore provides a way to test whether the supplied annotations are reproducible under the same detection method.

**Reproducing the reference annotations**

The official `sqrs125` utility was compiled and executed on a temporary copy of the original ECG record. The supplied `a01.qrs` file was retained unchanged, and the newly generated annotations were saved separately as `C:\msys64\tmp\sqrs125_test\a01.qrs`.

The compilation environment was provided by MinGW/MSYS2, which supplied the Windows-compatible GCC toolchain. Initial attempts to update or install packages encountered package-mirror timeouts; the availability of existing build tools and network connectivity was checked before proceeding with compilation.

**Accounting for the sampling frequency**

A direct comparison of the annotation sample indices would have been invalid because the original ECG was sampled at 100 Hz, whereas `sqrs125` internally processes recordings at 125 Hz. The fresh annotation indices were therefore converted to the original time base:

$$
n_{100}=\operatorname{round}\left(n_{125}\frac{100}{125}\right)
$$

where \(n_{125}\) denotes a sample index from the fresh annotations and \(n_{100}\) denotes the corresponding index on the original 100 Hz time base.

After conversion, the two annotation sets were compared using one-to-one beat matching with a tolerance of 50 ms, corresponding to five samples at 100 Hz. This allowed detections to be classified as matches, unmatched fresh detections, or unmatched reference annotations.

**Table 4.7.4.1: Comparison of freshly generated and supplied SQRS125 annotations**

| Measurement | Result |
|---|---:|
| Supplied reference annotations | 29,938 |
| Freshly generated annotations | 29,939 |
| True positives (TP) | 29,937 |
| False positives (FP) | 2 |
| False negatives (FN) | 1 |
| Sensitivity | 99.9967% |
| Precision | 99.9933% |
| F1 score | 99.9950% |
| Mean absolute timing difference | 4.60 ms |
| Median absolute timing difference | 0 ms |
| Maximum absolute timing difference | 10 ms |

The two annotation sets agree extremely closely: 29,937 fresh detections were matched to reference annotations within the specified tolerance. Among these matched beats, the median absolute timing difference was zero, and the maximum was one sample at the original sampling frequency. The fresh run produced one additional detection overall, while one supplied reference annotation remained unmatched.

**Errors encountered and resolutions**

- **MSYS2 package-mirror timeouts:** Checked network connectivity and the availability of existing build tools before proceeding with compilation.
- **Compilation environment:** Verified the required compiler and supporting utilities to build and execute the official `sqrs125` implementation.
- **Sampling-frequency mismatch:** Converted the fresh annotation indices from 125 Hz to the original 100 Hz time base before evaluating agreement.
- **Annotation comparison:** Used one-to-one matching within a fixed tolerance to calculate the performance metrics and identify unmatched detections without modifying the source annotation files.

**Conclusion**

The experiment demonstrates that the supplied `a01.qrs` annotations can be reproduced with near-perfect agreement by a fresh execution of `sqrs125`, provided that the sampling-frequency difference is accounted for. The small number of unmatched detections indicates that the two runs were not completely identical, but their overall agreement was exceptionally high.

This result must be interpreted carefully. **Reproducibility does not establish clinical accuracy.** Because both annotation sets were generated using the same detection algorithm, their agreement cannot independently verify whether the annotated samples correspond to the clinically correct R-wave fiducial points. The experiment validates the reproducibility of the annotation-generation process, not the correctness of the annotations themselves.

#### 4.7.5 SQRS125 and Its `-4` Method as a Possible Cause

The source code of `sqrs125` contains an explicit four-sample subtraction when assigning QRS annotation timestamps. This provides a concrete explanation for a substantial part of the timing difference previously observed between the supplied reference annotations and Detector 4.

The relevant statement appears twice in `app/sqrs125.c`, at lines 250 and 255:

```c
annot.time = now - (time - qtime) - 4;
```

To understand the adjustment, it is necessary to examine how the detector establishes its timing reference.

**Interpretation of the timestamp calculation**

When the filtered signal first crosses the adaptive slope threshold, the detector records the current local sample counter:

```c
qtime = time;
```

This marks the beginning of a candidate QRS event, as recognized by the detector. The algorithm subsequently monitors the waveform before deciding whether to annotate the event.

The timestamp expression can be interpreted as follows:

| Expression | Interpretation |
|---|---|
| `now` | Overall sample-time counter |
| `time` | Local sample counter, reset after a normal beat is annotated |
| `qtime` | Local counter when the initial slope threshold is crossed |
| `time - qtime` | Elapsed local samples since the initial threshold crossing |
| `now - (time - qtime)` | Reconstructed timestamp associated with that earlier crossing |
| `- 4` | Additional four-sample backward adjustment |

The subtraction of `time - qtime` reconstructs an earlier time associated with the onset of the detected event, rather than simply using the time at which the detection decision is completed. The additional `-4` then shifts the resulting annotation further backwards.

The source establishes that this adjustment is intentional in the timestamp calculation. However, the code alone does not establish why the original author selected exactly four samples.

**Determining the effective time base**

The original ECG is sampled at 100 Hz, whereas `sqrs125` is designed to operate at approximately 125 Hz. Its source contains the following condition:

```c
if (sampfreq((char *)NULL) < 120. ||
    sampfreq((char *)NULL) > 130.)
    setifreq(125.);
```

For the supplied 100 Hz record, this condition selects a 125 Hz input-processing rate. Consequently, the four-sample adjustment corresponds nominally to

$$
\Delta t = \frac{4}{125}
= 0.032\ \text{s}
= 32\ \text{ms}.
$$

It would be incorrect to interpret the subtraction directly as 40 ms merely because the original ECG has a 10 ms sampling interval. The adjustment is made within the detector's 125 Hz processing time base.

The fresh annotation output was also checked using WFDB's annotation-reading utilities. The displayed sample positions correspond to the original record's time grid, so the internal processing rate must be distinguished from the time base used when interpreting annotation positions. The fresh annotations were converted to the original 100 Hz grid before the full-record comparisons were performed.

**Testing the effect of the four-sample adjustment**

Rather than relying on the source code alone, a controlled experiment was performed. Both occurrences of

```c
annot.time = now - (time - qtime) - 4;
```

were changed in a separate experimental build to

```c
annot.time = now - (time - qtime);
```

The original source, supplied reference annotations, and unmodified detector output were preserved. The modified executable generated a separate annotation set, allowing the effect of removing the adjustment to be measured directly.

The experiment produced 29,939 annotations in both the modified and unmodified runs. After converting both annotation sets to the original 100 Hz time base, all 29,939 annotations could be paired, with no unmatched annotations.

**Table 4.7.5.1: Effect of removing the explicit four-sample adjustment**

| Measurement | Result |
|---|---:|
| Annotations compared | 29,939 |
| Unmatched annotations | 0 |
| Theoretical shift at 125 Hz | 32 ms |
| Measured mean shift | +32.056 ms |
| Annotations showing a +30 ms shift | 23,784 |
| Annotations showing a +40 ms shift | 6,155 |

The measured mean shift of +32.056 ms closely agrees with the theoretical 32 ms adjustment. The individual shifts appear as 30 ms or 40 ms after conversion to the 100 Hz sample grid, where each sample represents 10 ms. These discrete values are therefore consistent with the difference between the two sampling time bases and the rounding used during conversion.

This controlled comparison establishes that removing `-4` moves the annotations later by approximately 32 ms throughout the recording.

**Effect on Detector 4's timing error**

The modified annotations were then used as an experimental reference for the unchanged Detector 4. The comparison retained the same 50 ms beat-matching tolerance.

**Table 4.7.5.2: Detector 4 timing error against the original and modified SQRS125 annotations**

| Measurement | Original reference | Modified reference |
|---|---:|---:|
| Reference annotations | 29,938 | 29,939 |
| Matched beats | 29,938 | 29,937 |
| Mean signed timing error | +41.318 ms | +4.665 ms |
| Median signed timing error | +40 ms | 0 ms |
| Mean absolute timing error | 41.318 ms | 4.688 ms |
| 95th-percentile absolute error | 50 ms | 10 ms |
| Maximum absolute error | 50 ms | 20 ms |
| F1 score | Approximately 99.99% | 99.995% |

Removing the adjustment reduces Detector 4's mean signed timing error from +41.318 ms to +4.665 ms. The mean absolute error also decreases substantially, from 41.318 ms to 4.688 ms. The modified reference therefore produces much closer timing agreement with Detector 4.

The remaining difference should not automatically be interpreted as a detector error. It may reflect sample-grid quantization, the matching procedure, and differences between the timing conventions used by the two algorithms.

**Possible reasons for the `-4` adjustment**

The experiment establishes the effect of the adjustment but does not establish the original author's exact rationale. Several explanations are plausible:

1. **Compensation for detector decision delay:** The algorithm confirms a candidate QRS only after observing its slope pattern. The adjustment may compensate for the difference between the detector's decision time and its intended annotation time.
2. **Empirical alignment with the expected QRS location:** The four-sample constant may have been selected experimentally to align the algorithm's annotations with its intended QRS timing convention.
3. **An implementation-specific timing correction:** Since `sqrs125` is designed around a nominal 125 Hz processing rate, the adjustment may have been tuned to the timing behavior of this particular implementation rather than representing a universal physiological constant.

These explanations remain hypotheses. The source code and controlled experiment establish that the adjustment shifts timestamps; they do not establish why it was originally introduced or whether it improves clinical localization.

**Conclusion**

Three findings are supported by the investigation:

1. The source code explicitly subtracts four samples when assigning annotation timestamps.
2. Removing this subtraction shifts all 29,939 annotations later, producing a measured mean shift of +32.056 ms, consistent with the theoretical 32 ms adjustment at 125 Hz.
3. Detector 4's mean signed timing error decreases from +41.318 ms against the supplied annotations to +4.665 ms against the modified annotations.

Together, these results strongly support the four-sample adjustment as a major contributor to the timing difference observed between Detector 4 and the supplied SQRS125 annotations. However, the experiment does not establish that the adjustment accounts for the entire original offset or that either annotation set identifies the clinically correct R-wave fiducial point.

The modified SQRS125 output is therefore useful for investigating the algorithm's timing convention, but it is not an independent ground-truth reference. Detector 4 remains unchanged, and the original reference annotations are retained for the existing evaluation.


# 5. ECG-Derived Respiration (EDR)

## 5.1 RR-Interval-Based EDR

The first ECG-derived respiration (EDR) method uses variations in the intervals between consecutive R-peaks detected by the finalized Detector 4. The underlying principle is respiratory sinus arrhythmia (RSA), through which breathing can modulate heart rate and consequently produce respiratory-related fluctuations in the RR-interval sequence.

For consecutive R-peaks at sample indices \(r_i\) and \(r_{i+1}\), the RR interval is calculated as

$$
RR_i=\frac{r_{i+1}-r_i}{f_s}
$$

where \(f_s=100\text{ Hz}\) is the ECG sampling frequency.

Since the R-peaks occur at irregular time intervals, the resulting RR sequence is not uniformly sampled. To make it suitable for subsequent filtering and frequency-domain analysis, the beat-level sequence was interpolated onto a uniform time grid at 4 Hz.

The implementation generated two files:

- `results/tables/a01_edr_rr_beats.csv`: RR intervals associated with their beat-related timestamps.
- `results/tables/a01_edr_rr_uniform.csv`: the interpolated RR-interval signal sampled at 4 Hz.

### Results for record `a01`

| Measurement                | Result   |
| -------------------------- | -------- |
| R-peaks loaded             | 29,938   |
| RR intervals generated     | 29,937   |
| Mean RR interval           | 0.9875 s |
| Median RR interval         | 1.0000 s |
| Uniform sampling frequency | 4 Hz     |
| Uniform EDR samples        | 118,253  |
| Represented duration       | 29,563 s |

The represented duration is approximately 8 hours, 12 minutes, and 43 seconds, consistent with an overnight recording.

These results confirm successful generation of the RR-interval signal. They do not establish that the signal accurately represents respiration. RR intervals are influenced by other physiological processes, irregular beats, movement, and noise in addition to respiratory modulation.

The 4 Hz interpolation provides a uniform time axis for subsequent processing; it does not create additional physiological information. The respiratory relevance of this candidate must therefore be evaluated against an independently measured respiratory signal.

## 5.2 QRS-Morphology-Based EDR

The second EDR approach uses changes in the ECG waveform itself rather than changes in the time between heartbeats.

Respiratory movement can alter the position and orientation of the heart relative to the recording electrodes, changing the electrical relationship between the heart and the measured ECG. These changes can produce respiratory modulation of QRS amplitude and morphology.

Three candidate features were extracted around the R-peaks identified by Detector 4:

1. QRS peak-to-peak amplitude.
2. R-wave amplitude.
3. QRS area.

Each feature was measured beat by beat and interpolated onto a uniform 4 Hz time grid. The resulting signals were then compared against measured nasal airflow to determine whether their variations tracked respiratory activity.

The morphology extraction produced:

- `results/tables/a01_edr_morphology_beats.csv`
- `results/tables/a01_edr_morphology_uniform.csv`

The uniform morphology signal contained 118,256 samples, compared with 118,253 samples for the RR-derived signal. The small difference arises because the methods represent slightly different time ranges.

### 5.2.1 QRS Peak-to-Peak Amplitude

QRS peak-to-peak amplitude measures the voltage range of the ECG within a window centred on each detected R-peak.

For a measurement window \(W_i\) around the \(i\)-th R-peak, the feature is

$$
A_{\mathrm{pp},i}
=
\max_{n\in W_i}x[n]
-
\min_{n\in W_i}x[n]
$$

where \(x[n]\) is the ECG signal.

The implementation used a window extending approximately 80 ms on either side of each R-peak. The difference between the maximum and minimum values captures the local voltage range of the QRS complex.

Respiratory movement can change the observed ECG amplitude through changes in cardiac orientation and electrode geometry. Tracking peak-to-peak amplitude across successive beats may therefore reveal respiratory modulation.

However, this feature is not specific to respiration. Noise, body movement, changes in electrode contact, and other variations in the ECG can also affect its value. Its usefulness as an EDR signal must consequently be established through comparison with measured respiration.

### 5.2.2 R-Wave Amplitude

R-wave amplitude measures the height of the detected R-peak relative to an estimated local baseline.

If \(x[r_i]\) is the ECG voltage at the detected R-peak and \(b_i\) is the estimated baseline, the feature is

$$
A_{R,i}=x[r_i]-b_i
$$

The baseline correction is intended to reduce the influence of local voltage offsets, allowing changes in the R-wave height to be tracked across successive beats.

Unlike peak-to-peak amplitude, which considers both the maximum and minimum values within the QRS window, R-wave amplitude focuses on the detected R-peak relative to its baseline.

Respiratory movement may modulate this feature, but its value can also be affected by changes in the local baseline, electrode geometry, noise, and waveform morphology. The initial validation therefore evaluates R-wave amplitude as a separate candidate rather than assuming it is equivalent to peak-to-peak amplitude.

### 5.2.3 QRS Area

QRS area captures the accumulated magnitude of the baseline-corrected ECG within the QRS measurement window.

For the window \(W_i\), the implemented feature is

$$
A_{\mathrm{QRS},i}
=
\sum_{n\in W_i}|x[n]-b_i|
$$

where \(b_i\) denotes the estimated baseline used for the measurement.

Taking the absolute value prevents positive and negative waveform portions from cancelling each other. The feature therefore reflects the accumulated magnitude of the local waveform rather than its signed integral.

Changes in QRS amplitude and shape can produce corresponding changes in this feature. However, the area depends on the measurement window, baseline estimate, sampling frequency, and waveform morphology. It may also respond to non-respiratory changes in the ECG.

The feature was retained as a separate candidate so its measured relationship with respiration could be compared with those of R-wave amplitude and QRS peak-to-peak amplitude.

### Extracted morphology statistics

The following statistics were obtained from the generated morphology features for record `a01`.

| Feature                    | Mean   | Standard deviation |
| -------------------------- | ------ | ------------------ |
| R-wave amplitude           | 1.466  | 0.211              |
| QRS peak-to-peak amplitude | 2.218  | 0.270              |
| QRS area                   | 0.0624 | 0.00440            |

The units depend on the ECG signal units and the feature definition.

The nonzero standard deviations establish that the features vary across heartbeats, but do not demonstrate that the variation is respiratory in origin. Their suitability for EDR must be judged by the measured-reference comparison in Section 5.4.

## 5.3 Inspecting Respiration Channels in the Supplied Respiration Data

The ECG record `a01` contains a single ECG signal. The companion record `a01r` contains the additional respiratory and physiological measurements required for validation.

A local header-inspection script was used to identify the available signals, sampling frequencies, sample counts, and recording durations.

| Record | Sampling frequency | Number of signals | Number of samples | Duration    |
| ------ | ------------------ | ----------------- | ----------------- | ----------- |
| `a01`  | 100 Hz             | 1                 | 2,957,000         | 29,570.00 s |
| `a01r` | 100 Hz             | 4                 | 2,956,796         | 29,567.96 s |

The inspected channel names in `a01r` were:

| Channel  | Measurement       | Relevance to EDR validation                                      |
| -------- | ----------------- | ---------------------------------------------------------------- |
| `Resp C` | Chest effort      | Measures breathing-related chest movement                        |
| `Resp A` | Abdominal effort  | Measures breathing-related abdominal movement                    |
| `Resp N` | Nasal airflow     | Direct respiratory waveform reference for the initial comparison |
| `SpO2`   | Oxygen saturation | Measures a physiological consequence of respiratory disturbance  |

Nasal airflow (`Resp N`) was selected as the reference for the initial EDR validation.

The distinction between airflow and respiratory effort is important. During obstructive apnea, airflow may cease even though chest or abdominal respiratory effort continues. Consequently, agreement with airflow and agreement with respiratory effort describe different physiological relationships. Oxygen saturation was not selected as the primary EDR waveform reference because it does not directly measure respiratory cycles.

The local inspection did not find an `a01er.hea` header. The required channels were nevertheless available in `a01r`, so the initial validation could proceed using the locally available respiratory record.

Although both records report a sampling frequency of 100 Hz, their sample counts and durations differ slightly. The ECG and reference signals must therefore be aligned using their actual timestamps and common recording interval rather than assuming that their sample indices correspond exactly.

## 5.4 Validation and Performance of Each EDR Method

Generating a signal from the ECG does not establish that it represents respiration. The RR-interval and morphology-derived candidates were therefore compared against measured nasal airflow from `a01r`.

The validation script evaluated four candidates:

- RR interval.
- R-wave amplitude.
- QRS peak-to-peak amplitude.
- QRS area.

The recording was divided into 60-second windows with a 30-second step. This produced 984 analysis windows, allowing the relationship between each candidate and measured airflow to be examined over successive portions of the recording.

Two reported metrics were used.

**Median absolute correlation** measures waveform similarity between an EDR candidate and the airflow reference. The script searched for the strongest absolute correlation within a lag of ±5 seconds and permitted polarity inversion.

**Median respiratory-rate error** measures the difference between the respiratory rate estimated from an EDR candidate and that estimated from the airflow reference, expressed in breaths per minute.

### Initial validation results

| EDR candidate              | Median absolute correlation | Median rate error |
| -------------------------- | --------------------------- | ----------------- |
| QRS peak-to-peak amplitude | 0.620                       | 1 breath/min      |
| R-wave amplitude           | 0.613                       | 1 breath/min      |
| QRS area                   | 0.580                       | 2 breaths/min     |
| RR interval                | 0.507                       | 5 breaths/min     |

QRS peak-to-peak amplitude achieved the highest reported median absolute correlation and a median respiratory-rate error of 1 breath/min. R-wave amplitude produced similar results. QRS area performed less strongly on these metrics, while RR-interval EDR had the lowest median absolute correlation and the highest median rate error.

These findings make QRS peak-to-peak amplitude the leading candidate among the four features evaluated in this recording. They suggest that morphology-derived EDR may be more useful than RR-interval EDR for respiratory-rate estimation under the conditions represented by this initial experiment.

The results must nevertheless be interpreted in light of the validation procedure:

- **Lag and polarity selection:** selecting the strongest absolute correlation within ±5 seconds can inflate apparent waveform agreement relative to evaluation at a fixed alignment.
- **Overlapping windows:** the 30-second step means adjacent 60-second windows share data. The 984 windows are not independent observations.
- **Single-record evaluation:** these results come from `a01` and do not establish performance across different subjects or recordings.
- **Respiratory-event dependence:** performance during ordinary breathing does not necessarily predict performance during apnea.
- **Reference dependence:** agreement with nasal airflow does not establish agreement with respiratory effort or demonstrate apnea-detection accuracy.

The reported results are therefore preliminary evidence of relative performance, not a general validation of EDR accuracy.

An additional distinction is necessary when interpreting the respiratory-rate results: a low rate error does not guarantee that the EDR waveform follows the reference accurately throughout the window. Similarly, a high correlation does not guarantee a small respiratory-rate error. Both measurements are useful because they evaluate different aspects of the estimated respiratory signal.

## 5.5 Frequency-Based EDR

Frequency-based analysis estimates respiratory frequency by identifying periodic components in an ECG-derived signal. It is an analysis approach rather than a separate source of ECG information: spectral analysis can be applied to RR intervals, QRS peak-to-peak amplitude, R-wave amplitude, or QRS area.

For example, the uniformly sampled RR-interval signal can be analysed to identify a dominant frequency associated with respiratory modulation. The same procedure can be applied independently to the morphology-derived signals.

If the estimated respiratory frequency is \(f_{\mathrm{resp}}\) in hertz, the corresponding respiratory rate is

$$
RR_{\mathrm{rate}}=60f_{\mathrm{resp}}
\quad \text{breaths/min}
$$

The conversion is valid when the selected frequency represents the respiratory fundamental. A dominant spectral peak is not automatically respiratory: harmonics, noise, other physiological rhythms, and insufficient respiratory modulation can produce misleading estimates.

The existing implementation generates uniform 4 Hz EDR signals suitable for subsequent spectral analysis. However, the reported validation results do not establish a separately evaluated frequency-based EDR algorithm, its spectral settings, or its independent performance. No additional numerical results are therefore attributed to this subsection.

Frequency-based evaluation should be performed separately for the RR-derived and morphology-derived signals, using the measured airflow signal as the reference. This would allow the respiratory-rate estimates to be compared under the same windowing and alignment conditions.

## 5.6 Final Discussion

The EDR experiments established a pipeline for extracting respiratory surrogates from the finalized Detector 4 R-peaks and evaluating them against measured nasal airflow.

Two approaches were implemented. RR-interval EDR used variations in the time between successive heartbeats, while morphology-based EDR used changes in QRS amplitude and shape. The morphology approach produced three candidates: R-wave amplitude, QRS peak-to-peak amplitude, and QRS area.

In the initial comparison of 984 overlapping windows from record `a01`, QRS peak-to-peak amplitude achieved a median absolute correlation of 0.620 and a median respiratory-rate error of 1 breath/min. R-wave amplitude produced a median absolute correlation of 0.613 and the same median rate error. QRS area achieved 0.580 and 2 breaths/min, while RR-interval EDR achieved 0.507 and 5 breaths/min.

On these reported metrics, QRS peak-to-peak amplitude is the strongest candidate for further investigation. This is a preliminary finding from one recording, not evidence that the feature will generalize to other subjects or remain reliable during apnea.

The validation method also allowed the best absolute correlation to be selected within a ±5-second lag range and permitted polarity inversion. This makes the results useful for exploratory comparison but potentially optimistic as measures of waveform agreement. The overlapping windows further limit the interpretation of the number of observations.

Finally, EDR must not be treated as a direct substitute for airflow. Respiratory movement can modulate ECG morphology, but ECG-derived features may continue to vary even when airflow is absent. Reliable respiratory-rate estimation therefore does not, by itself, establish the ability to detect apnea.

The appropriate outcome of this stage is to retain morphology-based EDR—particularly QRS peak-to-peak amplitude—as a promising feature for further evaluation, while preserving RR-interval EDR and the other morphology features as comparison candidates. Frequency-based analysis and evaluation across additional recordings remain separate tasks. The eventual apnea-detection model must be assessed independently using the supplied event annotations and a properly separated evaluation set.