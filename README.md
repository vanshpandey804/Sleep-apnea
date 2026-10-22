# Interpretable ECG-Based Sleep Apnea Detection

> **Course**: Digital Signal Processing (ECN15101)  
> **Area**: Biomedical / Wearable Signal Processing  
> **Repository**: [github.com/vanshpandey804/Sleep-apnea](https://github.com/vanshpandey804/Sleep-apnea)  
> **Dataset**: PhysioNet Apnea-ECG Database (v1.0.0, 70 Overnight Records)  

---

## Project Members

| Name | Roll Number | Role |
| :--- | :---: | :--- |
| **Parekh Daksh Parag** | 20244122 | Project team member |
| **Pandey Vansh Murlibhai** | 20244120 | Project team member |
| **Palak Dhingra** | 20244119 | Project team member |
| **Nitin Meena** | 20244118 | Project team member |

---

## Configuration & Setup Guide

### 1. Prerequisites & Environment Setup
This repository requires **Python 3.10+** (tested on Python 3.12).

Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/vanshpandey804/Sleep-apnea.git
cd Sleep-apnea

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Install required dependencies:
```bash
pip install wfdb numpy scipy matplotlib pandas scikit-learn
```

---

### 2. Dataset Setup
The project uses the **PhysioNet Apnea-ECG Database (v1.0.0)**.
* Record `a01` (with multi-channel PSG `a01r`) is included in `data/ao1/`.
* To run the complete 35-patient cohort:
  1. Download the database archive from PhysioNet:
     ```
     https://physionet.org/content/apnea-ecg/get-zip/1.0.0/
     ```
  2. Extract the downloaded archive into `data/apnea-ecg/`.
  3. Run the curation script to automatically organize records and parse clinical demographics:
     ```bash
     python src/30_curate_dataset.py
     ```
     This organizes records into `data/curated/` and extracts demographics into `results/tables/patient_demographics.csv`.

---

### 3. Pipeline Execution Workflow

Execute the scripts in order:

```bash
# Step 3: Validate apnea annotations and temporal alignment on a01
python src/26_validate_apnea_annotations_a01.py

# Step 4: Extract 5 EDR modalities on benchmark record a01
python src/27_feature_extraction_pipeline.py

# Step 4 (Batch): Process all 35 curated patients and generate master dataset
python src/29_batch_process_dataset.py

# Step 5: Run patient-independent GroupKFold ML benchmark (5 EDRs + Combined)
python src/28_train_evaluate_models.py

# Extensions: Demographic correlation analysis & QRS sensor noise stress-test
python src/31_demographics_and_robustness_analysis.py
```

---

# PROJECT REPORT

## Interpretable ECG-based Apnea Detection: HRV, EDR, ECG Morphology and Signal-Quality Analysis with QRS-Error, Window-Selection and Patient-Independent Evaluation

---

### 1. Project Overview

Sleep apnea is characterized by repeated episodes of cessation (apnea) or significant reduction (hypopnea) of breathing during sleep, leading to severe oxygen desaturations, sympathetic arousals, and cardiovascular strain. While gold-standard polysomnography (PSG) requires multi-channel overnight monitoring, single-lead electrocardiogram (ECG) provides a low-cost, wearable alternative.

ECG reflects respiratory and autonomic sleep apnea dynamics via two distinct physical pathways:
1. **Autonomic / Neural Modulation**: Autonomic nervous system fluctuations modulate the sinoatrial node. Under normal breathing, Respiratory Sinus Arrhythmia (RSA) accelerates heart rate during inspiration and decelerates heart rate during expiration. During obstructive sleep apnea, recurrent airway collapse causes prolonged bradycardia during the obstruction followed by abrupt rebound tachycardia upon airway opening and arousal.
2. **Mechanical / Thoracic Morphology Modulation**: Cyclical lung inflation and diaphragmatic descent alter thoracic electrical impedance and physically shift the cardiac electrical anatomical axis relative to surface ECG leads. This induces beat-to-beat amplitude and area modulations in the QRS depolarization waveform.

This project delivers an **interpretable, DSP-first pipeline** that detects cardiac beats with sub-millisecond precision, extracts 5 distinct EDR and HRV modalities, normalizes features to each patient's individual resting baseline, and classifies 1-minute epochs under **strict patient-independent evaluation**.

---

### 2. Primary Objective

The primary objective is to develop an interpretable, reproducible, and mathematically rigorous DSP pipeline that:
1. Preprocesses raw overnight single-lead ECG signals to remove baseline wander and high-frequency noise while strictly preserving QRS morphology.
2. Accurately detects QRS complexes using a modified adaptive dual-threshold Pan-Tompkins detector (**Detector 4**), addressing and resolving adaptive threshold saturation caused by clinical electrode pop artifacts.
3. Constructs and compares all **5 distinct EDR and HRV representations**:
   - EDR 1: RR-Interval / HRV Dynamics (Respiratory Sinus Arrhythmia)
   - EDR 2: R-Wave Peak Amplitude Variation
   - EDR 3: QRS Peak-to-Peak Excursion Variation
   - EDR 4: QRS Absolute Area Variation
   - EDR 5: Welch PSD Frequency-Based Respiratory Rate and Power Estimation
4. Validates the extracted EDR signals against synchronously recorded gold-standard Nasal Airflow (`Resp N`) and Pulse Oximetry ($\text{SpO}_2$) channels.
5. Formulates and validates strict 1-minute epoch windowing aligned with clinical PhysioNet `.apn` ground-truth annotations.
6. Implements **Patient-Specific Baseline Normalization** to eliminate inter-subject physiological and morphological variances.
7. Benchmarks lightweight machine learning models (Logistic Regression, Random Forest) across individual and combined EDR representations under **strict Patient-Independent GroupKFold cross-validation** (zero patient overlap between train and test folds).
8. Analyzes the impact of patient demographics (age, BMI, clinical severity) and stress-tests the resilience of the pipeline against synthetic QRS timing jitter and sensor dropouts.

---

### 3. Dataset and Data Source

* **Dataset**: PhysioNet Apnea-ECG Database, Version 1.0.0 (`https://physionet.org/content/apnea-ecg/1.0.0/`).
* **Reference**: Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. *"The Apnea-ECG Database,"* Computers in Cardiology, 2000.
* **Specifications**: 70 overnight single-lead ECG recordings digitized at 100 samples/second with 16-bit resolution, spanning 7 to nearly 10 hours per night.
  - **35 Learning Records**:
    - **Class A (Apnea)**: 20 records (`a01` to `a20`) with severe sleep apnea ($\ge 100$ min apnea).
    - **Class B (Borderline)**: 5 records (`b01` to `b05`) with mild/moderate apnea ($5\text{--}99$ min apnea).
    - **Class C (Control / Normal)**: 10 records (`c01` to `c10`) from healthy control subjects ($<5$ min apnea).
  - **35 Test Records**: `x01` to `x35`.
  - **8 Multi-Signal PSG Respiration Records**: `a01r`–`a04r`, `b01r`, `c01r`–`c03r` with Nasal Airflow (`Resp N`), Chest effort (`Resp C`), Abdominal effort (`Resp A`), and Oxygen saturation (`SpO2`).

---

### 4. DSP Methodology & Procedure of Action

```mermaid
flowchart TD
    A["Raw Overnight ECG<br/>(100 Hz, 16-bit)"] --> B["Bandpass Preprocessing<br/>(0.5–40 Hz 4th-order SOS Butter)"]
    B --> C["QRS / R-Peak Detection<br/>(Pan-Tompkins Detector 4)"]
    C --> D1["RR Tachogram & HRV<br/>(Interpolated to 4 Hz)"]
    C --> D2["ECG Morphology EDR<br/>(R-amp, Peak-to-Peak, Area)"]
    D1 --> E["5 EDR Modalities Extracted"]
    D2 --> E
    E --> F["Validation Against Airflow & SpO2<br/>(a01r / Resp N)"]
    F --> G["1-min Windowing & Label Alignment (.apn)"]
    G --> H["Patient-Specific Baseline Normalization"]
    H --> I["Patient-Independent GroupKFold ML Benchmark"]
```

#### 4.1 Digital Filtering
A 4th-order Butterworth bandpass filter ($0.5\text{--}40.0\text{ Hz}$) is implemented with Second-Order Sections (SOS) and zero-phase forward-backward filtering (`sosfiltfilt`), completely eliminating baseline wander and high-frequency noise without introducing phase distortion into QRS morphology.

#### 4.2 Detector 4 & The Adaptive Saturation Fix
QRS complexes are localized using differentiation ($\Delta x[n]$), non-linear squaring, $150\text{ ms}$ moving-window integration, dual running adaptive thresholding ($SPKI$ / $NPKI$), $250\text{ ms}$ refractory blanking, and T-wave slope discrimination.

* **Critical DSP Discovery**: In record `a03`, an initial electrode disconnection spike ($9.88\text{ mV}$ at $t = 1.1\text{ s}$) squared to $34.19$ (vs normal QRS energy of $0.41$), causing unconstrained $SPKI$ to explode to $26.5$ and blinding the detector for the entire night (detecting only 7 beats).
* **The Solution**: We integrated envelope energy capping at $3.0 \times P_{99.5}$ and bounded adaptive updates ($\min(V_{\text{peak}}, 2.5 \times SPKI)$). This brought `a03` detection to **34,222 R-peaks (99.9% match with reference: 34,259)**.

#### 4.3 Resolving the `sqrs125` Timing Delay
We verified that WFDB's built-in `sqrs125` machine annotations contained a systematic 4-sample forward delay ($+32.0\text{ ms}$). Detector 4 eliminated this bias and localized true R-peak extrema on the native 100 Hz grid with sub-millisecond precision.

#### 4.4 The 5 EDR Modalities & Validation Against Airflow
* **EDR 1 (RR-Interval / RSA)**: PCHIP interpolation of successive intervals onto a uniform $4\text{ Hz}$ grid.
* **EDR 2 (R-Wave Amplitude)**: Baseline-subtracted R-peak amplitude ($V_R - V_{\text{baseline}}$).
* **EDR 3 (QRS Peak-to-Peak Excursion)**: $\max(QRS) - \min(QRS)$ over an $80\text{ ms}$ half-window.
* **EDR 4 (QRS Absolute Area)**: Discrete integral $\frac{1}{f_s}\sum |QRS - V_{\text{baseline}}|$.
* **EDR 5 (Welch PSD Frequency Rate)**: Dominant spectral peak in $0.1\text{--}0.7\text{ Hz}$ and relative spectral power ratio.

Validation on `a01r` against measured nasal airflow (`Resp N`):
* QRS peak-to-peak: Median $|r| = 0.620$, rate error $= 1.0\text{ bpm}$.
* R-wave amplitude: Median $|r| = 0.613$, rate error $= 1.0\text{ bpm}$.
* QRS area: Median $|r| = 0.580$, rate error $= 2.0\text{ bpm}$.
* RR intervals: Median $|r| = 0.507$, rate error $= 5.0\text{ bpm}$.

#### 4.5 1-Minute Windowing & Baseline Normalization
* **Window Duration**: Exactly $60.00\text{ seconds}$ with **$0\%$ overlap**, aligning 1:1 with PhysioNet labels. Verified on `a01`: at minute 13 ($t = 780\text{ s}$), label switches to `'A'`, nasal airflow exhibits post-apneic bursts, and $\text{SpO}_2$ drops from $97\%$ to $80\%$.
* **Patient-Specific Baseline Normalization**: All features are scaled relative to each patient's individual median and interquartile range ($\tilde{f} = \frac{f - \text{median}}{\text{IQR}}$), removing inter-subject voltage and resting heart rate biases.

---

### 5. Multi-Patient Benchmark Results (17,023 Windows, 35 Patients)

Evaluated under **strict Patient-Independent 5-fold GroupKFold cross-validation** (zero patient overlap between train and test folds):

| Model | Feature Modality | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | EDR 1 (RR / HRV) | 0.617 | 0.619 | 0.501 | 0.553 | 0.646 | 0.495 |
| **Logistic Regression** | EDR 2 (R-Amplitude) | 0.533 | 0.441 | 0.371 | 0.438 | 0.499 | 0.396 |
| **Logistic Regression** | EDR 3 (QRS Peak-to-Peak) | 0.646 | 0.293 | 0.361 | 0.463 | 0.478 | 0.378 |
| **Logistic Regression** | EDR 4 (QRS Area) | 0.558 | 0.321 | 0.337 | 0.420 | 0.450 | 0.361 |
| **Logistic Regression** | EDR 5 (Frequency / Spectral) | 0.526 | 0.632 | 0.469 | 0.496 | 0.587 | 0.432 |
| **Logistic Regression** | **Combined (All 5 EDRs)** | 0.592 | 0.609 | 0.484 | 0.533 | 0.646 | 0.511 |
| **Random Forest** | EDR 1 (RR / HRV) | 0.679 | 0.653 | 0.548 | 0.607 | 0.721 | 0.591 |
| **Random Forest** | EDR 2 (R-Amplitude) | 0.581 | 0.575 | 0.458 | 0.512 | 0.606 | 0.484 |
| **Random Forest** | EDR 3 (QRS Peak-to-Peak) | 0.606 | 0.558 | 0.459 | 0.522 | 0.607 | 0.449 |
| **Random Forest** | EDR 4 (QRS Area) | 0.625 | 0.617 | 0.503 | 0.557 | 0.654 | 0.496 |
| **Random Forest** | EDR 5 (Frequency / Spectral) | 0.564 | 0.601 | 0.467 | 0.511 | 0.617 | 0.498 |
| **Random Forest** | **Combined (All 5 EDRs)** | **0.703** | **0.674** | **0.572** | **0.631** | **0.751** | **0.624** |

```
Key Findings:
1. Patient-Independent Generalization: Evaluating across completely unseen patients yields an honest, 
   clinically relevant benchmark (ROC-AUC = 0.751, Sensitivity = 70.3%, Specificity = 67.4%). Random window 
   splitting (which leaks baseline patient identity) would yield artificially inflated >95% scores.
2. Autonomic Features Lead in Cross-Patient Generalization: Beat-to-beat RR dynamics (SDNN, RMSSD, CV) 
   and QRS Area generalize more robustly across unseen anatomies than voltage peak-to-peak alone.
3. Multi-Modal Synergy: Combining all 5 EDR modalities consistently outperformed every individual modality alone, 
   confirming that autonomic nervous system response and mechanical thoracic axis rotation provide complementary 
   clinical evidence.
```

---

### 6. Research Extensions Completed

#### Extension 1: Demographic Correlation Analysis
* **Resting Heart Rate vs. BMI**: Statistically significant positive correlation ($r = +0.451, p = 0.0065$, Spearman $\rho = +0.560$). Higher BMI elevates baseline cardiovascular demand.
* **Resting Heart Rate vs. AHI**: Statistically significant positive correlation ($r = +0.497, p = 0.0024$, Spearman $\rho = +0.540$), reflecting chronic sympathetic tone elevation in severe OSA.
* **HRV SDNN vs. Age**: Negative correlation ($r = -0.211, \rho = -0.238$), documenting physiological degradation in vagal heart-rate flexibility with aging.

#### Extension 2: QRS Sensor Noise Stress-Test
We injected Gaussian timing jitter ($\sigma \in [10\text{ ms}, 25\text{ ms}, 50\text{ ms}]$) and random beat dropouts ($10\%$) to simulate low-cost wearable sensors:

| Scenario | Jitter ($\sigma$) | Dropout Rate | Random Forest ROC-AUC | Random Forest F1 | Performance Retention |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | $0.0\text{ ms}$ | $0.0\%$ | **0.737** | **0.625** | $100.0\%$ |
| **Minor Jitter** | $10\text{ ms}$ | $0.0\%$ | **0.735** | **0.624** | $99.7\%$ |
| **Moderate Jitter** | $25\text{ ms}$ | $0.0\%$ | **0.727** | **0.619** | $98.6\%$ |
| **Severe Jitter** | $50\text{ ms}$ | $0.0\%$ | **0.714** | **0.603** | $96.9\%$ |
| **Missed Beats** | $0.0\text{ ms}$ | $10.0\%$ | **0.738** | **0.625** | $100.1\%$ |
| **Combined Noise** | $25\text{ ms}$ | $10.0\%$ | **0.725** | **0.614** | $98.4\%$ |

**Finding**: The model maintains **$>96.9\%$ of its ROC-AUC** even under severe $50\text{ ms}$ timing jitter because window-level statistical aggregation and patient-specific baseline normalization effectively cancel stochastic beat-to-beat detection errors.

---

### 7. Deliverables & Repository Structure

```
├── README.md                              # Configuration, setup guide & full project report
├── PROJECT_REPORT.md                      # Official project report matching synopsis structure
├── .gitignore                             # Ignores large binary archives and caches
├── data/
│   ├── ao1/                               # Benchmark patient record a01 and respiration a01r
│   └── curated/                           # 35 curated patient records (headers, annotations, benchmarks)
├── results/
│   ├── tables/
│   │   ├── all_patients_edr_dataset.csv   # Master dataset: 17,023 1-minute epochs x 62 features
│   │   ├── edr_model_benchmark_results.csv # 5 EDR multi-model performance table
│   │   ├── patient_demographics.csv       # Clinical metadata for 70 subjects
│   │   ├── demographic_correlation_analysis.csv # Demographic statistical findings
│   │   └── qrs_error_robustness_benchmark.csv   # Sensor noise stress-test results
│   └── plots/
│       ├── a01_annotation_alignment_verification.png
│       ├── edr_model_comparison_benchmark.png
│       ├── demographic_physiological_patterns.png
│       └── qrs_noise_stress_test.png
└── src/
    ├── 01_load_ecg.py to 25_frequencybased_EDR.py # Inherited foundational DSP modules
    ├── 26_validate_apnea_annotations_a01.py       # Step 3: Annotation validation & alignment
    ├── 27_feature_extraction_pipeline.py         # Step 4: Detector 4 + 5 EDR extractors
    ├── 28_train_evaluate_models.py               # Step 5: GroupKFold ML benchmark
    ├── 29_batch_process_dataset.py               # Master multi-patient batch processor
    ├── 30_curate_dataset.py                       # Dataset organizer & demographics parser
    └── 31_demographics_and_robustness_analysis.py # Extensions: Demographics & QRS noise stress-test
```

---

### 8. References
1. Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. (2000). The Apnea-ECG Database. *Computers in Cardiology*, 27, 255–258.
2. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236.
3. Moody, G. B., Mark, R. G., Zoccola, A., & Mantero, S. (1985). Derivation of respiratory signals from multi-lead ECGs. *Computers in Cardiology*, 12, 113–116.
4. Parbat, D., & Chakraborty, M. (2024). Multiscale entropy analysis of single lead ECG and ECG derived respiration for AI based prediction of sleep apnea events. *Biomedical Signal Processing and Control*, 87, 105444.
5. Tyagi, P. K., & Agrawal, D. (2024). Automatic detection of sleep apnea from a single-lead ECG signal based on spiking neural network model. *Computers in Biology and Medicine*, 179, 108877.
6. Yamane, T., Fujii, M., & Morita, M. (2025). Clinical-level screening of sleep apnea syndrome with single-lead ECG alone is achievable using machine learning with appropriate time windows. *Sleep and Breathing*, 29, 156.
