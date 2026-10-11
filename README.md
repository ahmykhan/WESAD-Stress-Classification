# Wearable Biosignal-Based Cognitive Stress and Affective State Classifier

A machine learning project that detects **stress vs. relaxed (baseline)** states from wearable physiological signals, using the public **WESAD** dataset. Raw chest-sensor recordings (ECG and EDA) are cleaned, segmented into 5-second windows, and converted into a compact, ML-ready feature table.

> **Project status:** Phase 1 (data exploration) and Phase 2 (preprocessing and feature extraction) are complete. Model training and evaluation are the next phases.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmykhan/WESAD-Stress-Classification/blob/main/AI_project.ipynb)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Roadmap](#project-roadmap)
4. [Repository Structure](#repository-structure)
5. [Phase 1: Data Exploration](#phase-1-data-exploration)
6. [Phase 2: Preprocessing and Feature Extraction](#phase-2-preprocessing-and-feature-extraction)
7. [Output Dataset Schema](#output-dataset-schema)
8. [Getting Started](#getting-started)
9. [Known Limitations and Notes](#known-limitations-and-notes)
10. [Next Steps](#next-steps)
11. [Acknowledgements and Citation](#acknowledgements-and-citation)

---

## Overview

Stress affects both physiology and cognition, and wearable sensors make it possible to measure it continuously and unobtrusively. This project builds a pipeline that:

- Loads raw multimodal biosignals from the WESAD dataset
- Isolates the **Baseline (Relaxed)** and **Stress** conditions
- Compresses 700 Hz raw signals into **5-second windows** with statistical features
- Produces a clean CSV that is ready for classical ML models

## Dataset

**WESAD (Wearable Stress and Affect Detection)** is a multimodal dataset collected from 15 subjects wearing two devices:

| Device | Location | Signals |
|---|---|---|
| RespiBAN | Chest | ECG, EDA, EMG, respiration, temperature, accelerometer (700 Hz) |
| Empatica E4 | Wrist | BVP, EDA, temperature, accelerometer, HR, IBI |

Each subject's `.pkl` file contains synchronized signals plus a label stream:

| Label | Meaning |
|---|---|
| 0 | Not defined / transient |
| 1 | Baseline (Relaxed) |
| 2 | Stress |
| 3 | Amusement |
| 4 | Meditation |

**Subjects used:** S2 to S17, excluding S12 (14 or 15 usable subjects depending on availability; this run processed **15 subjects**: S2 to S11 and S13 to S17). S1 and S12 are not part of the usable WESAD release.

> The dataset is **not included** in this repository because of its size and licensing. See [Getting Started](#getting-started) for download instructions.

## Project Roadmap

- [x] **Phase 1:** Dataset exploration and signal inspection
- [x] **Phase 2:** Preprocessing, windowing, and feature extraction
- [ ] **Phase 3:** Model training (baseline classifiers)
- [ ] **Phase 4:** Evaluation, comparison, and analysis
- [ ] **Phase 5:** Extension to affective states (amusement, meditation) and wrist-sensor signals

## Repository Structure

```
.
├── AI_project.ipynb     # Main notebook (Phases 1 and 2)
├── Data/                # Processed data (placeholder)
├── rawdata/             # Place raw WESAD files here (placeholder)
└── README.md
```

## Phase 1: Data Exploration

Goals of this phase:

- Mount Google Drive in Colab and locate all WESAD files
- Inspect the Empatica E4 CSV files (e.g., `BVP.csv`), where the first row holds the start timestamp and the second row holds the sampling rate
- Inspect the synchronized `.pkl` files and identify the chest-sensor ECG and EDA channels along with the label vector
- Build a proof-of-concept cleaned file for one subject (S2), keeping only Relaxed and Stressed samples

## Phase 2: Preprocessing and Feature Extraction

The pipeline runs across all usable subjects and performs the following steps:

1. **Load** each subject's `.pkl` file (`encoding='latin1'`)
2. **Extract** chest ECG, chest EDA, and the label stream
3. **Filter** to labels `1` (Relaxed) and `2` (Stressed), discarding all other samples
4. **Window** the signal into non-overlapping blocks of **5 seconds**
   (700 Hz × 5 s = **3,500 samples per window**)
5. **Label** each window using the majority (mode) label within it
6. **Compute features** for each signal: mean, standard deviation, maximum, minimum
7. **Export** the combined table to `WESAD_ML_Ready_Features.csv`

### Result

| Metric | Value |
|---|---|
| Subjects processed | 15 |
| Window length | 5 s (3,500 samples at 700 Hz) |
| Total ML-ready rows | **5,504** |
| Features per window | 8 (4 ECG + 4 EDA) |
| Target classes | Relaxed, Stressed |

## Output Dataset Schema

| Column | Type | Description |
|---|---|---|
| `Subject` | string | Subject ID (e.g., `S2`) |
| `State` | string | Target label: `Relaxed` or `Stressed` |
| `ECG_Mean` | float | Mean ECG amplitude in the window |
| `ECG_Std` | float | Standard deviation of ECG |
| `ECG_Max` | float | Maximum ECG value |
| `ECG_Min` | float | Minimum ECG value |
| `EDA_Mean` | float | Mean electrodermal activity (skin conductance) |
| `EDA_Std` | float | Standard deviation of EDA |
| `EDA_Max` | float | Maximum EDA value |
| `EDA_Min` | float | Minimum EDA value |

Example rows:

```
Subject  State    ECG_Mean  ECG_Std   ECG_Max   ECG_Min    EDA_Mean  EDA_Std   EDA_Max   EDA_Min
S2       Relaxed  0.008192  0.129848  0.979294  -0.220459  5.472599  0.107337  5.756760  5.310822
S2       Relaxed -0.000039  0.139631  0.957870  -0.222336  5.237507  0.056015  5.378342  5.145645
```

## Getting Started

### Requirements

- Python 3.8+
- `numpy`, `pandas`
- Google Colab (recommended) or a local Jupyter environment

```bash
pip install numpy pandas jupyter
```

### Steps

1. **Download WESAD** from the [official source](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) and place it so each subject folder contains its `.pkl` file, e.g. `WESAD/S2/S2.pkl`.
2. **Open the notebook** `AI_project.ipynb` in Colab (use the badge above) or locally.
3. **Update the path variables** to match your setup:
   ```python
   drive_path  = '/content/drive/.../WESAD'                       # folder holding S2, S3, ...
   output_file = '/content/drive/MyDrive/WESAD_ML_Ready_Features.csv'
   ```
4. **Run all cells.** The final cell writes `WESAD_ML_Ready_Features.csv`.

## Known Limitations and Notes

These are documented honestly so later phases can account for them:

- **Chest sensor only (so far).** Only ECG and EDA from the chest device are used; wrist signals (BVP, HR, IBI, TEMP, ACC) are not yet included.
- **Simple statistical features.** Mean, std, min, and max do not capture clinically meaningful measures such as heart rate variability or skin conductance response peaks.
- **Filtered-then-windowed.** Windows are cut after removing non-Relaxed/Stressed samples, so a window near a removed segment boundary may stitch together non-adjacent signal. A cleaner approach is to window within each continuous labeled segment.
- **Trailing samples dropped.** The loop bound (`total_rows - window_samples`) can discard the final partial window for each subject.
- **Binary task.** Amusement and Meditation conditions are currently excluded.
- **Subject-dependent signals.** Absolute ECG/EDA levels vary widely across people, so per-subject normalization will likely matter.

## Next Steps

- Normalize features per subject (e.g., z-score against each person's baseline)
- Use **subject-wise splits** (such as Leave-One-Subject-Out cross-validation) rather than random row splits, to avoid leakage between windows of the same person
- Train baseline models: Logistic Regression, Random Forest, SVM, XGBoost
- Report accuracy, precision, recall, F1, and confusion matrices
- Add physiology-aware features (HRV from R-peaks, SCR counts, tonic/phasic EDA)
- Extend to multi-class affect classification and wrist-sensor fusion

## Acknowledgements and Citation

This project uses the WESAD dataset. If you use it, please cite the original paper:

> Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., and Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.* Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI '18).
