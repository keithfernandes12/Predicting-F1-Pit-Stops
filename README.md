# Predicting F1 Pit Stops

## Kaggle Playground Series - Season 6, Episode 5

[Competition Link](https://www.kaggle.com/competitions/playground-series-s6e5/overview)

---

## Overview

This competition challenges participants to predict whether an F1 driver will make a pit stop on the next lap, given real-time race telemetry and tyre data. It is a **binary classification** problem drawn from real Formula 1 race data spanning 2022-2025.

---

## Goal

Predict `PitNextLap` - whether a driver will pit on the following lap (`1`) or not (`0`) - for each row in the test set.

---

## Dataset

### Competition Data (Kaggle)

| File                    | Description                                         |
| ----------------------- | --------------------------------------------------- |
| `train.csv`             | Labeled training data with lap-level race telemetry |
| `test.csv`              | Unlabeled test data for generating predictions      |
| `sample_submission.csv` | Correct submission format (`id`, `PitNextLap`)      |

### External Dataset

- `f1_strategy_dataset_v4.csv` - [F1 Strategy Dataset - Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction)

An external F1 strategy dataset used to supplement training. It shares the same schema as the competition data and uses real driver abbreviations (e.g., `ALB`, `VER`) instead of encoded IDs. It also includes an additional `Normalized_TyreLife` feature not present in the competition files.

### Features

| Feature                  | Description                                        | External Dataset |
| ------------------------ | -------------------------------------------------- | :--------------: |
| `id`                     | Unique row identifier                              | No               |
| `Driver`                 | Driver identifier (encoded in competition data)    | Yes              |
| `Compound`               | Tyre compound in use (HARD, MEDIUM, SOFT, etc.)    | Yes              |
| `Race`                   | Grand Prix name                                    | Yes              |
| `Year`                   | Season year                                        | Yes              |
| `PitStop`                | Whether the driver pitted on the current lap (0/1) | Yes              |
| `LapNumber`              | Current lap number                                 | Yes              |
| `Stint`                  | Current stint number (resets after each pit stop)  | Yes              |
| `TyreLife`               | Age of current tyres in laps                       | Yes              |
| `Normalized_TyreLife`    | Tyre life normalized within the stint              | Yes (only)       |
| `Position`               | Current race position                              | Yes              |
| `LapTime (s)`            | Lap time in seconds                                | Yes              |
| `LapTime_Delta`          | Change in lap time vs previous lap                 | Yes              |
| `Cumulative_Degradation` | Accumulated tyre degradation over the stint        | Yes              |
| `RaceProgress`           | Fraction of race completed (0.0 to 1.0)            | Yes              |
| `Position_Change`        | Change in position since last lap                  | Yes              |

### Target

| Column       | Type         | Description                       |
| ------------ | ------------ | --------------------------------- |
| `PitNextLap` | Binary (0/1) | `1` = driver pits on the next lap |

---

## Evaluation

Submissions are evaluated on **Area Under the ROC Curve (AUC-ROC)** between the predicted probability and the observed `PitNextLap` label.

---

## Approach

- Exploratory data analysis of tyre degradation, lap time trends, and stint patterns
- Feature engineering (e.g., rolling lap time trends, tyre age thresholds)
- Gradient boosted tree models (XGBoost / LightGBM)
- Threshold tuning and cross-validation strategy

---

## Project Structure

```text
Predicting-F1-Pit-Stops/
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── README.md
└── .gitignore
```
