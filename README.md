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

| File                   | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `train.csv`            | Labeled training data with lap-level race telemetry  |
| `test.csv`             | Unlabeled test data for generating predictions       |
| `sample_submission.csv`| Correct submission format (`id`, `PitNextLap`)       |

### Features

| Feature                  | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| `id`                     | Unique row identifier                                   |
| `Driver`                 | Encoded driver identifier                               |
| `Compound`               | Tyre compound in use (HARD, MEDIUM, SOFT, etc.)         |
| `Race`                   | Grand Prix name                                         |
| `Year`                   | Season year (2022-2025)                                 |
| `PitStop`                | Whether the driver pitted on the current lap (0/1)      |
| `LapNumber`              | Current lap number                                      |
| `Stint`                  | Current stint number (resets after each pit stop)       |
| `TyreLife`               | Age of current tyres in laps                            |
| `Position`               | Current race position                                   |
| `LapTime (s)`            | Lap time in seconds                                     |
| `LapTime_Delta`          | Change in lap time vs previous lap                      |
| `Cumulative_Degradation` | Accumulated tyre degradation over the stint             |
| `RaceProgress`           | Fraction of race completed (0.0 to 1.0)                 |
| `Position_Change`        | Change in position since last lap                       |

### Target

| Column       | Type          | Description                            |
| ------------ | ------------- | -------------------------------------- |
| `PitNextLap` | Binary (0/1)  | `1` = driver pits on the next lap      |

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
