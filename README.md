# Predicting F1 Pit Stops

**Kaggle Playground Series S6E5** — Binary classification: predict whether an F1 driver will pit on the next lap.  
Metric: **AUC-ROC** · [Competition link](https://www.kaggle.com/competitions/playground-series-s6e5/overview)

---

## Results

| Model | OOF AUC |
| --- | --- |
| LightGBM (default) | 0.9389 |
| XGBoost (default) | 0.9385 |
| CatBoost (default) | 0.9384 |
| **Rank Ensemble (LGB + XGB + CAT)** | **0.9399** |
| Tuned + DART + Stacking *(in progress)* | ~0.944+ expected |

> Baseline OOF AUC without external dataset: ~0.930. Adding 76,929 rows from the external F1 strategy dataset is the single biggest boost (+0.010).

---

## Approach

### 1 · Key Findings from EDA

- **2023 anomaly**: 2023 data has a ~1% positive rate vs ~27% in other years — a synthetic generation artifact. `is_year2023` is the most impactful feature by SHAP (mean |SHAP| = 1.178).
- **Train/test share races**: rows are lap-level splits of the same races, so GroupKFold by Race×Year is required to prevent within-race leakage.
- **External dataset overlap**: 24,442 of 101,371 external rows duplicate training rows — deduplicated before use.

### 2 · Feature Engineering (45 features)

Features are computed on a combined train + test + external frame (sorted by Driver/Race/Year/Lap) to allow correct lag and rolling window computation.

| Group | Features |
| --- | --- |
| **Lag** | `LT_lag1/2`, `LTD_lag1`, `TL_lag1`, `PitStop_lag1`, `CD_lag1` |
| **Rolling** | `LT_roll3_mean/std`, `LT_roll5_std`, `LTD_roll3_mean`, `CD_roll3_mean` |
| **Stint** | `NormTyreLife`, `TyreLife_compound_pct`, `tyre_life_vs_field_max`, `Deg_per_lap` |
| **Degradation** | `deg_acceleration` (Δ wear per lap) |
| **Race context** | `EstTotalLaps`, `LapsRemaining`, `LT_race_compound_mean`, `LT_vs_pace` |
| **Interactions** | `LR_x_TL`, `TL_x_Stint`, `RP_x_TL`, `LT_acceleration`, `Deg_x_NormTL`, `LapsRemaining_x_NormTL` |
| **Strategic window** | `laps_rem_vs_compound_max`, `can_finish_on_current` |
| **Flags** | `is_year2023`, `is_pretesting`, `is_real_driver`, `Compound_ord` |
| **Target encodings** | `Driver_TE`, `Race_TE`, `Race_Year_TE` (OOF, computed inside CV folds) |

**SHAP insight**: `Stint` is rank-2 by mean |SHAP| (0.769) but absent from LGB gain importance top-20 — trees split on it slowly. The two strategic window features give the model cleaner decision boundaries.

### 3 · Cross-Validation

`GroupKFold(n_splits=5)` grouped by `Race_Year`. For each fold, the training set is the fold's train rows **plus all external rows** (weighted at 0.8 to account for the different driver encoding scheme).

### 4 · Models

- **LightGBM** — primary model; handles NaN natively, fastest iteration
- **XGBoost** — diversity through different tree construction algorithm
- **CatBoost** — symmetric trees add further ensemble diversity
- **LGB DART** — dropout regularisation, typically +0.001–0.002 over standard DART
- **Hyperparameter tuning**: Optuna TPE sampler, 3-fold CV during search (fast), then 5-fold with best params

### 5 · Ensemble

1. **Rank-average blend** — convert each model's scores to percentile ranks then average (normalises calibration differences between models)
2. **Optimised weighted blend** — `scipy.optimize.minimize` on OOF AUC
3. **Stacking** — logistic regression meta-learner trained on OOF predictions using GroupKFold
4. **Pseudo-labeling** — high-confidence test rows (prob > 0.90 or < 0.05) added to training at weight 0.5; only applied if OOF AUC improves

---

## Project Structure

```text
Predicting-F1-Pit-Stops/
├── src/
│   ├── data.py          # load_train/test/external, dedup_external
│   ├── features.py      # build_features() — single source of truth for all 45 features
│   ├── cv.py            # GroupKFold runner + OOF target encoding
│   ├── models.py        # LGB / XGB / CatBoost / DART factory functions
│   ├── ensemble.py      # rank_avg, weighted blend, stacking, pseudo_label
│   └── utils.py         # save_submission, rank_avg, optimise_weights
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_tuning.ipynb
│   ├── 05_ensemble.ipynb
│   └── kaggle_submission.ipynb   ← self-contained Kaggle upload notebook
├── scripts/
│   ├── run_baseline.py           ← 5-fold CV, all 3 models
│   ├── run_tuning.py             ← Optuna search (30+20+10 trials)
│   ├── run_ensemble.py           ← DART + stacking + pseudo-labeling
│   ├── run_features_v2.py        ← regenerate cache + test v2 features
│   ├── run_shap.py               ← SHAP analysis
│   └── check_results.py          ← quick OOF AUC from cache
├── outputs/
│   ├── shap_summary.png
│   ├── shap_importance.csv
│   └── lgb_importance.csv
├── Dataset/                      ← raw data (gitignored)
├── submissions/                  ← generated CSVs (gitignored)
├── cache/                        ← feature cache .pkl files (gitignored)
└── requirements.txt
```

---

## Reproducing Results

```bash
pip install -r requirements.txt

# 1. Build feature cache
py -3 -c "
import sys; sys.path.insert(0, '.')
from src.data import load_train, load_test, load_external, dedup_external
from src.features import prepare_all
from pathlib import Path
train, test, ext = load_train(), load_test(), load_external()
ext = dedup_external(train, ext)
tr, te, ex = prepare_all(train, test, ext)
Path('cache').mkdir(exist_ok=True)
tr.to_pickle('cache/train_feat.pkl')
te.to_pickle('cache/test_feat.pkl')
ex.to_pickle('cache/ext_feat.pkl')
print('Cache built.')
"

# 2. Baseline CV (≈30 min)
py -3 scripts/run_baseline.py

# 3. Hyperparameter tuning (≈4 hours)
py -3 scripts/run_tuning.py

# 4. Final ensemble + submission
py -3 scripts/run_ensemble.py
```

Or upload `notebooks/kaggle_submission.ipynb` directly to Kaggle for a fully self-contained run.

---

## Dataset

| File | Rows | Description |
| --- | --- | --- |
| `train.csv` | 439,140 | Labeled lap-level telemetry |
| `test.csv` | 188,165 | Unlabeled test rows |
| `f1_strategy_dataset_v4.csv` | 101,371 (76,929 unique) | External supplement — [source](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) |

All data lives in `Dataset/` (gitignored). Download from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e5/data).
