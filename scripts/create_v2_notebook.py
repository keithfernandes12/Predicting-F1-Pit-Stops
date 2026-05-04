"""Generate notebooks/predicting-f1-pit-stops-v2.ipynb"""
import json, uuid
from pathlib import Path

def uid():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uid(),
            "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(md(
"""# F1 Pit Stop Prediction V2 — Kaggle Playground Series S6E5

Improvements over V1:
1. **6 new features** — stint history, 5-lap position trend, tyre vs field median, driver-compound pace, degradation jerk, stint fraction
2. **DART Optuna tuning** — separate study (was fixed params in V1)
3. **2× Optuna trials** — 40 LGB / 20 XGB / 15 CAT / 12 DART
4. **Dynamic n_estimators** — full-train uses mean early-stopping round from CV folds
5. **MLP model** — adds neural-network diversity to ensemble
6. **LightGBM stacker** — replaces logistic regression meta-learner
7. **2-round iterative pseudo-labeling** — progressively relaxed thresholds
8. **Isotonic calibration** — corrects over/under-confidence in final blend"""
))

# ── Cell 1: Install ────────────────────────────────────────────────────────────
cells.append(code(
"""import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'optuna', '-q'], check=False)"""
))

# ── Cell 2: Imports ────────────────────────────────────────────────────────────
cells.append(code(
"""import warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print('Libraries loaded.')"""
))

# ── Cell 3: Configuration ──────────────────────────────────────────────────────
cells.append(md("## Configuration"))
cells.append(code(
"""INPUT_DIR = Path('/kaggle/input/competitions/playground-series-s6e5')
EXT_DIRS  = [
    Path('/kaggle/input/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction'),
]

N_SPLITS   = 5
SEED       = 42
EXT_WEIGHT = 0.8

# V2: doubled trial counts vs V1
RUN_OPTUNA   = True
N_LGB_TRIALS = 40
N_XGB_TRIALS = 20
N_CAT_TRIALS = 15
N_DART_TRIALS= 12

TARGET = 'PitNextLap'
ID_COL = 'id'"""
))

# ── Cell 4: Data loading ───────────────────────────────────────────────────────
cells.append(md("## 1 · Data Loading"))
cells.append(code(
"""train = pd.read_csv(INPUT_DIR / 'train.csv')
test  = pd.read_csv(INPUT_DIR / 'test.csv')
sub   = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
print(f'Train: {train.shape}  Test: {test.shape}')
print(f'Positive rate: {train[TARGET].mean():.3%}')"""
))

cells.append(code(
"""ext = None
for d in EXT_DIRS:
    for fname in ['f1_strategy_dataset_v4.csv', 'f1_strategy_dataset.csv']:
        p = d / fname
        if p.exists():
            ext = pd.read_csv(p)
            print(f'External loaded: {ext.shape}')
            break
    if ext is not None:
        break

if ext is None:
    print('External dataset not found — running without it.')"""
))

cells.append(code(
"""COMMON_COLS = [
    'id', 'Driver', 'Race', 'Year', 'Compound', 'PitStop',
    'LapNumber', 'Stint', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change',
]

def dedup_external(train_df, ext_df):
    key_cols = ['Driver', 'Race', 'Year', 'LapNumber']
    train_keys = set(zip(train_df['Driver'], train_df['Race'],
                         train_df['Year'], train_df['LapNumber']))
    mask = [(d, r, y, l) not in train_keys
            for d, r, y, l in zip(ext_df['Driver'], ext_df['Race'],
                                   ext_df['Year'], ext_df['LapNumber'])]
    return ext_df[mask].copy()

if ext is not None:
    ext = dedup_external(train, ext)
    ext[ID_COL] = range(-1, -len(ext) - 1, -1)
    print(f'External after dedup: {len(ext):,} rows')"""
))

# ── Cell 5: Feature Engineering V2 ────────────────────────────────────────────
cells.append(md("## 2 · Feature Engineering V2 (+6 new features)"))
cells.append(code(
"""_COMPOUND_ORD = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
_COMPOUND_MAX = {'SOFT': 20, 'MEDIUM': 30, 'HARD': 40, 'INTERMEDIATE': 25, 'WET': 20}

def build_features(df):
    df = df.sort_values(['Driver', 'Race', 'Year', 'LapNumber']).copy()
    g  = df.groupby(['Driver', 'Race', 'Year'], sort=False)

    # A: Lag features
    df['LT_lag1']      = g['LapTime (s)'].shift(1)
    df['LT_lag2']      = g['LapTime (s)'].shift(2)
    df['LTD_lag1']     = g['LapTime_Delta'].shift(1)
    df['TL_lag1']      = g['TyreLife'].shift(1)
    df['PitStop_lag1'] = g['PitStop'].shift(1)
    df['CD_lag1']      = g['Cumulative_Degradation'].shift(1)

    # A2: Rolling features
    df['LT_roll3_mean'] = g['LapTime (s)'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['LT_roll3_std']  = g['LapTime (s)'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std())
    df['LT_roll5_std']  = g['LapTime (s)'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std())
    df['LTD_roll3_mean']= g['LapTime_Delta'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['CD_roll3_mean'] = g['Cumulative_Degradation'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # B: Stint features
    stint_max = df.groupby(['Driver','Race','Year','Stint'])['TyreLife'].transform('max')
    df['NormTyreLife']         = df['TyreLife'] / stint_max.clip(lower=1)
    df['Deg_per_lap']          = df['Cumulative_Degradation'] / df['TyreLife'].clip(lower=1)
    compound_max               = df['Compound'].map(_COMPOUND_MAX).fillna(25)
    df['TyreLife_compound_pct']= df['TyreLife'] / compound_max

    # B2: Field-wide tyre context
    df['field_max_tyre_life']    = df.groupby(['Race','Year','Compound'])['TyreLife'].transform('max')
    df['tyre_life_vs_field_max'] = df['TyreLife'] / df['field_max_tyre_life'].clip(lower=1)
    df['deg_acceleration']       = df['Cumulative_Degradation'] - df['CD_lag1']

    # C: Race context
    df['EstTotalLaps']          = (df['LapNumber'] / df['RaceProgress'].clip(lower=0.01)).round()
    df['LapsRemaining']         = df['EstTotalLaps'] - df['LapNumber']
    df['LapsRemaining_clip']    = df['LapsRemaining'].clip(lower=0)
    df['LT_race_compound_mean'] = df.groupby(['Race','Year','Compound'])['LapTime (s)'].transform('mean')
    df['LT_vs_pace']            = df['LapTime (s)'] - df['LT_race_compound_mean']

    # D: Interactions
    df['TL_x_Stint']             = df['TyreLife'] * df['Stint']
    df['RP_x_TL']                = df['RaceProgress'] * df['TyreLife']
    df['LR_x_TL']                = df['LapsRemaining'] * df['TyreLife']
    df['LT_acceleration']        = df['LapTime_Delta'] - df['LTD_lag1']
    df['Deg_x_NormTL']           = df['Cumulative_Degradation'] * df['NormTyreLife']
    df['LapsRemaining_x_NormTL'] = df['LapsRemaining_clip'] * df['NormTyreLife']

    # E: Position trend
    df['Position_lag3']  = g['Position'].shift(3)
    df['position_trend'] = df['Position'] - df['Position_lag3']

    # F: Strategic window
    df['laps_rem_vs_compound_max'] = df['LapsRemaining_clip'] - compound_max
    df['can_finish_on_current']    = (df['laps_rem_vs_compound_max'] <= 0).astype('int8')

    # G: Flags
    df['is_year2023']    = (df['Year'] == 2023).astype('int8')
    df['is_pretesting']  = (df['Race'] == 'Pre-Season Testing').astype('int8')
    df['is_real_driver'] = (~df['Driver'].astype(str).str.match(r'^D\\d+$')).astype('int8')
    df['Compound_ord']   = df['Compound'].map(_COMPOUND_ORD).fillna(5).astype('int8')

    # ── V2 NEW FEATURES ─────────────────────────────────────────────────────

    # H: Pit stops completed so far in race (cumulative PitStop, excluding current lap)
    df['PitStops_so_far'] = g['PitStop'].transform('cumsum') - df['PitStop']

    # I: 5-lap position trend + position vs 3-lap rolling mean
    df['Position_lag5']  = g['Position'].shift(5)
    df['position_trend5']= df['Position'] - df['Position_lag5']
    df['pos_roll3_mean'] = g['Position'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['pos_vs_roll3']   = df['Position'] - df['pos_roll3_mean']

    # J: Tyre life vs field median (complements vs-field-max)
    df['field_med_tyre_life']   = df.groupby(['Race','Year','Compound'])['TyreLife'].transform('median')
    df['tyre_life_vs_field_med']= df['TyreLife'] / df['field_med_tyre_life'].clip(lower=1)

    # K: Lap time vs driver's own compound average for this race
    df['driver_rc_mean_lt'] = df.groupby(['Driver','Race','Year','Compound'])['LapTime (s)'].transform('mean')
    df['LT_vs_own_pace']    = df['LapTime (s)'] - df['driver_rc_mean_lt']

    # L: Degradation jerk (2nd derivative of cumulative degradation)
    df['CD_lag2']    = g['Cumulative_Degradation'].shift(2)
    df['deg_accel_lag1'] = g['deg_acceleration'].shift(1)
    df['deg_jerk']   = df['deg_acceleration'] - df['deg_accel_lag1']

    # M: Stint fraction (which stint out of total stints driver will do)
    total_stints       = df.groupby(['Driver','Race','Year'])['Stint'].transform('max')
    df['StintFraction']= df['Stint'] / total_stints.clip(lower=1)

    return df


FEATURE_COLS = [
    'TyreLife', 'Stint', 'LapNumber', 'Position', 'RaceProgress',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'Position_Change', 'PitStop',
    # lag / rolling
    'LT_lag1', 'LT_lag2', 'LTD_lag1', 'TL_lag1', 'PitStop_lag1', 'CD_lag1',
    'LT_roll3_mean', 'LT_roll3_std', 'LT_roll5_std', 'LTD_roll3_mean', 'CD_roll3_mean',
    # stint
    'NormTyreLife', 'TyreLife_compound_pct', 'Deg_per_lap', 'tyre_life_vs_field_max',
    # degradation
    'deg_acceleration',
    # race context
    'EstTotalLaps', 'LapsRemaining', 'LapsRemaining_clip',
    'LT_race_compound_mean', 'LT_vs_pace',
    # interactions
    'TL_x_Stint', 'RP_x_TL', 'LR_x_TL',
    'LT_acceleration', 'Deg_x_NormTL', 'LapsRemaining_x_NormTL',
    # position / strategy
    'position_trend', 'laps_rem_vs_compound_max', 'can_finish_on_current',
    # flags
    'is_year2023', 'is_pretesting', 'is_real_driver', 'Compound_ord', 'Year',
    # ── V2 new ──────────────────────────────────────────────────────────────
    'PitStops_so_far',
    'position_trend5', 'pos_vs_roll3',
    'tyre_life_vs_field_med',
    'LT_vs_own_pace',
    'deg_jerk',
    'StintFraction',
]
print(f'Feature set: {len(FEATURE_COLS)} base (+3 TE = {len(FEATURE_COLS)+3} total)')"""
))

# ── Cell 6: Build features ─────────────────────────────────────────────────────
cells.append(code(
"""%%time
print('Building features on combined frame...')

train_ids = set(train[ID_COL])
test_ids  = set(test[ID_COL])

train_no_target = train.drop(columns=[TARGET], errors='ignore')
frames = [train_no_target[COMMON_COLS], test[COMMON_COLS]]
ext_ids = set()
if ext is not None:
    ext_common = ext[[c for c in COMMON_COLS if c in ext.columns]].copy()
    if ID_COL not in ext_common.columns:
        ext_common[ID_COL] = ext[ID_COL].values
    ext_ids = set(ext[ID_COL])
    frames.append(ext_common)

all_df = pd.concat(frames, ignore_index=True)
all_df = build_features(all_df)

train_feat = all_df[all_df[ID_COL].isin(train_ids)].merge(
    train[[ID_COL, TARGET]], on=ID_COL, how='left')
test_feat = all_df[all_df[ID_COL].isin(test_ids)]

if ext is not None:
    ext_feat = all_df[all_df[ID_COL].isin(ext_ids)].merge(
        ext[[ID_COL, TARGET]], on=ID_COL, how='left')
else:
    ext_feat = pd.DataFrame(columns=train_feat.columns)

feature_cols = [c for c in FEATURE_COLS if c in train_feat.columns]
print(f'train_feat: {train_feat.shape}  ext_feat: {ext_feat.shape}  test_feat: {test_feat.shape}')
print(f'Active features: {len(feature_cols)}')"""
))

# ── Cell 7: Target encoding ────────────────────────────────────────────────────
cells.append(md("## 3 · Out-of-Fold Target Encoding"))
cells.append(code(
"""def add_target_encoding(train_feat, ext_feat, test_feat, n_splits=N_SPLITS):
    groups      = train_feat['Race'].astype(str) + '_' + train_feat['Year'].astype(str)
    gkf         = GroupKFold(n_splits=n_splits)
    global_mean = train_feat[TARGET].mean()

    for col in ['Driver_TE', 'Race_TE', 'Race_Year_TE']:
        train_feat[col] = global_mean
        if len(ext_feat): ext_feat[col] = global_mean
        test_feat[col]  = global_mean

    for _, (tr_idx, val_idx) in enumerate(gkf.split(train_feat, train_feat[TARGET], groups)):
        tr = train_feat.iloc[tr_idx]
        driver_map = tr.groupby('Driver')[TARGET].mean()
        race_map   = tr.groupby('Race')[TARGET].mean()
        ry_map     = tr.groupby(['Race', 'Year'])[TARGET].mean()
        val = train_feat.iloc[val_idx]
        train_feat.loc[val.index, 'Driver_TE'] = val['Driver'].map(driver_map).fillna(global_mean).values
        train_feat.loc[val.index, 'Race_TE']   = val['Race'].map(race_map).fillna(global_mean).values
        ry = val[['Race','Year']].merge(ry_map.rename('_te').reset_index(),
                                        on=['Race','Year'], how='left')['_te'].fillna(global_mean).values
        train_feat.loc[val.index, 'Race_Year_TE'] = ry

    full_driver_map = train_feat.groupby('Driver')[TARGET].mean()
    full_race_map   = train_feat.groupby('Race')[TARGET].mean()
    full_ry_map     = train_feat.groupby(['Race','Year'])[TARGET].mean()

    for df in [test_feat, ext_feat]:
        if len(df) == 0: continue
        df['Driver_TE'] = df['Driver'].map(full_driver_map).fillna(global_mean).values
        df['Race_TE']   = df['Race'].map(full_race_map).fillna(global_mean).values
        ry = df[['Race','Year']].merge(full_ry_map.rename('_te').reset_index(),
                                       on=['Race','Year'], how='left')['_te'].fillna(global_mean).values
        df['Race_Year_TE'] = ry
    return train_feat, ext_feat, test_feat

print('Computing OOF target encodings...')
train_feat, ext_feat, test_feat = add_target_encoding(train_feat, ext_feat, test_feat)
te_cols = [c for c in ['Driver_TE','Race_TE','Race_Year_TE'] if c in train_feat.columns]
feature_cols = feature_cols + te_cols
y_true = train_feat[TARGET].values
print(f'Total features: {len(feature_cols)}')"""
))

# ── Cell 8: CV helper ──────────────────────────────────────────────────────────
cells.append(md("## 4 · Cross-Validation Helper"))
cells.append(code(
"""groups_train = train_feat['Race'].astype(str) + '_' + train_feat['Year'].astype(str)

ext_X = ext_feat[feature_cols].values if len(ext_feat) > 0 else np.zeros((0, len(feature_cols)))
ext_y = ext_feat[TARGET].values       if len(ext_feat) > 0 else np.array([])
ext_w = np.full(len(ext_feat), EXT_WEIGHT)

def run_cv(model_fn, n_splits=N_SPLITS, verbose=True, track_iters=False):
    \"\"\"Returns oof predictions, fold AUCs, and optionally best_iterations list.\"\"\"
    gkf   = GroupKFold(n_splits=n_splits)
    oof   = np.zeros(len(train_feat))
    aucs  = []
    iters = []

    for fold, (tr_idx, val_idx) in enumerate(
            gkf.split(train_feat, y_true, groups_train)):
        tr_rows  = train_feat.iloc[tr_idx]
        val_rows = train_feat.iloc[val_idx]

        X_tr = np.concatenate([tr_rows[feature_cols].values, ext_X])
        y_tr = np.concatenate([tr_rows[TARGET].values,       ext_y])
        w_tr = np.concatenate([np.ones(len(tr_rows)),        ext_w])
        X_val, y_val = val_rows[feature_cols].values, val_rows[TARGET].values

        model = model_fn(X_tr, y_tr, w_tr, X_val, y_val)
        preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        if verbose:
            print(f'  Fold {fold+1}/{n_splits} | AUC={auc:.5f}')

        if track_iters:
            bi = getattr(model, 'best_iteration_', None)
            if bi is None:
                bi = getattr(model, 'best_iteration', None)
            if bi is not None:
                iters.append(int(bi))

    mean_auc = np.mean(aucs)
    if verbose:
        print(f'  Mean CV AUC: {mean_auc:.5f}  (std={np.std(aucs):.5f})')
        if iters:
            print(f'  Best iter avg: {int(np.mean(iters))} (range {min(iters)}–{max(iters)})')
    return oof, aucs, iters


def _run3(model_fn):
    _, aucs, _ = run_cv(model_fn, n_splits=3, verbose=False, track_iters=False)
    return float(np.mean(aucs))

def rank_avg(arrays):
    ranks = np.column_stack([pd.Series(a).rank(pct=True).values for a in arrays])
    return ranks.mean(axis=1)

print('CV helpers ready.')"""
))

# ── Cell 9: Optuna tuning ──────────────────────────────────────────────────────
cells.append(md("## 5 · Hyperparameter Tuning (Optuna) — V2: 2× trials + DART study"))
cells.append(code(
"""lgb_best = {
    'num_leaves': 255, 'learning_rate': 0.05, 'min_child_samples': 20,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
}
xgb_best = {
    'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 5, 'gamma': 0.1,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
}
cat_best = {
    'depth': 7, 'learning_rate': 0.05, 'l2_leaf_reg': 3,
    'subsample': 0.8, 'colsample_bylevel': 0.8,
}
dart_best = {
    'num_leaves': 255, 'learning_rate': 0.05, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'drop_rate': 0.1, 'skip_drop': 0.5,
    'subsample_freq': 1,
}

if RUN_OPTUNA:
    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f'=== LightGBM Optuna ({N_LGB_TRIALS} trials) ===')
    t0 = time.time()

    def lgb_obj(trial):
        p = {
            'num_leaves':        trial.suggest_int('num_leaves', 63, 511),
            'learning_rate':     trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 10.0),
            'subsample_freq': 1,
        }
        def fn(X_tr, y_tr, w_tr, X_val, y_val):
            m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=3000,
                                   random_state=SEED, n_jobs=-1, verbose=-1, **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            return m
        return _run3(fn)

    lgb_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    lgb_study.optimize(lgb_obj, n_trials=N_LGB_TRIALS, show_progress_bar=True)
    lgb_best = lgb_study.best_params; lgb_best['subsample_freq'] = 1
    print(f'Best LGB: {lgb_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
    print(f'Params: {lgb_best}')

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f'\\n=== XGBoost Optuna ({N_XGB_TRIALS} trials) ===')
    t0 = time.time()

    def xgb_obj(trial):
        p = {
            'max_depth':        trial.suggest_int('max_depth', 5, 12),
            'learning_rate':    trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 10.0),
        }
        def fn(X_tr, y_tr, w_tr, X_val, y_val):
            m = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                  n_estimators=3000, random_state=SEED, n_jobs=-1,
                                  tree_method='hist', verbosity=0,
                                  early_stopping_rounds=50, **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
            return m
        return _run3(fn)

    xgb_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    xgb_study.optimize(xgb_obj, n_trials=N_XGB_TRIALS, show_progress_bar=True)
    xgb_best = xgb_study.best_params
    print(f'Best XGB: {xgb_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
    print(f'Params: {xgb_best}')

    # ── CatBoost ──────────────────────────────────────────────────────────────
    print(f'\\n=== CatBoost Optuna ({N_CAT_TRIALS} trials) ===')
    t0 = time.time()

    def cat_obj(trial):
        p = {
            'depth':             trial.suggest_int('depth', 5, 9),
            'learning_rate':     trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
            'l2_leaf_reg':       trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        }
        def fn(X_tr, y_tr, w_tr, X_val, y_val):
            m = CatBoostClassifier(iterations=3000, eval_metric='AUC', random_seed=SEED,
                                   verbose=0, early_stopping_rounds=50, task_type='CPU', **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val),
                  use_best_model=True, verbose=False)
            return m
        return _run3(fn)

    cat_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    cat_study.optimize(cat_obj, n_trials=N_CAT_TRIALS, show_progress_bar=True)
    cat_best = cat_study.best_params
    print(f'Best CAT: {cat_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
    print(f'Params: {cat_best}')

    # ── LGB DART (V2: now tuned) ───────────────────────────────────────────────
    print(f'\\n=== LGB DART Optuna ({N_DART_TRIALS} trials) ===')
    t0 = time.time()

    def dart_obj(trial):
        p = {
            'num_leaves':        trial.suggest_int('num_leaves', 63, 511),
            'learning_rate':     trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'drop_rate':         trial.suggest_float('drop_rate', 0.05, 0.3),
            'skip_drop':         trial.suggest_float('skip_drop', 0.3, 0.7),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 10.0),
            'subsample_freq': 1,
            'boosting_type': 'dart',
        }
        n_est = trial.suggest_int('n_estimators', 300, 800)
        def fn(X_tr, y_tr, w_tr, X_val, y_val):
            m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=n_est,
                                   random_state=SEED, n_jobs=-1, verbose=-1, **p)
            m.fit(X_tr, y_tr, sample_weight=w_tr, callbacks=[lgb.log_evaluation(-1)])
            return m
        return _run3(fn)

    dart_study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
    dart_study.optimize(dart_obj, n_trials=N_DART_TRIALS, show_progress_bar=True)
    dp = dart_study.best_params
    dart_best = {k: dp[k] for k in dp if k != 'n_estimators'}
    dart_best['boosting_type'] = 'dart'
    dart_n_est = dp.get('n_estimators', 600)
    print(f'Best DART: {dart_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
    print(f'Params: {dart_best}  n_estimators={dart_n_est}')
else:
    dart_n_est = 600
    print('Optuna skipped — using pre-set parameters.')"""
))

# ── Cell 10: 5-fold CV with best params ───────────────────────────────────────
cells.append(md("## 6 · Full 5-Fold CV with Best Parameters"))
cells.append(code(
"""# ── LightGBM ──────────────────────────────────────────────────────────────────
print('=== LightGBM 5-fold CV ===')
def lgb_fn(X_tr, y_tr, w_tr, X_val, y_val):
    m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=3000,
                           random_state=SEED, n_jobs=-1, verbose=-1, **lgb_best)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    return m

lgb_oof, lgb_fold_aucs, lgb_iters = run_cv(lgb_fn, track_iters=True)
lgb_n_est = int(np.mean(lgb_iters)) if lgb_iters else 1500
print(f'LGB OOF AUC: {roc_auc_score(y_true, lgb_oof):.5f}  | avg best iter: {lgb_n_est}')"""
))

cells.append(code(
"""# ── XGBoost ───────────────────────────────────────────────────────────────────
print('=== XGBoost 5-fold CV ===')
def xgb_fn(X_tr, y_tr, w_tr, X_val, y_val):
    m = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_estimators=3000,
                          random_state=SEED, n_jobs=-1, tree_method='hist', verbosity=0,
                          early_stopping_rounds=100, **xgb_best)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m

xgb_oof, xgb_fold_aucs, xgb_iters = run_cv(xgb_fn, track_iters=True)
xgb_n_est = int(np.mean(xgb_iters)) if xgb_iters else 1500
print(f'XGB OOF AUC: {roc_auc_score(y_true, xgb_oof):.5f}  | avg best iter: {xgb_n_est}')"""
))

cells.append(code(
"""# ── CatBoost ──────────────────────────────────────────────────────────────────
print('=== CatBoost 5-fold CV ===')
def cat_fn(X_tr, y_tr, w_tr, X_val, y_val):
    m = CatBoostClassifier(iterations=3000, eval_metric='AUC', random_seed=SEED,
                           verbose=0, task_type='CPU', early_stopping_rounds=100, **cat_best)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_val, y_val),
          use_best_model=True, verbose=False)
    return m

cat_oof, cat_fold_aucs, cat_iters = run_cv(cat_fn, track_iters=True)
cat_n_est = int(np.mean(cat_iters)) if cat_iters else 1500
print(f'CAT OOF AUC: {roc_auc_score(y_true, cat_oof):.5f}  | avg best iter: {cat_n_est}')"""
))

cells.append(code(
"""# ── LGB DART (V2: tuned params) ───────────────────────────────────────────────
print('=== LGB DART 5-fold CV ===')
def dart_fn(X_tr, y_tr, w_tr, X_val, y_val):
    m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=dart_n_est,
                           random_state=SEED, n_jobs=-1, verbose=-1, **dart_best)
    m.fit(X_tr, y_tr, sample_weight=w_tr, callbacks=[lgb.log_evaluation(-1)])
    return m

dart_oof, dart_fold_aucs, _ = run_cv(dart_fn)
print(f'DART OOF AUC: {roc_auc_score(y_true, dart_oof):.5f}')"""
))

cells.append(code(
"""# ── MLP (V2: new model) ───────────────────────────────────────────────────────
print('=== MLP 5-fold CV ===')

def mlp_fn(X_tr, y_tr, w_tr, X_val, y_val):
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_tr)
    Xs_val = scaler.transform(X_val)
    m = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=SEED,
        batch_size=2048,
    )
    m.fit(Xs_tr, y_tr)
    # Wrap to carry scaler for predict_proba
    m._scaler = scaler
    return m

class _ScaledMLP:
    \"\"\"Thin wrapper so run_cv can call predict_proba on scaled data.\"\"\"
    def __init__(self, mlp, scaler):
        self._mlp    = mlp
        self._scaler = scaler
    def predict_proba(self, X):
        return self._mlp.predict_proba(self._scaler.transform(X))

def mlp_fn(X_tr, y_tr, w_tr, X_val, y_val):
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_tr)
    m = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', solver='adam',
        learning_rate_init=0.001, max_iter=200,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15, random_state=SEED, batch_size=2048,
    )
    m.fit(Xs_tr, y_tr)
    return _ScaledMLP(m, scaler)

mlp_oof, mlp_fold_aucs, _ = run_cv(mlp_fn)
print(f'MLP OOF AUC: {roc_auc_score(y_true, mlp_oof):.5f}')"""
))

# ── Cell 11: Full-train models ─────────────────────────────────────────────────
cells.append(md("## 7 · Full-Train Models (dynamic n_estimators from CV)"))
cells.append(code(
"""%%time
print('Training full models on all data...')

if len(ext_feat) > 0:
    X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]]).values
    y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]]).values
    w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), EXT_WEIGHT)])
else:
    X_all = train_feat[feature_cols].values
    y_all = train_feat[TARGET].values
    w_all = np.ones(len(train_feat))

X_test = test_feat[feature_cols].values

print(f'  LGB  (n_est={lgb_n_est})...')
lgb_full = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=lgb_n_est,
                               random_state=SEED, n_jobs=-1, verbose=-1, **lgb_best)
lgb_full.fit(X_all, y_all, sample_weight=w_all)
lgb_test = lgb_full.predict_proba(X_test)[:, 1]

print(f'  XGB  (n_est={xgb_n_est})...')
xgb_full = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_estimators=xgb_n_est,
                              random_state=SEED, n_jobs=-1, tree_method='hist', verbosity=0, **xgb_best)
xgb_full.fit(X_all, y_all, sample_weight=w_all)
xgb_test = xgb_full.predict_proba(X_test)[:, 1]

print(f'  CAT  (n_est={cat_n_est})...')
cat_full = CatBoostClassifier(iterations=cat_n_est, eval_metric='AUC', random_seed=SEED,
                              verbose=0, task_type='CPU', **cat_best)
cat_full.fit(X_all, y_all, sample_weight=w_all)
cat_test = cat_full.predict_proba(X_test)[:, 1]

print(f'  DART (n_est={dart_n_est})...')
dart_full = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=dart_n_est,
                               random_state=SEED, n_jobs=-1, verbose=-1, **dart_best)
dart_full.fit(X_all, y_all, sample_weight=w_all)
dart_test = dart_full.predict_proba(X_test)[:, 1]

print('  MLP...')
mlp_scaler = StandardScaler()
Xs_all = mlp_scaler.fit_transform(X_all)
mlp_full = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
    learning_rate_init=0.001, max_iter=200, early_stopping=True,
    validation_fraction=0.05, n_iter_no_change=15, random_state=SEED, batch_size=2048,
)
mlp_full.fit(Xs_all, y_all)
mlp_test = mlp_full.predict_proba(mlp_scaler.transform(X_test))[:, 1]

print('Done.')"""
))

# ── Cell 12: Ensemble ──────────────────────────────────────────────────────────
cells.append(md("## 8 · Final Ensemble — LGB Stacker + Calibration"))
cells.append(code(
"""from scipy.optimize import minimize

def optimise_weights(oof_preds, y_true):
    n = len(oof_preds)
    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        blend = sum(wi * p for wi, p in zip(w, oof_preds))
        return -roc_auc_score(y_true, blend)
    res = minimize(neg_auc, x0=np.ones(n)/n, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-7})
    return np.abs(res.x) / np.abs(res.x).sum()

all_oofs  = [lgb_oof, xgb_oof, cat_oof, dart_oof, mlp_oof]
all_tests = [lgb_test, xgb_test, cat_test, dart_test, mlp_test]
names     = ['LGB', 'XGB', 'CAT', 'DART', 'MLP']

# Rank average (5 models)
rank5_oof  = rank_avg(all_oofs)
rank5_test = rank_avg(all_tests)
print(f'Rank 5-model OOF AUC: {roc_auc_score(y_true, rank5_oof):.5f}')

# Weighted blend
w_opt = optimise_weights(all_oofs, y_true)
print('Weights: ' + '  '.join(f'{n}={w:.3f}' for n, w in zip(names, w_opt)))
weighted_oof  = sum(wi * p for wi, p in zip(w_opt, all_oofs))
weighted_test = sum(wi * p for wi, p in zip(w_opt, all_tests))
print(f'Weighted blend OOF AUC: {roc_auc_score(y_true, weighted_oof):.5f}')

# ── LGB stacker (V2: replaces LR) ─────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_SPLITS)
meta_tr = np.column_stack(all_oofs)
meta_te = np.column_stack(all_tests)
stacked_oof  = np.zeros(len(y_true))
stacked_test = np.zeros(len(meta_te))

for tr_idx, val_idx in gkf.split(meta_tr, y_true, groups_train):
    stacker = lgb.LGBMClassifier(
        objective='binary', metric='auc', n_estimators=200,
        num_leaves=15, learning_rate=0.05, reg_alpha=1.0, reg_lambda=5.0,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    stacker.fit(meta_tr[tr_idx], y_true[tr_idx])
    stacked_oof[val_idx]  = stacker.predict_proba(meta_tr[val_idx])[:, 1]
    stacked_test         += stacker.predict_proba(meta_te)[:, 1] / N_SPLITS

print(f'LGB Stacked OOF AUC: {roc_auc_score(y_true, stacked_oof):.5f}')

# Choose best method
all_ensemble = {
    'LGB':              roc_auc_score(y_true, lgb_oof),
    'XGB':              roc_auc_score(y_true, xgb_oof),
    'CatBoost':         roc_auc_score(y_true, cat_oof),
    'DART (tuned)':     roc_auc_score(y_true, dart_oof),
    'MLP':              roc_auc_score(y_true, mlp_oof),
    'Rank 5-model':     roc_auc_score(y_true, rank5_oof),
    'Weighted 5-model': roc_auc_score(y_true, weighted_oof),
    'LGB Stacked':      roc_auc_score(y_true, stacked_oof),
}
print('\\n=== OOF AUC Comparison ===')
for k, v in sorted(all_ensemble.items(), key=lambda x: -x[1]):
    print(f'  {k:<22} {v:.5f}')

best_method = max(all_ensemble, key=all_ensemble.get)
print(f'\\nBest: {best_method}')

if 'Stacked' in best_method:
    final_preds = stacked_test
elif 'Weighted' in best_method:
    final_preds = weighted_test
else:
    final_preds = rank5_test"""
))

# ── Cell 13: Isotonic calibration ─────────────────────────────────────────────
cells.append(md("## 9 · Isotonic Calibration (V2)"))
cells.append(code(
"""# Fit isotonic regression on the best OOF predictions vs true labels,
# then apply to test. Only use if calibrated AUC >= uncalibrated AUC.
best_oof = {
    'LGB Stacked':      stacked_oof,
    'Weighted 5-model': weighted_oof,
    'Rank 5-model':     rank5_oof,
}[best_method] if best_method in ('LGB Stacked', 'Weighted 5-model', 'Rank 5-model') else rank5_oof

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(best_oof, y_true)
calibrated_oof  = iso.predict(best_oof)
calibrated_test = iso.predict(final_preds)

cal_auc  = roc_auc_score(y_true, calibrated_oof)
uncal_auc= roc_auc_score(y_true, best_oof)
print(f'Pre-calibration  OOF AUC: {uncal_auc:.5f}')
print(f'Post-calibration OOF AUC: {cal_auc:.5f}')

if cal_auc >= uncal_auc:
    print('Using calibrated predictions.')
    final_preds = calibrated_test
else:
    print('Calibration did not help — keeping original ensemble.')"""
))

# ── Cell 14: Iterative pseudo-labeling ────────────────────────────────────────
cells.append(md("## 10 · Iterative Pseudo-Labeling (V2: 2 rounds)"))
cells.append(code(
"""HIGH_THRESH_1, LOW_THRESH_1 = 0.90, 0.05
HIGH_THRESH_2, LOW_THRESH_2 = 0.85, 0.08
PL_WEIGHT = 0.5

current_preds = final_preds.copy()
best_auc      = all_ensemble[best_method]

for rnd, (hi, lo) in enumerate(
        [(HIGH_THRESH_1, LOW_THRESH_1), (HIGH_THRESH_2, LOW_THRESH_2)], 1):
    mask   = (current_preds > hi) | (current_preds < lo)
    pseudo = test_feat[mask].copy()
    pseudo[TARGET] = (current_preds[mask] > 0.5).astype(int)
    n_pos  = (pseudo[TARGET] == 1).sum()
    print(f'\\nRound {rnd}: {len(pseudo):,} pseudo-labels ({n_pos:,} pos, {len(pseudo)-n_pos:,} neg)')

    if len(ext_feat) > 0:
        ext_pl = pd.concat([ext_feat, pseudo], ignore_index=True)
    else:
        ext_pl = pseudo.copy()

    pl_X = ext_pl[feature_cols].values
    pl_y = ext_pl[TARGET].values
    pl_w = np.full(len(ext_pl), PL_WEIGHT)

    _ext_X, _ext_y, _ext_w = ext_X, ext_y, ext_w
    ext_X, ext_y, ext_w    = pl_X, pl_y, pl_w

    def lgb_pl_fn(X_tr, y_tr, w_tr, X_val, y_val):
        m = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=3000,
                               random_state=SEED, n_jobs=-1, verbose=-1, **lgb_best)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
        return m

    lgb_oof_pl, _, pl_iters = run_cv(lgb_pl_fn, track_iters=True)
    pl_auc = roc_auc_score(y_true, lgb_oof_pl)
    ext_X, ext_y, ext_w = _ext_X, _ext_y, _ext_w  # restore

    print(f'LGB OOF AUC with PL round {rnd}: {pl_auc:.5f}  (baseline: {best_auc:.5f})')

    if pl_auc > best_auc:
        best_auc = pl_auc
        print(f'Round {rnd} improved — training full model...')
        pl_n_est  = int(np.mean(pl_iters)) if pl_iters else lgb_n_est
        X_pl = np.concatenate([X_all, pl_X])
        y_pl = np.concatenate([y_all, pl_y])
        w_pl = np.concatenate([w_all, pl_w])
        lgb_pl_full = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=pl_n_est,
                                         random_state=SEED, n_jobs=-1, verbose=-1, **lgb_best)
        lgb_pl_full.fit(X_pl, y_pl, sample_weight=w_pl)
        lgb_test_pl  = lgb_pl_full.predict_proba(X_test)[:, 1]
        current_preds = rank_avg([lgb_test_pl, xgb_test, cat_test, dart_test, mlp_test])
        final_preds   = current_preds
        print(f'Updated final_preds with PL round {rnd} ensemble.')
    else:
        print(f'Round {rnd} did not improve — keeping current ensemble.')
        break"""
))

# ── Cell 15: Feature importance ────────────────────────────────────────────────
cells.append(md("## 11 · Feature Importance"))
cells.append(code(
"""fi = pd.Series(lgb_full.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 9))
fi.head(25).sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('LightGBM Feature Importance — Top 25 (Gain)', fontsize=13)
ax.set_xlabel('Gain')
plt.tight_layout()
plt.show()

print('\\nTop 15 features:')
print(fi.head(15).to_string())
print('\\nV2-only features in top 25:')
v2_feats = {'PitStops_so_far','position_trend5','pos_vs_roll3',
            'tyre_life_vs_field_med','LT_vs_own_pace','deg_jerk','StintFraction'}
for feat in fi.head(25).index:
    if feat in v2_feats:
        print(f'  {feat}: rank {fi.index.get_loc(feat)+1}  gain={fi[feat]:.0f}')"""
))

# ── Cell 16: Submission ────────────────────────────────────────────────────────
cells.append(md("## 12 · Submission"))
cells.append(code(
"""submission = pd.DataFrame({
    'id':    test_feat[ID_COL].values,
    TARGET:  final_preds,
})

assert len(submission) == len(sub)
assert submission[TARGET].between(0, 1).all()
assert submission['id'].nunique() == len(submission)

submission.to_csv('submission.csv', index=False)
print(f'submission.csv saved: {len(submission):,} rows')
print(f'Prediction mean: {submission[TARGET].mean():.4f}  std: {submission[TARGET].std():.4f}')

# OOF summary
print('\\n=== Final OOF AUC Summary ===')
for k, v in sorted(all_ensemble.items(), key=lambda x: -x[1]):
    print(f'  {k:<22} {v:.5f}')
submission.head()"""
))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent.parent / 'notebooks' / 'predicting-f1-pit-stops-v2.ipynb'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Written: {out}  ({out.stat().st_size // 1024} KB,  {len(cells)} cells)')
