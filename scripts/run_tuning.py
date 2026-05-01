"""
Optuna hyperparameter search: LGB (30 trials), XGB (20 trials), CatBoost (10 trials).
Uses 3-fold CV during search (faster), then full 5-fold with best params.
Saves best params + tuned OOF predictions + a tuned submission.
"""
import sys, warnings, pickle, time
warnings.filterwarnings('ignore')
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.features import get_feature_cols, TARGET
from src.cv import run_cv, add_target_encoding
from src.models import make_lgb_fn, make_xgb_fn, make_cat_fn, train_full
from src.utils import rank_avg, save_submission

ROOT = pathlib.Path(__file__).resolve().parent.parent
cache = ROOT / 'cache'
train_feat = pd.read_pickle(cache / 'train_feat.pkl')
ext_feat   = pd.read_pickle(cache / 'ext_feat.pkl')
test_feat  = pd.read_pickle(cache / 'test_feat.pkl')

print('Applying target encoding...')
train_feat, ext_feat, test_feat = add_target_encoding(train_feat, ext_feat, test_feat)
feature_cols = get_feature_cols(train_feat)
te_cols = [c for c in ['Driver_TE', 'Race_TE', 'Race_Year_TE'] if c in train_feat.columns]
feature_cols = feature_cols + te_cols
print(f'Feature count: {len(feature_cols)}')
y_true = train_feat[TARGET].values


def _run3(fn):
    _, aucs = run_cv(train_feat, ext_feat, feature_cols, fn, n_splits=3, verbose=False)
    return float(np.mean(aucs))


# ── LightGBM ────────────────────────────────────────────────────────────────
print('\n=== LightGBM Optuna (30 trials, 3-fold) ===')
t0 = time.time()

def lgb_obj(trial):
    p = {
        'num_leaves': trial.suggest_int('num_leaves', 63, 511),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'subsample_freq': 1,
    }
    return _run3(make_lgb_fn(p, early_stopping_rounds=50))

lgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_obj, n_trials=30, show_progress_bar=True)
print(f'Best LGB (3-fold): {lgb_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
print(f'Best params: {lgb_study.best_params}')

print('\n=== LGB best params → 5-fold CV ===')
lgb_oof, lgb_aucs = run_cv(train_feat, ext_feat, feature_cols, make_lgb_fn(lgb_study.best_params))
print(f'LGB tuned OOF AUC: {roc_auc_score(y_true, lgb_oof):.5f}')


# ── XGBoost ─────────────────────────────────────────────────────────────────
print('\n=== XGBoost Optuna (20 trials, 3-fold) ===')
t0 = time.time()

def xgb_obj(trial):
    p = {
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    return _run3(make_xgb_fn(p, early_stopping_rounds=50))

xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_obj, n_trials=20, show_progress_bar=True)
print(f'Best XGB (3-fold): {xgb_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
print(f'Best params: {xgb_study.best_params}')

print('\n=== XGB best params → 5-fold CV ===')
xgb_oof, xgb_aucs = run_cv(train_feat, ext_feat, feature_cols, make_xgb_fn(xgb_study.best_params))
print(f'XGB tuned OOF AUC: {roc_auc_score(y_true, xgb_oof):.5f}')


# ── CatBoost ─────────────────────────────────────────────────────────────────
print('\n=== CatBoost Optuna (10 trials, 3-fold) ===')
t0 = time.time()

def cat_obj(trial):
    p = {
        'depth': trial.suggest_int('depth', 5, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'early_stopping_rounds': 50,
    }
    return _run3(make_cat_fn(p))

cat_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
cat_study.optimize(cat_obj, n_trials=10, show_progress_bar=True)
print(f'Best CAT (3-fold): {cat_study.best_value:.5f}  ({(time.time()-t0)/60:.1f}m)')
print(f'Best params: {cat_study.best_params}')

print('\n=== CAT best params → 5-fold CV ===')
cat_oof, cat_aucs = run_cv(train_feat, ext_feat, feature_cols,
                            make_cat_fn({**cat_study.best_params, 'early_stopping_rounds': 100}))
print(f'CAT tuned OOF AUC: {roc_auc_score(y_true, cat_oof):.5f}')


# ── Ensemble ─────────────────────────────────────────────────────────────────
rank_oof = rank_avg([lgb_oof, xgb_oof, cat_oof])
print(f'\nRank ensemble OOF AUC: {roc_auc_score(y_true, rank_oof):.5f}')


# ── Generate tuned test predictions ─────────────────────────────────────────
X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]]).values
y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]]).values
w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), 0.8)])
X_test = test_feat[feature_cols].values
n_est = 1500

print('\nTraining full tuned models...')
lgb_test = train_full(X_all, y_all, w_all, 'lgb', lgb_study.best_params, n_est).predict_proba(X_test)[:, 1]
xgb_test = train_full(X_all, y_all, w_all, 'xgb', xgb_study.best_params, n_est).predict_proba(X_test)[:, 1]
cat_test = train_full(X_all, y_all, w_all, 'cat', cat_study.best_params, n_est).predict_proba(X_test)[:, 1]
rank_test = rank_avg([lgb_test, xgb_test, cat_test])

# ── Save ─────────────────────────────────────────────────────────────────────
best_params = {'lgb': lgb_study.best_params, 'xgb': xgb_study.best_params, 'cat': cat_study.best_params}
oof_tuned  = {'lgb': lgb_oof, 'xgb': xgb_oof, 'cat': cat_oof}
test_tuned = {'lgb': lgb_test, 'xgb': xgb_test, 'cat': cat_test}

with open(cache / 'best_params.pkl', 'wb') as f: pickle.dump(best_params, f)
with open(cache / 'oof_preds_tuned.pkl', 'wb') as f: pickle.dump(oof_tuned, f)
with open(cache / 'test_preds_tuned.pkl', 'wb') as f: pickle.dump(test_tuned, f)

sub = save_submission(test_feat['id'], rank_test, tag='tuned_rank_ensemble')

print('\n=== TUNING RESULTS SUMMARY ===')
print(f'  LGB tuned  : {roc_auc_score(y_true, lgb_oof):.5f}')
print(f'  XGB tuned  : {roc_auc_score(y_true, xgb_oof):.5f}')
print(f'  CAT tuned  : {roc_auc_score(y_true, cat_oof):.5f}')
print(f'  Rank ens   : {roc_auc_score(y_true, rank_oof):.5f}')
print(f'\nSubmission: {sub}')
