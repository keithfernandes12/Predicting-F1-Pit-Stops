"""
Regenerate feature cache with v2 features (6 new features added to FEATURE_COLS):
  CD_lag1, LT_roll5_std, CD_roll3_mean, tyre_life_vs_field_max,
  deg_acceleration, position_trend

Then run a quick LGB baseline (default params) to measure the AUC delta.
Run AFTER run_tuning.py has written best_params.pkl — uses tuned params if available.
"""
import sys, warnings, pickle
warnings.filterwarnings('ignore')
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.data import load_train, load_test, load_external, dedup_external
from src.features import prepare_all, get_feature_cols, TARGET
from src.cv import run_cv, add_target_encoding
from src.models import make_lgb_fn, DEFAULT_LGB_PARAMS
from src.utils import rank_avg

ROOT = pathlib.Path(__file__).resolve().parent.parent
cache = ROOT / 'cache'

print('Loading raw data...')
train = load_train()
test  = load_test()
ext   = load_external()
ext   = dedup_external(train, ext)
print(f'  train={len(train):,}  test={len(test):,}  ext_unique={len(ext):,}')

print('Building v2 features...')
train_feat, test_feat, ext_feat = prepare_all(train, test, ext)

# Verify new features exist
new_feats = ['CD_lag1', 'LT_roll5_std', 'CD_roll3_mean',
             'tyre_life_vs_field_max', 'deg_acceleration', 'position_trend']
present = [f for f in new_feats if f in train_feat.columns]
missing = [f for f in new_feats if f not in train_feat.columns]
print(f'New features present: {present}')
if missing:
    print(f'WARNING — missing: {missing}')

print('Saving v2 cache...')
train_feat.to_pickle(cache / 'train_feat.pkl')
ext_feat.to_pickle(cache / 'ext_feat.pkl')
test_feat.to_pickle(cache / 'test_feat.pkl')

print('Applying target encoding...')
train_feat, ext_feat, test_feat = add_target_encoding(train_feat, ext_feat, test_feat)
feature_cols = get_feature_cols(train_feat) + [c for c in ['Driver_TE', 'Race_TE', 'Race_Year_TE'] if c in train_feat.columns]
print(f'Total features: {len(feature_cols)}')
y_true = train_feat[TARGET].values

# Use tuned params if available, else defaults
best_params_path = cache / 'best_params.pkl'
if best_params_path.exists():
    best_params = pickle.load(open(best_params_path, 'rb'))
    lgb_params = best_params.get('lgb', {})
    print('Using tuned LGB params.')
else:
    lgb_params = {}
    print('Using default LGB params (run run_tuning.py first for best results).')

print('\n=== LGB 5-fold CV with v2 features ===')
lgb_oof, lgb_aucs = run_cv(train_feat, ext_feat, feature_cols, make_lgb_fn(lgb_params))
v2_auc = roc_auc_score(y_true, lgb_oof)
print(f'\nLGB v2 OOF AUC: {v2_auc:.5f}')

# Quick feature importance to confirm new features are useful
import lightgbm as lgb_lib
X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]]).values
y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]]).values
w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), 0.8)])
params_full = {**DEFAULT_LGB_PARAMS, **lgb_params, 'n_estimators': 500}
model = lgb_lib.LGBMClassifier(**params_full)
model.fit(X_all, y_all, sample_weight=w_all)
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print('\nTop 15 features by gain:')
print(fi.head(15).to_string())
print('\nNew feature ranks:')
for f in new_feats:
    if f in fi.index:
        rank = list(fi.index).index(f) + 1
        print(f'  {f:<30} rank={rank:>3}  gain={fi[f]:.0f}')
