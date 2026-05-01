"""Run full 5-fold CV for LGB, XGB, CatBoost and save results."""
import sys, warnings, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.features import get_feature_cols, TARGET
from src.cv import run_cv, add_target_encoding
from src.models import make_lgb_fn, make_xgb_fn, make_cat_fn, train_full
from src.utils import rank_avg, save_submission

cache = Path('cache')
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
oof_preds = {}
fold_aucs_all = {}

# ── LightGBM ────────────────────────────────────────────────────────────────
print('\n=== LightGBM 5-fold CV ===')
lgb_fn = make_lgb_fn()
lgb_oof, lgb_aucs = run_cv(train_feat, ext_feat, feature_cols, lgb_fn)
oof_preds['lgb'] = lgb_oof
fold_aucs_all['lgb'] = lgb_aucs
print(f'LGB OOF AUC: {roc_auc_score(y_true, lgb_oof):.5f}')

# ── XGBoost ─────────────────────────────────────────────────────────────────
print('\n=== XGBoost 5-fold CV ===')
xgb_fn = make_xgb_fn()
xgb_oof, xgb_aucs = run_cv(train_feat, ext_feat, feature_cols, xgb_fn)
oof_preds['xgb'] = xgb_oof
fold_aucs_all['xgb'] = xgb_aucs
print(f'XGB OOF AUC: {roc_auc_score(y_true, xgb_oof):.5f}')

# ── CatBoost ─────────────────────────────────────────────────────────────────
print('\n=== CatBoost 5-fold CV ===')
cat_fn = make_cat_fn()
cat_oof, cat_aucs = run_cv(train_feat, ext_feat, feature_cols, cat_fn)
oof_preds['cat'] = cat_oof
fold_aucs_all['cat'] = cat_aucs
print(f'CAT OOF AUC: {roc_auc_score(y_true, cat_oof):.5f}')

# ── Ensemble ─────────────────────────────────────────────────────────────────
ensemble_oof = rank_avg([lgb_oof, xgb_oof, cat_oof])
print(f'\nRank ensemble OOF AUC: {roc_auc_score(y_true, ensemble_oof):.5f}')

# ── Test predictions ────────────────────────────────────────────────────────
X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]]).values
y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]]).values
w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), 0.8)])
X_test = test_feat[feature_cols].values

print('\nTraining full models for test predictions...')
lgb_full = train_full(X_all, y_all, w_all, 'lgb', n_estimators=1000)
lgb_test = lgb_full.predict_proba(X_test)[:, 1]

xgb_full = train_full(X_all, y_all, w_all, 'xgb', n_estimators=1000)
xgb_test = xgb_full.predict_proba(X_test)[:, 1]

cat_full = train_full(X_all, y_all, w_all, 'cat', n_estimators=1000)
cat_test = cat_full.predict_proba(X_test)[:, 1]

test_preds = {'lgb': lgb_test, 'xgb': xgb_test, 'cat': cat_test}
ensemble_test = rank_avg([lgb_test, xgb_test, cat_test])

# ── Save ─────────────────────────────────────────────────────────────────────
with open(cache / 'oof_preds_baseline.pkl', 'wb') as f:
    pickle.dump(oof_preds, f)
with open(cache / 'test_preds_baseline.pkl', 'wb') as f:
    pickle.dump(test_preds, f)
with open(cache / 'fold_aucs_baseline.pkl', 'wb') as f:
    pickle.dump(fold_aucs_all, f)

sub = save_submission(test_feat['id'], ensemble_test, tag='baseline_rank_ensemble')

print('\n=== RESULTS SUMMARY ===')
for name, oof in oof_preds.items():
    print(f'  {name.upper():<10} OOF AUC: {roc_auc_score(y_true, oof):.5f}  |  folds: {[round(a,5) for a in fold_aucs_all[name]]}')
print(f'  ENSEMBLE   OOF AUC: {roc_auc_score(y_true, ensemble_oof):.5f}')
print(f'\nSubmission: {sub}')
