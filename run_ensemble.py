"""
Final ensemble: loads tuned OOF + test predictions, blends via rank avg,
tries stacking, runs pseudo-labeling, generates final submissions.
Run AFTER run_tuning.py has completed.
"""
import sys, warnings, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.features import get_feature_cols, TARGET
from src.cv import run_cv, add_target_encoding
from src.models import make_lgb_fn, make_lgb_dart_fn, train_full
from src.ensemble import pseudo_label, stack_predict
from src.utils import rank_avg, optimise_weights, save_submission

cache = Path('cache')
train_feat = pd.read_pickle(cache / 'train_feat.pkl')
ext_feat   = pd.read_pickle(cache / 'ext_feat.pkl')
test_feat  = pd.read_pickle(cache / 'test_feat.pkl')

train_feat, ext_feat, test_feat = add_target_encoding(train_feat, ext_feat, test_feat)
feature_cols = get_feature_cols(train_feat) + [c for c in ['Driver_TE', 'Race_TE', 'Race_Year_TE'] if c in train_feat.columns]
y_true = train_feat[TARGET].values

with open(cache / 'oof_preds_tuned.pkl', 'rb') as f:
    oof_tuned = pickle.load(f)
with open(cache / 'test_preds_tuned.pkl', 'rb') as f:
    test_tuned = pickle.load(f)
with open(cache / 'best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

lgb_oof = oof_tuned['lgb']
xgb_oof = oof_tuned['xgb']
cat_oof = oof_tuned['cat']
lgb_test = test_tuned['lgb']
xgb_test = test_tuned['xgb']
cat_test = test_tuned['cat']

print('=== TUNED MODEL RESULTS ===')
print(f'LGB  : {roc_auc_score(y_true, lgb_oof):.5f}')
print(f'XGB  : {roc_auc_score(y_true, xgb_oof):.5f}')
print(f'CAT  : {roc_auc_score(y_true, cat_oof):.5f}')


# ── DART mode LGB ────────────────────────────────────────────────────────────
print('\n=== LGB DART mode (n_estimators=600) ===')
dart_fn = make_lgb_dart_fn(n_estimators=600, params=best_params.get('lgb', {}))
dart_oof, dart_aucs = run_cv(train_feat, ext_feat, feature_cols, dart_fn)
print(f'LGB DART OOF AUC: {roc_auc_score(y_true, dart_oof):.5f}')


# ── Rank blend ───────────────────────────────────────────────────────────────
all_oofs = [lgb_oof, xgb_oof, cat_oof, dart_oof]
all_tests = [lgb_test, xgb_test, cat_test]

rank_oof = rank_avg([lgb_oof, xgb_oof, cat_oof])
rank_test = rank_avg([lgb_test, xgb_test, cat_test])
print(f'\nRank ensemble (LGB+XGB+CAT): {roc_auc_score(y_true, rank_oof):.5f}')

rank_oof_dart = rank_avg([lgb_oof, xgb_oof, cat_oof, dart_oof])
# DART test preds need separate full-train DART model
X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]]).values
y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]]).values
w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), 0.8)])
X_test = test_feat[feature_cols].values

dart_params = {**best_params.get('lgb', {}), 'boosting_type': 'dart', 'drop_rate': 0.1, 'skip_drop': 0.5}
dart_model = train_full(X_all, y_all, w_all, 'lgb', dart_params, n_estimators=600)
dart_test_preds = dart_model.predict_proba(X_test)[:, 1]

rank_test_dart = rank_avg([lgb_test, xgb_test, cat_test, dart_test_preds])
print(f'Rank ensemble (LGB+XGB+CAT+DART): {roc_auc_score(y_true, rank_oof_dart):.5f}')


# ── Optimised weighted blend ─────────────────────────────────────────────────
w_opt = optimise_weights([lgb_oof, xgb_oof, cat_oof, dart_oof], y_true)
print(f'\nOptimised weights: LGB={w_opt[0]:.3f} XGB={w_opt[1]:.3f} CAT={w_opt[2]:.3f} DART={w_opt[3]:.3f}')
weighted_oof = sum(wi * p for wi, p in zip(w_opt, [lgb_oof, xgb_oof, cat_oof, dart_oof]))
weighted_test = sum(wi * p for wi, p in zip(w_opt, [lgb_test, xgb_test, cat_test, dart_test_preds]))
print(f'Weighted ensemble OOF AUC: {roc_auc_score(y_true, weighted_oof):.5f}')


# ── Stacking ─────────────────────────────────────────────────────────────────
print('\n=== Stacking (LR meta-learner) ===')
stacked_oof, stacked_test = stack_predict(
    [lgb_oof, xgb_oof, cat_oof, dart_oof],
    [lgb_test, xgb_test, cat_test, dart_test_preds],
    y_true,
    train_feat,
)


# ── Pseudo-labeling ──────────────────────────────────────────────────────────
print('\n=== Pseudo-labeling ===')
train_pl, ext_pl = pseudo_label(
    train_feat, ext_feat, test_feat,
    rank_test, feature_cols,
    high_thresh=0.90, low_thresh=0.05, pseudo_weight=0.5
)
best_lgb_fn = make_lgb_fn(best_params.get('lgb', {}))
lgb_oof_pl, _ = run_cv(train_pl, ext_pl, feature_cols, best_lgb_fn)
print(f'LGB OOF AUC (with pseudo-labels): {roc_auc_score(y_true, lgb_oof_pl):.5f}')


# ── Final comparison ─────────────────────────────────────────────────────────
print('\n=== FINAL OOF AUC COMPARISON ===')
all_results = {
    'LGB tuned':            roc_auc_score(y_true, lgb_oof),
    'XGB tuned':            roc_auc_score(y_true, xgb_oof),
    'CAT tuned':            roc_auc_score(y_true, cat_oof),
    'LGB DART':             roc_auc_score(y_true, dart_oof),
    'Rank 3-model':         roc_auc_score(y_true, rank_oof),
    'Rank 4-model (+DART)': roc_auc_score(y_true, rank_oof_dart),
    'Weighted 4-model':     roc_auc_score(y_true, weighted_oof),
    'Stacked (LR meta)':    roc_auc_score(y_true, stacked_oof),
    'LGB + pseudo':         roc_auc_score(y_true, lgb_oof_pl),
}
for k, v in sorted(all_results.items(), key=lambda x: -x[1]):
    print(f'  {k:<28} {v:.5f}')

# Best OOF method determines submission
best_method = max(all_results, key=all_results.get)
print(f'\nBest OOF method: {best_method}')

if 'Stacked' in best_method:
    final_test = stacked_test
elif 'DART' in best_method and '4' in best_method:
    final_test = rank_test_dart
elif 'Weighted' in best_method:
    final_test = weighted_test
elif 'pseudo' in best_method:
    # Re-train full model with pseudo labels for test predictions
    X_pl = pd.concat([train_pl[feature_cols], ext_pl[feature_cols]]).values
    y_pl = pd.concat([train_pl[TARGET], ext_pl[TARGET]]).values
    w_pl = np.concatenate([np.ones(len(train_pl)),
                            ext_pl.get('_pseudo_weight', pd.Series(0.8, index=ext_pl.index)).values])
    final_test = train_full(X_pl, y_pl, w_pl, 'lgb', best_params.get('lgb', {}), 1500).predict_proba(X_test)[:, 1]
else:
    final_test = rank_test

sub = save_submission(test_feat['id'], final_test, tag='final_ensemble')
print(f'\nFinal submission: {sub}')

# Also save rank 4-model and stacked as alternatives
sub2 = save_submission(test_feat['id'], rank_test_dart, tag='rank4_lgb_xgb_cat_dart')
print(f'Alternative (rank 4-model): {sub2}')
sub3 = save_submission(test_feat['id'], stacked_test, tag='stacked_lr_meta')
print(f'Alternative (stacked LR):   {sub3}')
