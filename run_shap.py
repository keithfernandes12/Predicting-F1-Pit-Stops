"""
SHAP analysis: train full LGB on all data, compute SHAP values on validation sample,
save summary plot and feature importance CSV.
Run after run_tuning.py (uses best params if available, else defaults).
"""
import sys, warnings, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb_lib
from pathlib import Path

from src.features import get_feature_cols, TARGET
from src.cv import add_target_encoding
from src.models import DEFAULT_LGB_PARAMS

cache = Path('cache')
train_feat = pd.read_pickle(cache / 'train_feat.pkl')
ext_feat   = pd.read_pickle(cache / 'ext_feat.pkl')
test_feat  = pd.read_pickle(cache / 'test_feat.pkl')
train_feat, ext_feat, test_feat = add_target_encoding(train_feat, ext_feat, test_feat)

feature_cols = get_feature_cols(train_feat) + [c for c in ['Driver_TE','Race_TE','Race_Year_TE'] if c in train_feat.columns]

# Use best params if available
best_params_path = cache / 'best_params.pkl'
if best_params_path.exists():
    best_params = pickle.load(open(best_params_path, 'rb'))
    params = {**DEFAULT_LGB_PARAMS, **best_params.get('lgb', {}), 'n_estimators': 500}
    print('Using tuned LGB params.')
else:
    params = {**DEFAULT_LGB_PARAMS, 'n_estimators': 500}
    print('Using default LGB params (run run_tuning.py for better results).')

X_all = pd.concat([train_feat[feature_cols], ext_feat[feature_cols]])
y_all = pd.concat([train_feat[TARGET], ext_feat[TARGET]])
w_all = np.concatenate([np.ones(len(train_feat)), np.full(len(ext_feat), 0.8)])

print('Training LGB for SHAP analysis...')
model = lgb_lib.LGBMClassifier(**params)
model.fit(X_all, y_all, sample_weight=w_all)

try:
    import shap
    print('Computing SHAP values on 5k sample...')
    sample = train_feat.sample(5000, random_state=42)
    X_sample = sample[feature_cols].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols,
                      show=False, max_display=30)
    plt.title('SHAP Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved shap_summary.png')

    # Save mean |SHAP| per feature
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    mean_abs_shap.to_csv('shap_importance.csv', header=['mean_abs_shap'])
    print('Saved shap_importance.csv')
    print('\nTop 20 features by mean |SHAP|:')
    print(mean_abs_shap.head(20).to_string())
except ImportError:
    print('shap not installed — pip install shap')

# LGB gain importance as fallback
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
fi.to_csv('lgb_importance.csv', header=['gain'])
print('\nTop 20 by LGB gain importance:')
print(fi.head(20).to_string())
