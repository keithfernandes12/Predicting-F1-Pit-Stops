import sys, pickle
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.utils import rank_avg

cache = Path('cache')
train_feat = pd.read_pickle(cache / 'train_feat.pkl')
y_true = train_feat['PitNextLap'].values

results = {}
for name in ['lgb', 'xgb', 'cat']:
    p = cache / f'{name}_oof.pkl'
    if p.exists():
        data = pickle.load(open(p, 'rb'))
        auc = roc_auc_score(y_true, data['oof'])
        results[name] = data['oof']
        print(f'{name.upper()} OOF AUC: {auc:.5f}  folds: {[round(a,5) for a in data["aucs"]]}')
    else:
        print(f'{name.upper()}: not found')

if len(results) > 1:
    ens = rank_avg(list(results.values()))
    print(f'Rank ensemble ({"+".join(results.keys())}): {roc_auc_score(y_true, ens):.5f}')
