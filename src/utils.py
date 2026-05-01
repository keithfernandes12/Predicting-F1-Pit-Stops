import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from datetime import datetime

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)


def auc(y_true, y_pred) -> float:
    return roc_auc_score(y_true, y_pred)


def save_submission(test_ids: pd.Series, preds: np.ndarray, tag: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"sub_{ts}_{tag}.csv" if tag else f"sub_{ts}.csv"
    path = SUBMISSIONS_DIR / name
    pd.DataFrame({"id": test_ids, "PitNextLap": preds}).to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def rank_avg(arrays: list) -> np.ndarray:
    """Rank-based average blending — normalises calibration across models."""
    from scipy.stats import rankdata
    n = len(arrays[0])
    return np.mean([rankdata(a) / n for a in arrays], axis=0)


def optimise_weights(oof_preds: list, y_true: np.ndarray) -> np.ndarray:
    """Find weights that maximise OOF AUC via Nelder-Mead."""
    from scipy.optimize import minimize

    n = len(oof_preds)

    def neg_auc(w):
        w = np.abs(w)
        w /= w.sum()
        blend = sum(wi * p for wi, p in zip(w, oof_preds))
        return -auc(y_true, blend)

    res = minimize(neg_auc, x0=np.ones(n) / n, method="Nelder-Mead")
    best_w = np.abs(res.x)
    best_w /= best_w.sum()
    return best_w
