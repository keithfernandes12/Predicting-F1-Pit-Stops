import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import rank_avg, optimise_weights
from src.features import TARGET


def blend_oof(oof_list: list, y_true: np.ndarray, method: str = "rank") -> np.ndarray:
    """
    Blend OOF predictions.
    method: 'rank' (rank-based avg), 'weighted' (AUC-optimised weights), 'mean'
    """
    if method == "rank":
        return rank_avg(oof_list)
    if method == "weighted":
        w = optimise_weights(oof_list, y_true)
        print(f"Optimised weights: {w}")
        return sum(wi * p for wi, p in zip(w, oof_list))
    return np.mean(oof_list, axis=0)


def blend_test(test_pred_list: list, oof_list: list, y_true: np.ndarray, method: str = "rank") -> np.ndarray:
    if method == "rank":
        return rank_avg(test_pred_list)
    if method == "weighted":
        w = optimise_weights(oof_list, y_true)
        return sum(wi * p for wi, p in zip(w, test_pred_list))
    return np.mean(test_pred_list, axis=0)


def pseudo_label(
    train_feat: pd.DataFrame,
    ext_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    test_preds: np.ndarray,
    feature_cols: list,
    high_thresh: float = 0.90,
    low_thresh: float = 0.05,
    pseudo_weight: float = 0.5,
):
    """
    Add high-confidence test rows as pseudo-labeled training data.
    Returns (new_train_feat, new_ext_feat) ready for another CV run.
    """
    high_conf = (test_preds > high_thresh) | (test_preds < low_thresh)
    pseudo = test_feat[high_conf].copy()
    pseudo[TARGET] = (test_preds[high_conf] > 0.5).astype(int)
    pseudo["_pseudo_weight"] = pseudo_weight

    n = high_conf.sum()
    pos = (test_preds[high_conf] > 0.5).sum()
    print(f"Pseudo-labeling: {n:,} rows ({pos:,} positive, {n - pos:,} negative)")

    new_ext = pd.concat([ext_feat, pseudo], ignore_index=True)
    return train_feat, new_ext


def stack_predict(
    oof_list: list,
    test_pred_list: list,
    y_true: np.ndarray,
    train_feat: pd.DataFrame,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Simple logistic regression meta-learner on OOF predictions.
    Returns test-set stacked predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    meta_train = np.column_stack(oof_list)
    meta_test = np.column_stack(test_pred_list)

    groups = train_feat["Race"].astype(str) + "_" + train_feat["Year"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)

    stacked_oof = np.zeros(len(y_true))
    stacked_test = np.zeros(len(meta_test))

    for tr_idx, val_idx in gkf.split(meta_train, y_true, groups):
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(meta_train[tr_idx], y_true[tr_idx])
        stacked_oof[val_idx] = lr.predict_proba(meta_train[val_idx])[:, 1]
        stacked_test += lr.predict_proba(meta_test)[:, 1] / n_splits

    print(f"Stacked OOF AUC: {roc_auc_score(y_true, stacked_oof):.5f}")
    return stacked_oof, stacked_test
