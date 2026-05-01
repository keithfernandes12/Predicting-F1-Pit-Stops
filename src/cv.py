import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from src.features import TARGET, ID_COL

N_SPLITS = 5


def get_groups(df: pd.DataFrame) -> pd.Series:
    return df["Race"].astype(str) + "_" + df["Year"].astype(str)


def run_cv(
    train_feat: pd.DataFrame,
    ext_feat: pd.DataFrame,
    feature_cols: list,
    model_fn,
    n_splits: int = N_SPLITS,
    ext_sample_weight: float = 0.8,
    verbose: bool = True,
):
    """
    GroupKFold CV grouped by (Race, Year).

    For each fold:
      - train  = train_fold rows + ALL external rows
      - val    = holdout train rows only

    model_fn: callable(X_tr, y_tr, w_tr, X_val, y_val) -> fitted model with predict_proba

    Returns: (oof_preds np.ndarray aligned to train_feat index, list of per-fold AUCs)
    """
    groups = get_groups(train_feat)
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(train_feat))
    fold_aucs = []

    ext_X = ext_feat[feature_cols].values
    ext_y = ext_feat[TARGET].values
    ext_w = np.full(len(ext_feat), ext_sample_weight)

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(train_feat, train_feat[TARGET], groups)
    ):
        tr_rows = train_feat.iloc[tr_idx]
        val_rows = train_feat.iloc[val_idx]

        X_tr = np.concatenate([tr_rows[feature_cols].values, ext_X], axis=0)
        y_tr = np.concatenate([tr_rows[TARGET].values, ext_y], axis=0)
        w_tr = np.concatenate([np.ones(len(tr_rows)), ext_w], axis=0)

        X_val = val_rows[feature_cols].values
        y_val = val_rows[TARGET].values

        model = model_fn(X_tr, y_tr, w_tr, X_val, y_val)
        preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = preds

        fold_auc = roc_auc_score(y_val, preds)
        fold_aucs.append(fold_auc)
        if verbose:
            print(f"  Fold {fold + 1}/{n_splits} | AUC = {fold_auc:.5f} | val_rows = {len(val_idx):,}")

    mean_auc = np.mean(fold_aucs)
    if verbose:
        print(f"  Mean CV AUC: {mean_auc:.5f}  (std={np.std(fold_aucs):.5f})")

    return oof_preds, fold_aucs


def add_target_encoding(
    train_feat: pd.DataFrame,
    ext_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    n_splits: int = N_SPLITS,
) -> tuple:
    """
    Out-of-fold target encoding for Driver, Race, and (Race, Year).
    Computes encodings inside each CV fold to prevent leakage.
    Returns updated (train_feat, ext_feat, test_feat).
    """
    groups = get_groups(train_feat)
    gkf = GroupKFold(n_splits=n_splits)
    global_mean = train_feat[TARGET].mean()

    te_cols = ["Driver_TE", "Race_TE", "Race_Year_TE"]
    for col in te_cols:
        train_feat[col] = global_mean
        ext_feat[col] = global_mean
        test_feat[col] = global_mean

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(train_feat, train_feat[TARGET], groups)
    ):
        tr = train_feat.iloc[tr_idx]

        driver_map = tr.groupby("Driver")[TARGET].mean()
        race_map = tr.groupby("Race")[TARGET].mean()
        ry_map = tr.groupby(["Race", "Year"])[TARGET].mean()

        def _apply(df, map_, col, keys):
            if isinstance(keys, str):
                df.loc[df.index, col] = df[keys].map(map_).fillna(global_mean).values
            else:
                merged = df[keys].merge(
                    map_.rename("_te").reset_index(), on=keys, how="left"
                )["_te"].fillna(global_mean).values
                df.loc[df.index, col] = merged

        val_rows = train_feat.iloc[val_idx]
        train_feat.loc[val_rows.index, "Driver_TE"] = val_rows["Driver"].map(driver_map).fillna(global_mean).values
        train_feat.loc[val_rows.index, "Race_TE"] = val_rows["Race"].map(race_map).fillna(global_mean).values
        ry = val_rows[["Race", "Year"]].merge(
            ry_map.rename("_te").reset_index(), on=["Race", "Year"], how="left"
        )["_te"].fillna(global_mean).values
        train_feat.loc[val_rows.index, "Race_Year_TE"] = ry

    # For test/ext use full train encodings
    full_driver_map = train_feat.groupby("Driver")[TARGET].mean()
    full_race_map = train_feat.groupby("Race")[TARGET].mean()
    full_ry_map = train_feat.groupby(["Race", "Year"])[TARGET].mean()

    for df in [test_feat, ext_feat]:
        df["Driver_TE"] = df["Driver"].map(full_driver_map).fillna(global_mean).values
        df["Race_TE"] = df["Race"].map(full_race_map).fillna(global_mean).values
        ry = df[["Race", "Year"]].merge(
            full_ry_map.rename("_te").reset_index(), on=["Race", "Year"], how="left"
        )["_te"].fillna(global_mean).values
        df["Race_Year_TE"] = ry

    return train_feat, ext_feat, test_feat
