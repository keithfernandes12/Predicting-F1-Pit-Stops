import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


# ── LightGBM ─────────────────────────────────────────────────────────────────

DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 255,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def make_lgb_fn(params: dict = None, early_stopping_rounds: int = 100):
    p = {**DEFAULT_LGB_PARAMS, **(params or {})}

    def model_fn(X_tr, y_tr, w_tr, X_val, y_val):
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        return model

    return model_fn


# ── XGBoost ──────────────────────────────────────────────────────────────────

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 3000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
}


def make_xgb_fn(params: dict = None, early_stopping_rounds: int = 100):
    p = {**DEFAULT_XGB_PARAMS, **(params or {})}

    def model_fn(X_tr, y_tr, w_tr, X_val, y_val):
        model = xgb.XGBClassifier(**p, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        return model

    return model_fn


# ── CatBoost ─────────────────────────────────────────────────────────────────

DEFAULT_CAT_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 0,
    "early_stopping_rounds": 100,
    "task_type": "CPU",
}


def make_cat_fn(params: dict = None):
    p = {**DEFAULT_CAT_PARAMS, **(params or {})}

    def model_fn(X_tr, y_tr, w_tr, X_val, y_val):
        model = CatBoostClassifier(**p)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
        return model

    return model_fn


# ── Full-train predictor (no CV, for final submission) ────────────────────────

def train_full(X_tr, y_tr, w_tr, model_name: str = "lgb", params: dict = None, n_estimators: int = None):
    """Train on the full dataset (no eval set) for final submission."""
    if model_name == "lgb":
        p = {**DEFAULT_LGB_PARAMS, **(params or {})}
        if n_estimators:
            p["n_estimators"] = n_estimators
        model = lgb.LGBMClassifier(**p)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_name == "xgb":
        p = {**DEFAULT_XGB_PARAMS, **(params or {})}
        if n_estimators:
            p["n_estimators"] = n_estimators
        model = xgb.XGBClassifier(**p)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    elif model_name == "cat":
        p = {**DEFAULT_CAT_PARAMS, **(params or {})}
        if n_estimators:
            p["iterations"] = n_estimators
        p.pop("early_stopping_rounds", None)
        model = CatBoostClassifier(**p)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
