import numpy as np
import pandas as pd

TARGET = "PitNextLap"
ID_COL = "id"

COMMON_COLS = [
    "id", "Driver", "Race", "Year", "Compound", "PitStop",
    "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]

FEATURE_COLS = [
    "TyreLife", "Stint", "LapNumber", "Position", "RaceProgress",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "Position_Change",
    "PitStop",
    # engineered
    "LT_lag1", "LT_lag2", "LTD_lag1", "TL_lag1", "PitStop_lag1",
    "LT_roll3_mean", "LT_roll3_std", "LTD_roll3_mean",
    "NormTyreLife", "Deg_per_lap",
    "EstTotalLaps", "LapsRemaining", "LT_race_compound_mean", "LT_vs_pace",
    "TL_x_Stint", "RP_x_TL", "LR_x_TL",
    "is_year2023", "is_pretesting", "is_real_driver", "Compound_ord",
    # year as numeric
    "Year",
]

_COMPOUND_ORD = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}


def _assign_ext_ids(ext: pd.DataFrame) -> pd.DataFrame:
    ext = ext.copy()
    ext[ID_COL] = range(-1, -len(ext) - 1, -1)
    return ext


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features in-place on the combined train+test+external frame.
    df must be sorted by [Driver, Race, Year, LapNumber] before calling, OR
    this function will sort it internally.
    """
    df = df.sort_values(["Driver", "Race", "Year", "LapNumber"]).copy()
    g = df.groupby(["Driver", "Race", "Year"], sort=False)

    # --- A: Lag / rolling features ---
    df["LT_lag1"] = g["LapTime (s)"].shift(1)
    df["LT_lag2"] = g["LapTime (s)"].shift(2)
    df["LTD_lag1"] = g["LapTime_Delta"].shift(1)
    df["TL_lag1"] = g["TyreLife"].shift(1)
    df["PitStop_lag1"] = g["PitStop"].shift(1)

    lt_shifted = g["LapTime (s)"].shift(1)
    df["LT_roll3_mean"] = (
        lt_shifted.groupby(df["Driver"].astype(str) + "_" + df["Race"] + "_" + df["Year"].astype(str))
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["LT_roll3_std"] = (
        lt_shifted.groupby(df["Driver"].astype(str) + "_" + df["Race"] + "_" + df["Year"].astype(str))
        .transform(lambda x: x.rolling(3, min_periods=1).std())
    )
    ltd_shifted = g["LapTime_Delta"].shift(1)
    df["LTD_roll3_mean"] = (
        ltd_shifted.groupby(df["Driver"].astype(str) + "_" + df["Race"] + "_" + df["Year"].astype(str))
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # --- B: Stint features ---
    stint_max = df.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"].transform("max")
    df["NormTyreLife"] = df["TyreLife"] / stint_max.clip(lower=1)
    df["Deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)

    # --- C: Race context features ---
    df["EstTotalLaps"] = (df["LapNumber"] / df["RaceProgress"].clip(lower=0.01)).round()
    df["LapsRemaining"] = df["EstTotalLaps"] - df["LapNumber"]
    df["LT_race_compound_mean"] = df.groupby(["Race", "Year", "Compound"])["LapTime (s)"].transform("mean")
    df["LT_vs_pace"] = df["LapTime (s)"] - df["LT_race_compound_mean"]

    # --- D: Interaction & flag features ---
    df["TL_x_Stint"] = df["TyreLife"] * df["Stint"]
    df["RP_x_TL"] = df["RaceProgress"] * df["TyreLife"]
    df["LR_x_TL"] = df["LapsRemaining"] * df["TyreLife"]
    df["is_year2023"] = (df["Year"] == 2023).astype(np.int8)
    df["is_pretesting"] = (df["Race"] == "Pre-Season Testing").astype(np.int8)
    df["is_real_driver"] = (~df["Driver"].astype(str).str.match(r"^D\d+$")).astype(np.int8)
    df["Compound_ord"] = df["Compound"].map(_COMPOUND_ORD).fillna(5).astype(np.int8)

    return df


def prepare_all(train: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame):
    """
    Full pipeline:
      1. Combine train (no target) + test + external for feature engineering
      2. Compute features on combined frame
      3. Split back into train_feat, test_feat, ext_feat
    Returns train_feat (with PitNextLap), test_feat, ext_feat (with PitNextLap)
    """
    train_ids = set(train[ID_COL])
    test_ids = set(test[ID_COL])

    ext = _assign_ext_ids(ext)
    ext_ids = set(ext[ID_COL])

    # Drop target from train for combining
    train_no_target = train.drop(columns=[TARGET], errors="ignore")

    # Align columns — external may have Normalized_TyreLife, drop it for combining
    ext_common = ext[[c for c in COMMON_COLS if c in ext.columns]].copy()
    if ID_COL not in ext_common.columns:
        ext_common[ID_COL] = ext[ID_COL].values

    all_df = pd.concat(
        [train_no_target[COMMON_COLS], test[COMMON_COLS], ext_common],
        ignore_index=True,
    )

    all_df = build_features(all_df)

    train_feat = all_df[all_df[ID_COL].isin(train_ids)].merge(
        train[[ID_COL, TARGET]], on=ID_COL, how="left"
    )
    test_feat = all_df[all_df[ID_COL].isin(test_ids)]
    ext_feat = all_df[all_df[ID_COL].isin(ext_ids)].merge(
        ext[[ID_COL, TARGET]], on=ID_COL, how="left"
    )

    return train_feat, test_feat, ext_feat


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return model-ready feature columns present in df."""
    return [c for c in FEATURE_COLS if c in df.columns]
