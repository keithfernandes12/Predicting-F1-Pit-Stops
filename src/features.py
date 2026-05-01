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
    # lag / rolling
    "LT_lag1", "LT_lag2", "LTD_lag1", "TL_lag1", "PitStop_lag1",
    "CD_lag1",
    "LT_roll3_mean", "LT_roll3_std", "LT_roll5_std", "LTD_roll3_mean",
    "CD_roll3_mean",
    # stint
    "NormTyreLife", "TyreLife_compound_pct", "Deg_per_lap",
    "tyre_life_vs_field_max",
    # degradation trend
    "deg_acceleration",
    # race context
    "EstTotalLaps", "LapsRemaining", "LapsRemaining_clip",
    "LT_race_compound_mean", "LT_vs_pace",
    # interactions
    "TL_x_Stint", "RP_x_TL", "LR_x_TL",
    "LT_acceleration", "Deg_x_NormTL",
    "LapsRemaining_x_NormTL",
    # position trend
    "position_trend",
    # flags
    "is_year2023", "is_pretesting", "is_real_driver", "Compound_ord",
    "Year",
]

_COMPOUND_ORD = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

# Typical max stint length by compound (in laps) — derived from domain knowledge
# Used to normalise TyreLife independent of observed stint completeness
_COMPOUND_TYPICAL_MAX = {"SOFT": 20, "MEDIUM": 30, "HARD": 40, "INTERMEDIATE": 25, "WET": 20}


def _assign_ext_ids(ext: pd.DataFrame) -> pd.DataFrame:
    ext = ext.copy()
    ext[ID_COL] = range(-1, -len(ext) - 1, -1)
    return ext


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features on the combined train+test+external frame.
    Frame is sorted internally; original index preserved for id-based splitting.
    """
    df = df.sort_values(["Driver", "Race", "Year", "LapNumber"]).copy()

    # Single groupby key reused for all operations
    grp_key = ["Driver", "Race", "Year"]
    g = df.groupby(grp_key, sort=False)

    # --- A: Lag features ---
    df["LT_lag1"] = g["LapTime (s)"].shift(1)
    df["LT_lag2"] = g["LapTime (s)"].shift(2)
    df["LTD_lag1"] = g["LapTime_Delta"].shift(1)
    df["TL_lag1"] = g["TyreLife"].shift(1)
    df["PitStop_lag1"] = g["PitStop"].shift(1)
    df["CD_lag1"] = g["Cumulative_Degradation"].shift(1)

    # Rolling features: shift(1) inside transform ensures no current-lap leakage
    df["LT_roll3_mean"] = g["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["LT_roll3_std"] = g["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    )
    df["LT_roll5_std"] = g["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )
    df["LTD_roll3_mean"] = g["LapTime_Delta"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["CD_roll3_mean"] = g["Cumulative_Degradation"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # --- B: Stint features ---
    stint_max = df.groupby(["Driver", "Race", "Year", "Stint"])["TyreLife"].transform("max")
    df["NormTyreLife"] = df["TyreLife"] / stint_max.clip(lower=1)
    df["Deg_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
    # TyreLife normalised by compound-typical max — doesn't depend on observed stint length
    compound_max = df["Compound"].map(_COMPOUND_TYPICAL_MAX).fillna(25)
    df["TyreLife_compound_pct"] = df["TyreLife"] / compound_max
    # Data-driven: max tyre life observed at this compound in this race/year (field-wide context)
    df["field_max_tyre_life"] = df.groupby(["Race", "Year", "Compound"])["TyreLife"].transform("max")
    df["tyre_life_vs_field_max"] = df["TyreLife"] / df["field_max_tyre_life"].clip(lower=1)
    # Degradation acceleration: how much degradation occurred since last lap
    df["deg_acceleration"] = df["Cumulative_Degradation"] - df["CD_lag1"]

    # --- C: Race context features ---
    df["EstTotalLaps"] = (df["LapNumber"] / df["RaceProgress"].clip(lower=0.01)).round()
    df["LapsRemaining"] = df["EstTotalLaps"] - df["LapNumber"]
    df["LT_race_compound_mean"] = df.groupby(["Race", "Year", "Compound"])["LapTime (s)"].transform("mean")
    df["LT_vs_pace"] = df["LapTime (s)"] - df["LT_race_compound_mean"]

    # --- C2: Race context extras ---
    df["LapsRemaining_clip"] = df["LapsRemaining"].clip(lower=0)

    # --- D: Interaction & flag features ---
    df["TL_x_Stint"] = df["TyreLife"] * df["Stint"]
    df["RP_x_TL"] = df["RaceProgress"] * df["TyreLife"]
    df["LR_x_TL"] = df["LapsRemaining"] * df["TyreLife"]

    # second derivative of lap time — accelerating degradation is a pit signal
    df["LT_acceleration"] = df["LapTime_Delta"] - df["LTD_lag1"]

    # degradation × NormTyreLife — worn tyre degrading fast
    df["Deg_x_NormTL"] = df["Cumulative_Degradation"] * df["NormTyreLife"]

    # how many laps left relative to how far through the stint — urgency signal
    df["LapsRemaining_x_NormTL"] = df["LapsRemaining_clip"] * df["NormTyreLife"]

    df["is_year2023"] = (df["Year"] == 2023).astype(np.int8)
    df["is_pretesting"] = (df["Race"] == "Pre-Season Testing").astype(np.int8)
    df["is_real_driver"] = (~df["Driver"].astype(str).str.match(r"^D\d+$")).astype(np.int8)
    df["Compound_ord"] = df["Compound"].map(_COMPOUND_ORD).fillna(5).astype(np.int8)

    # Position trend over last 3 laps (positive = losing places, negative = gaining)
    df["Position_lag3"] = g["Position"].shift(3)
    df["position_trend"] = df["Position"] - df["Position_lag3"]

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
