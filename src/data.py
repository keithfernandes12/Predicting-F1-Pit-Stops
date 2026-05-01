import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "Dataset"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "test.csv")


def load_external() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "f1_strategy_dataset_v4.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


def dedup_external(train: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Remove external rows that already exist in train (by Driver/Race/Year/LapNumber)."""
    key_cols = ["Driver", "Race", "Year", "LapNumber"]
    train_keys = set(zip(train[key_cols[0]], train[key_cols[1]], train[key_cols[2]], train[key_cols[3]]))
    mask = [
        (d, r, y, l) not in train_keys
        for d, r, y, l in zip(ext["Driver"], ext["Race"], ext["Year"], ext["LapNumber"])
    ]
    return ext[mask].copy()


def load_all():
    train = load_train()
    test = load_test()
    ext_raw = load_external()
    ext = dedup_external(train, ext_raw)
    sample_sub = load_sample_submission()
    print(f"Train: {len(train):,} rows  |  Test: {len(test):,} rows  |  External (unique): {len(ext):,} rows")
    return train, test, ext, sample_sub
