"""Loading, cleaning and splitting.

The odd-value rule (CLAUDE.md > Known data issues) is applied here:
rows with semester 0 or monthly cost 0 are dropped if they make up 2% of the
data or less; otherwise the values are set to missing, a flag column is added
for each, and median imputation happens later inside the model pipeline.
Rows are never dropped or changed based on the target value.

Run:  python -m src.data
Writes data/processed/clean.csv, train.csv and test.csv.

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


# ---------------------------------------------------------------- loading
def load_raw() -> pd.DataFrame:
    """Read the raw Kaggle CSV (never modified) and check its columns."""
    df = pd.read_csv(C.RAW_CSV, encoding=C.RAW_ENCODING)
    df.columns = df.columns.str.strip()
    missing = set(C.RAW_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(C.RAW_COLUMNS)
    if missing or extra:
        raise ValueError(f"Unexpected columns. Missing: {missing}, extra: {extra}")
    return df[C.RAW_COLUMNS]


# ---------------------------------------------------------------- odd values
def odd_value_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean mask per odd-value rule, plus the combined 'any' mask."""
    masks = {col: df[col] == value for col, value in C.ODD_VALUE_RULES.items()}
    masks["any"] = np.logical_or.reduce(list(masks.values()))
    return masks


def odd_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of odd rows (separately and combined) and their Yes/No split.

    Run on the raw data, before cleaning. Aggregates only, no individual rows.
    """
    rows = []
    for name, mask in odd_value_masks(df).items():
        label = "combined (either)" if name == "any" else f"{name} == {C.ODD_VALUE_RULES[name]}"
        target = df.loc[mask, C.TARGET]
        rows.append({
            "rule": label,
            "rows": int(mask.sum()),
            "share_of_data": mask.mean(),
            "yes": int((target == C.POSITIVE_LABEL).sum()),
            "no": int((target == C.NEGATIVE_LABEL).sum()),
        })
    report = pd.DataFrame(rows)
    report["action"] = odd_value_action(df)
    return report


def odd_value_action(df: pd.DataFrame) -> str:
    """'drop' if the combined odd rows are <= threshold, else 'flag_and_impute'."""
    share = odd_value_masks(df)["any"].mean()
    return "drop" if share <= C.ODD_VALUE_THRESHOLD else "flag_and_impute"


# ---------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the odd-value rule and encode the target as 0/1.

    Returns a new DataFrame; the input is not changed.
    """
    df = df.copy()
    masks = odd_value_masks(df)

    if odd_value_action(df) == "drop":
        df = df.loc[~masks["any"]]
    else:
        # Keep the rows: flag, then set to missing. Median imputation happens
        # inside the model pipeline (src/features.py) so it is fitted on
        # training folds only.
        for col in C.ODD_VALUE_RULES:
            df[C.ODD_VALUE_FLAGS[col]] = masks[col].astype(int)
            df[col] = df[col].astype(float).mask(masks[col])

    df[C.TARGET] = df[C.TARGET].map(C.TARGET_MAP)
    if df[C.TARGET].isna().any():
        raise ValueError(f"Unexpected values in {C.TARGET}")
    df[C.TARGET] = df[C.TARGET].astype(int)
    return df.reset_index(drop=True)


def flag_columns(df: pd.DataFrame) -> list[str]:
    """Odd-value flag columns present in df (empty when odd rows were dropped)."""
    return [f for f in C.ODD_VALUE_FLAGS.values() if f in df.columns]


# ---------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame):
    """Stratified 80/20 split with the fixed seed.

    Returns X_train, X_test, y_train, y_test. X keeps all feature columns
    (and any flag columns); pick a feature set with config.FEATURE_SETS.
    """
    X = df.drop(columns=C.TARGET)
    y = df[C.TARGET]
    return train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )


def _load_split(path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    df = pd.read_csv(path)
    return df.drop(columns=C.TARGET), df[C.TARGET]


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """X_train, y_train from data/processed/train.csv (target is 0/1)."""
    return _load_split(C.TRAIN_CSV)


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """X_test, y_test. Only src/evaluate.py may call this, after tuning is done."""
    return _load_split(C.TEST_CSV)


def load_clean() -> pd.DataFrame:
    """The full cleaned dataset (used by EDA)."""
    if not C.CLEAN_CSV.exists():
        raise FileNotFoundError(f"{C.CLEAN_CSV} not found. Run `python -m src.data` first.")
    return pd.read_csv(C.CLEAN_CSV)


# ---------------------------------------------------------------- main
def main() -> None:
    raw = load_raw()
    print(f"Raw data: {len(raw)} rows, {raw.shape[1]} columns")
    print(f"Missing values: {int(raw.isna().sum().sum())}, duplicate rows: {int(raw.duplicated().sum())}")

    report = odd_value_report(raw)
    print("\nOdd values (raw data):")
    print(report.to_string(index=False, formatters={"share_of_data": "{:.2%}".format}))

    df = clean(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    print(f"\nCleaned data: {len(df)} rows -> {C.CLEAN_CSV.relative_to(C.ROOT)}")

    X_train, X_test, y_train, y_test = split_data(df)
    pd.concat([X_train, y_train], axis=1).to_csv(C.TRAIN_CSV, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(C.TEST_CSV, index=False)
    for name, y in [("train", y_train), ("test", y_test)]:
        print(f"{name:5s}: {len(y)} rows, Yes rate {y.mean():.1%}")


if __name__ == "__main__":
    main()
