"""Preprocessing as an (unfitted) ColumnTransformer.

Put the result inside an imblearn.pipeline.Pipeline together with the model
(and SMOTE if used), so encoding, scaling and imputation are fitted on the
training folds only. Never fit it on the full dataset or the test set.

Example (in src/train.py):
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([
        ("prep", build_preprocessor("no_health", scale=True)),
        ("model", LogisticRegression(...)),
    ])
    X_train, y_train = load_train()
    pipe.fit(X_train, y_train)   # the preprocessor selects the feature-set columns itself

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config as C


def build_preprocessor(
    feature_set: str, scale: bool, flag_cols: list[str] | tuple[str, ...] = ()
) -> ColumnTransformer:
    """Build the preprocessing step for one feature set.

    feature_set: "full" or "no_health" (see config.FEATURE_SETS).
    scale:       True for Logistic Regression (standardise numeric features),
                 False for tree models (numeric features left unscaled).
    flag_cols:   odd-value flag columns (src.data.flag_columns(df)). Empty with
                 the current data, because the odd rows were dropped.

    Columns not in the feature set (e.g. Health_condition for "no_health")
    are dropped.
    """
    if feature_set not in C.FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; use one of {list(C.FEATURE_SETS)}")
    features = C.FEATURE_SETS[feature_set]

    numeric = [c for c in C.NUMERIC_COLS if c in features]
    ordinal = [c for c in C.ORDINAL_COLS if c in features]
    onehot = [c for c in C.ONEHOT_COLS if c in features]

    # The median imputer does nothing when there are no missing values; it is
    # here so the flag-and-impute branch of the odd-value rule stays leak-free.
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = [
        ("num", Pipeline(numeric_steps), numeric),
        ("ord", OrdinalEncoder(
            categories=[C.ORDINAL_ORDERS[c] for c in ordinal],
            handle_unknown="use_encoded_value", unknown_value=-1,
        ), ordinal),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False), onehot),
    ]
    if flag_cols:
        transformers.append(("flag", "passthrough", list(flag_cols)))

    return ColumnTransformer(
        [t for t in transformers if t[2]],  # skip empty groups
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output column names of a *fitted* preprocessor (for coefficients / importances)."""
    return list(preprocessor.get_feature_names_out())
