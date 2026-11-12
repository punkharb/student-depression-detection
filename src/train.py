"""Train and tune Logistic Regression, Random Forest and XGBoost.

Grid: 3 models x 2 imbalance methods (class weights, SMOTENC) x 2 feature sets
(full, no_health). For each combination, on the training set only:

1. Randomized search, stratified 5-fold CV, selects hyperparameters by ROC-AUC
   (how well the model ranks Yes above No, independent of any cut-off).
2. The decision threshold is tuned to maximise F1 with an inner stratified
   5-fold CV (sklearn TunedThresholdClassifierCV). This raises recall over the
   default 0.5 cut-off without collapsing to "Yes for everyone" (F2 did that
   on no_health: precision 0.32, accuracy 0.43).
3. Scores come from an outer 5-fold CV in which the threshold is re-tuned on
   every outer training fold, so recall / precision / F1 are not
   optimistic about the threshold. The hyperparameters were chosen on these
   same folds, so all scores (ROC-AUC most of all) carry the usual
   CV-selection optimism. The untouched test set (src/evaluate.py) gives the
   unbiased estimate.

All preprocessing and SMOTENC sit inside an imblearn Pipeline, so they are
fitted on the training folds only. The test set is never loaded here; it
belongs to src/evaluate.py.

Run:  python -m src.train                    (all models)
      python -m src.train --models logreg    (subset)
Table  -> reports/results/cv_results.csv
Models -> models/<model>_<imbalance>_<feature_set>.joblib

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from scipy.stats import loguniform, randint, uniform
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TunedThresholdClassifierCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src import config as C
from src.data import flag_columns, load_train
from src.features import build_preprocessor

MODELS = ("logreg", "random_forest", "xgboost")
IMBALANCE_METHODS = ("class_weight", "smote")

N_ITER = 40  # random configurations tried per model / imbalance / feature set

# Hyperparameter search spaces (prefixed with "model__" when searched).
PARAM_DISTS = {
    "logreg": {
        "C": loguniform(1e-3, 1e2),
        "l1_ratio": uniform(0, 1),  # 0 = L2, 1 = L1, between = elastic net
    },
    "random_forest": {
        "max_depth": [None, 4, 6, 8, 12, 16],
        "min_samples_leaf": randint(1, 30),
        "max_features": ["sqrt", 0.3, 0.5, 0.8, 1.0],
    },
    "xgboost": {
        "n_estimators": randint(100, 800),
        "max_depth": randint(2, 7),
        "learning_rate": loguniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": loguniform(1, 20),
        "reg_lambda": loguniform(0.1, 10),
    },
}
SMOTE_DISTS = {"k_neighbors": [3, 5, 7, 10], "sampling_strategy": [0.5, 0.75, 1.0]}


class FoldBalancedXGB(XGBClassifier):
    """XGBoost with scale_pos_weight = negatives / positives of the data passed
    to fit, i.e. of each CV training fold (never the validation fold)."""

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        return super().fit(X, y, **kwargs)


def _make_model(model: str, balanced: bool):
    seed = C.RANDOM_STATE
    if model == "logreg":
        return LogisticRegression(
            class_weight="balanced" if balanced else None,
            solver="saga", max_iter=5000, random_state=seed,
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced" if balanced else None,
            n_jobs=-1, random_state=seed,
        )
    if model == "xgboost":
        return (FoldBalancedXGB if balanced else XGBClassifier)(
            eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown model {model!r}; use one of {MODELS}")


def build_pipeline(
    model: str,
    imbalance: str,
    feature_set: str,
    flag_cols: list[str] | tuple[str, ...] = (),
) -> Pipeline:
    """Unfitted imblearn Pipeline.

    imbalance:  "class_weight": preprocessing -> classifier with
                class_weight="balanced" (XGBoost: scale_pos_weight = neg/pos of
                each training fold).
                "smote": select feature-set columns (median-impute numerics)
                -> SMOTENC -> preprocessing -> classifier, no class weights.
                SMOTENC runs on the raw columns so synthetic students get real
                category labels, and only sees the feature set's columns
                (no_health never uses Health_condition).
    """
    if imbalance not in IMBALANCE_METHODS:
        raise ValueError(f"Unknown imbalance method {imbalance!r}; use one of {IMBALANCE_METHODS}")
    balanced = imbalance == "class_weight"
    steps = []
    if not balanced:
        cols = C.FEATURE_SETS[feature_set] + list(flag_cols)
        numeric = [c for c in cols if c in C.NUMERIC_COLS]
        categorical = [c for c in cols if c not in numeric]
        # SMOTENC cannot take NaN (flag-and-impute branch), so impute here too.
        select = ColumnTransformer(
            [("num", SimpleImputer(strategy="median"), numeric), ("keep", "passthrough", categorical)],
            remainder="drop", verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        steps += [
            ("select", select),
            ("smote", SMOTENC(categorical_features=categorical, random_state=C.RANDOM_STATE)),
        ]
    steps += [
        ("prep", build_preprocessor(feature_set, scale=model == "logreg", flag_cols=flag_cols)),
        ("model", _make_model(model, balanced)),
    ]
    return Pipeline(steps)


def search_space(model: str, imbalance: str) -> dict:
    space = {f"model__{k}": v for k, v in PARAM_DISTS[model].items()}
    if imbalance == "smote":
        space.update({f"smote__{k}": v for k, v in SMOTE_DISTS.items()})
    return space


def tune(
    model: str,
    imbalance: str,
    feature_set: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = N_ITER,
) -> tuple[TunedThresholdClassifierCV, dict]:
    """Search hyperparameters (ROC-AUC), then tune the threshold (F1). Training data only.

    Returns the fitted threshold-tuned model and one results row with nested-CV
    mean / std for every metric, so precision and F1 always sit next to recall.
    """
    pipe = build_pipeline(model, imbalance, feature_set, flag_columns(X))
    search = RandomizedSearchCV(
        pipe, search_space(model, imbalance), n_iter=n_iter, scoring="roc_auc",
        cv=C.make_cv(), random_state=C.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X, y)

    final = TunedThresholdClassifierCV(search.best_estimator_, scoring=C.CV_SCORING["f1"], cv=C.make_cv())
    # Outer CV re-tunes the threshold on each outer training fold; hyperparameters
    # were selected on these folds, so scores keep the usual selection optimism.
    scores = cross_validate(final, X, y, cv=C.make_cv(), scoring=C.CV_SCORING, n_jobs=-1)
    final.fit(X, y)

    row = {"model": model, "imbalance": imbalance, "feature_set": feature_set}
    for metric in C.CV_METRICS:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()
    row["threshold"] = final.best_threshold_
    row["search_roc_auc"] = search.best_score_
    row["best_params"] = {
        k.removeprefix("model__"): (round(v, 4) if isinstance(v, float) else v)
        for k, v in search.best_params_.items()
    }
    # A model that says Yes to everyone has precision = Yes rate; flag anything at or below it.
    row["low_precision"] = bool(row["precision_mean"] <= y.mean())
    return final, row


def merge_results(old: pd.DataFrame, new: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Replace the rows of re-run models, keep the rest (so --models can run piecemeal)."""
    return pd.concat([old[~old["model"].isin(models)], new], ignore_index=True)


def model_path(model: str, imbalance: str, feature_set: str):
    return C.MODELS_DIR / f"{model}_{imbalance}_{feature_set}.joblib"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args(argv)

    X_train, y_train = load_train()  # training set only, never the test set
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training set: {len(y_train)} rows, Yes rate {y_train.mean():.1%}")

    rows = []
    for model in args.models:
        for imbalance in IMBALANCE_METHODS:
            for feature_set in C.FEATURE_SETS:
                final, row = tune(model, imbalance, feature_set, X_train, y_train)
                joblib.dump(final, model_path(model, imbalance, feature_set))
                rows.append(row)
                print(f"  {model:13s} {imbalance:12s} {feature_set:9s} "
                      f"recall {row['recall_mean']:.3f}  precision {row['precision_mean']:.3f}  "
                      f"f1 {row['f1_mean']:.3f}  roc_auc {row['roc_auc_mean']:.3f}  "
                      f"threshold {row['threshold']:.2f}  {row['best_params']}")
                if row["low_precision"]:
                    print(f"    WARNING: precision {row['precision_mean']:.3f} is no better than "
                          f"predicting Yes for everyone. Flag to the team.")

    path = C.RESULTS_DIR / "cv_results.csv"
    table = pd.DataFrame(rows)
    if path.exists():
        table = merge_results(pd.read_csv(path), table, args.models)
    table.to_csv(path, index=False)
    print(f"  table   {path.relative_to(C.ROOT)}")
    print(f"\n{C.SCREENING_NOTE}.")


if __name__ == "__main__":
    # Run main() from the importable src.train module, not __main__, so pickled
    # models reference src.train.FoldBalancedXGB and load in evaluate.py / the demo.
    from src.train import main as _main

    _main()
