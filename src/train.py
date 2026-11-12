"""Train and tune Logistic Regression.

Grid: models x 2 imbalance methods (class weights, SMOTE) x 2 feature sets
(full, no_health). Each combination is tuned with stratified 5-fold CV on the
training set only, scored on recall, precision, F1, ROC-AUC and accuracy, and
refit on recall (CLAUDE.md > Recall as the tuning objective). The test set is
never loaded here; it belongs to src/evaluate.py.

All preprocessing and SMOTE sit inside an imblearn Pipeline, so they are
fitted on the training folds only.

Run:  python -m src.train                    (all models)
      python -m src.train --models logreg    (subset)
Table  -> reports/results/cv_results.csv
Models -> models/<model>_<imbalance>_<feature_set>.joblib

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src import config as C
from src.data import flag_columns, load_train
from src.features import build_preprocessor

MODELS = ("logreg",)
IMBALANCE_METHODS = ("class_weight", "smote")

# Hyperparameter grids (keys are prefixed with "model__" when searched).
PARAM_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
}


def _make_model(model: str, balanced: bool):
    seed = C.RANDOM_STATE
    if model == "logreg":
        return LogisticRegression(
            class_weight="balanced" if balanced else None, max_iter=2000, random_state=seed
        )
    raise ValueError(f"Unknown model {model!r}; use one of {MODELS}")


def build_pipeline(
    model: str,
    imbalance: str,
    feature_set: str,
    flag_cols: list[str] | tuple[str, ...] = (),
) -> Pipeline:
    """Unfitted imblearn Pipeline: preprocessing -> [SMOTE] -> classifier.

    imbalance:  "class_weight" (class_weight="balanced") or "smote"
                (SMOTE on the training folds only, no class weights).
    """
    if imbalance not in IMBALANCE_METHODS:
        raise ValueError(f"Unknown imbalance method {imbalance!r}; use one of {IMBALANCE_METHODS}")
    balanced = imbalance == "class_weight"
    steps = [("prep", build_preprocessor(feature_set, scale=model == "logreg", flag_cols=flag_cols))]
    if not balanced:
        # ponytail: plain SMOTE on encoded features (as the issue asks); one-hot
        # columns get fractional values. SMOTENC is the upgrade if that matters.
        steps.append(("smote", SMOTE(random_state=C.RANDOM_STATE)))
    steps.append(("model", _make_model(model, balanced)))
    return Pipeline(steps)


def tune(
    model: str,
    imbalance: str,
    feature_set: str,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: dict | None = None,
) -> tuple[GridSearchCV, dict]:
    """Grid search with stratified 5-fold CV on the training data, refit on recall.

    Returns the fitted search and one results row: the best configuration's
    mean / std for every CV metric, so precision and F1 always sit next to recall.
    """
    pipe = build_pipeline(model, imbalance, feature_set, flag_columns(X))
    grid = param_grid if param_grid is not None else PARAM_GRIDS[model]
    search = GridSearchCV(
        pipe,
        {f"model__{k}": v for k, v in grid.items()},
        scoring=C.CV_SCORING,
        refit="recall",
        cv=C.make_cv(),
        n_jobs=-1,
    )
    search.fit(X, y)

    i = search.best_index_
    res = search.cv_results_
    row = {"model": model, "imbalance": imbalance, "feature_set": feature_set}
    for metric in C.CV_METRICS:
        row[f"{metric}_mean"] = res[f"mean_test_{metric}"][i]
        row[f"{metric}_std"] = res[f"std_test_{metric}"][i]
    row["best_params"] = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    # A model that says Yes to everyone has precision = Yes rate; flag anything at or below it.
    row["low_precision"] = bool(row["precision_mean"] <= y.mean())
    return search, row


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
                search, row = tune(model, imbalance, feature_set, X_train, y_train)
                joblib.dump(search.best_estimator_, model_path(model, imbalance, feature_set))
                rows.append(row)
                print(f"  {model:13s} {imbalance:12s} {feature_set:9s} "
                      f"recall {row['recall_mean']:.3f}  precision {row['precision_mean']:.3f}  "
                      f"f1 {row['f1_mean']:.3f}  roc_auc {row['roc_auc_mean']:.3f}  {row['best_params']}")
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
    main()
