"""Majority-class baseline: predicts "No" for every student.

Every model must beat this reference on recall and F1 (CLAUDE.md > Evaluation).
It is scored with stratified 5-fold CV on the training set only; test-set
scores come only from src/evaluate.py.

Run:  python -m src.baseline   (run `python -m src.data` first)
Table -> reports/results/baseline_cv.csv
Models -> models/baseline_full.joblib, models/baseline_no_health.joblib

The saved baselines include the same preprocessor as the tuned models, so
src/evaluate.py and the demo can pass raw feature columns to every pipeline
in the same way. DummyClassifier ignores the features, so both feature sets
give identical scores; they are kept separate so each comparison is like for like.

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src import config as C
from src.data import flag_columns, load_train
from src.features import build_preprocessor

MODEL_NAME = "majority_class"


def build_baseline(feature_set: str, flag_cols: list[str] | tuple[str, ...] = ()) -> Pipeline:
    """Unfitted pipeline: preprocessing for the feature set + most-frequent classifier."""
    return Pipeline([
        ("prep", build_preprocessor(feature_set, scale=False, flag_cols=flag_cols)),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])


def cv_scores(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Mean and std of each CV metric (stratified 5-fold, fixed seed)."""
    scores = cross_validate(pipe, X, y, cv=C.make_cv(), scoring=C.CV_SCORING)
    row = {}
    for metric in C.CV_METRICS:
        fold_scores = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = fold_scores.mean()
        row[f"{metric}_std"] = fold_scores.std()
    return row


def model_path(feature_set: str):
    return C.MODELS_DIR / f"baseline_{feature_set}.joblib"


def main() -> None:
    X_train, y_train = load_train()  # training set only, never the test set
    flags = flag_columns(X_train)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training set: {len(y_train)} rows, Yes rate {y_train.mean():.1%}")
    rows = []
    for feature_set in C.FEATURE_SETS:
        pipe = build_baseline(feature_set, flags)
        rows.append({"feature_set": feature_set, "model": MODEL_NAME,
                     **cv_scores(pipe, X_train, y_train)})

        pipe.fit(X_train, y_train)
        predicted = pipe.predict(X_train.head(1))[0]
        if predicted != C.TARGET_MAP[C.NEGATIVE_LABEL]:
            raise RuntimeError(f"Baseline for {feature_set} predicts {predicted}, expected 'No' (0)")
        path = model_path(feature_set)
        joblib.dump(pipe, path)
        print(f"  model   {path.relative_to(C.ROOT)}  (always predicts '{C.NEGATIVE_LABEL}')")

    table = pd.DataFrame(rows)
    path = C.RESULTS_DIR / "baseline_cv.csv"
    table.to_csv(path, index=False)
    print(f"  table   {path.relative_to(C.ROOT)}")

    print(f"\nBaseline, {C.CV_FOLDS}-fold CV on the training set (mean +/- std):")
    summary = pd.DataFrame({
        m: [f"{r[f'{m}_mean']:.3f} +/- {r[f'{m}_std']:.3f}" for r in rows] for m in C.CV_METRICS
    }, index=table["feature_set"])
    print(summary.to_string())
    print(f"\n{C.SCREENING_NOTE}.")


if __name__ == "__main__":
    main()
