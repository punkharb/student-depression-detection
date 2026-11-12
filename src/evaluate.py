"""Test-set evaluation: every tuned model and the baseline on the same test set.

Run only after tuning is finished (python -m src.baseline, python -m src.train).
This is the only script that loads the test set. Nothing here is fitted or
selected on it: the saved pipelines are only asked to predict.

Grid: 3 models x 2 imbalance methods x 2 feature sets, plus the majority-class
baseline for each feature set. Each tuned model predicts with the threshold
tuned for F1 on the training folds (src/train.py), not with 0.5.

Success criterion (CLAUDE.md > Evaluation): the best model beats the baseline
on F1 and recall. "Best" is the configuration with the best CV F1 per feature
set (the demo model, src/train.py), never the best test score, so the test set
is not used for model selection.

Run:  python -m src.evaluate
Table   -> reports/results/test_results.csv
Figures -> reports/figures/confusion_matrices_<feature_set>.png
           reports/figures/roc_curves_<feature_set>.png

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import baseline, train  # noqa: E402
from src import config as C  # noqa: E402
from src.data import load_test  # noqa: E402

BASELINE = baseline.MODEL_NAME
NO_IMBALANCE = "none"
TEST_METRICS = ["recall", "precision", "f1", "roc_auc", "accuracy"]  # recall first: priority

MODEL_LABELS = {
    BASELINE: "Baseline (always No)",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
IMBALANCE_LABELS = {NO_IMBALANCE: "", "class_weight": "class weights", "smote": "SMOTENC"}

# Categorical slots 1-3 of the dataviz reference palette, fixed order per model
# (checked with its validator: CVD and normal-vision separation pass).
MODEL_COLORS = {"logreg": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}
BASELINE_COLOR = "#8a8a85"
IMBALANCE_LINESTYLES = {"class_weight": "-", "smote": "--"}  # second channel, not colour alone


def label(model: str, imbalance: str) -> str:
    """Display name, e.g. 'XGBoost, SMOTENC'."""
    extra = IMBALANCE_LABELS[imbalance]
    return f"{MODEL_LABELS[model]}, {extra}" if extra else MODEL_LABELS[model]


# ---------------------------------------------------------------- loading
def load_models(feature_set: str) -> tuple[list[dict], list[str]]:
    """Saved pipelines for one feature set: the baseline first, then the tuned grid.

    The baseline is required. Missing tuned models are returned by name and
    skipped, so the script can run on the baselines alone before training is done.

    joblib unpickles, so it can run code: only load files this repo's own
    src/baseline.py and src/train.py wrote locally (models/ is not committed).
    """
    path = baseline.model_path(feature_set)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.baseline` first.")
    entries = [{"feature_set": feature_set, "model": BASELINE, "imbalance": NO_IMBALANCE,
                "pipe": joblib.load(path)}]
    missing = []
    for model in train.MODELS:
        for imbalance in train.IMBALANCE_METHODS:
            path = train.model_path(model, imbalance, feature_set)
            if path.exists():
                entries.append({"feature_set": feature_set, "model": model,
                                "imbalance": imbalance, "pipe": joblib.load(path)})
            else:
                missing.append(path.name)
    return entries, missing


def load_cv_results() -> pd.DataFrame | None:
    path = C.RESULTS_DIR / "cv_results.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------- scoring
def predict(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predicted labels (at the model's own threshold) and P(Yes)."""
    return np.asarray(pipe.predict(X)), pipe.predict_proba(X)[:, 1]


def score(y_true, y_pred, y_prob) -> dict:
    """Test metrics plus confusion counts. Recall first; accuracy never alone."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_of(pipe) -> float:
    """Decision threshold tuned in src/train.py (NaN for the baseline, which never says Yes)."""
    return float(getattr(pipe, "best_threshold_", np.nan))


def add_flags(table: pd.DataFrame, cv: pd.DataFrame | None, yes_rate: float) -> pd.DataFrame:
    """Mark the CV-chosen model, success vs baseline, and low precision.

    chosen_by_cv:   best CV F1 for the feature set (train.pick_demo), not a test score.
    beats_baseline: test F1 and recall both above that feature set's baseline.
    low_precision:  precision no better than saying Yes to everyone (= test Yes rate).
    """
    table = table.copy()
    base = table[table["model"] == BASELINE].set_index("feature_set")
    ref_f1 = table["feature_set"].map(base["f1"])
    ref_recall = table["feature_set"].map(base["recall"])
    tuned = table["model"] != BASELINE
    table["beats_baseline"] = tuned & (table["f1"] > ref_f1) & (table["recall"] > ref_recall)
    table["low_precision"] = tuned & (table["precision"] <= yes_rate)

    table["chosen_by_cv"] = False
    if cv is not None:
        for feature_set, row in train.pick_demo(cv).items():
            hit = ((table["feature_set"] == feature_set) & (table["model"] == row["model"])
                   & (table["imbalance"] == row["imbalance"]))
            table.loc[hit, "chosen_by_cv"] = True
        cv_cols = cv[["model", "imbalance", "feature_set", "recall_mean", "f1_mean"]].rename(
            columns={"recall_mean": "cv_recall", "f1_mean": "cv_f1"})
        table = table.merge(cv_cols, on=["model", "imbalance", "feature_set"], how="left")
    return table


# ---------------------------------------------------------------- figures
def plot_confusion_matrices(feature_set: str, preds: list[dict], y_true) -> plt.Figure:
    """One panel per model (baseline first): test-set counts, labels No / Yes."""
    n = len(preds)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.3 * rows), squeeze=False)
    for ax, p in zip(axes.flat, preds):
        cm = confusion_matrix(y_true, p["y_pred"], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=[C.NEGATIVE_LABEL, C.POSITIVE_LABEL]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        thr = "" if np.isnan(p["threshold"]) else f"  (threshold {p['threshold']:.2f})"
        ax.set_title(label(p["model"], p["imbalance"]) + thr, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Test-set confusion matrices: {feature_set} features", fontsize=12)
    fig.text(0.99, 0.005, f"{C.SCREENING_NOTE}. Test set, n = {len(y_true)}.",
             ha="right", va="bottom", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def plot_roc_curves(feature_set: str, preds: list[dict], y_true) -> plt.Figure:
    """ROC curve per model; dot = the model's operating point at its tuned threshold."""
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color=BASELINE_COLOR, linestyle=":", linewidth=1.5,
            label=f"{MODEL_LABELS[BASELINE]} (AUC 0.50)")
    for p in preds:
        if p["model"] == BASELINE:
            continue
        fpr, tpr, _ = roc_curve(y_true, p["y_prob"])
        color = MODEL_COLORS[p["model"]]
        ax.plot(fpr, tpr, color=color, linestyle=IMBALANCE_LINESTYLES[p["imbalance"]],
                linewidth=2, label=f"{label(p['model'], p['imbalance'])} (AUC {p['roc_auc']:.2f})")
        ax.plot(p["fp"] / (p["fp"] + p["tn"]), p["recall"], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (No students flagged Yes)")
    ax.set_ylabel("Recall / true positive rate (Yes students found)")
    ax.set_title(f"Test-set ROC curves: {feature_set} features\n"
                 "dot = operating point at the threshold tuned for F1", fontsize=11)
    ax.grid(color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.text(0.99, 0.005, f"{C.SCREENING_NOTE}. Test set, n = {len(y_true)}.",
             ha="right", va="bottom", fontsize=8, color="#5f5e5a")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def save(fig: plt.Figure, name: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure  {_show(path)}")


def _show(path) -> str:
    """Path relative to the repo when inside it (tests write to temp dirs)."""
    return str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else str(path)


# ---------------------------------------------------------------- report
def success_summary(table: pd.DataFrame) -> list[str]:
    """One line per feature set: does the CV-chosen model beat the baseline?"""
    lines = []
    for feature_set, group in table.groupby("feature_set", sort=False):
        base = group[group["model"] == BASELINE].iloc[0]
        chosen = group[group["chosen_by_cv"]]
        if chosen.empty:
            lines.append(f"  {feature_set:9s} no CV-chosen model (run python -m src.train): not checked")
            continue
        row = chosen.iloc[0]
        verdict = "PASS" if row["beats_baseline"] else "FAIL"
        lines.append(
            f"  {feature_set:9s} {verdict}  {label(row['model'], row['imbalance'])}: "
            f"recall {row['recall']:.3f} vs {base['recall']:.3f}, "
            f"F1 {row['f1']:.3f} vs {base['f1']:.3f} (baseline)")
    return lines


def main() -> None:
    X_test, y_test = load_test()  # the only place the test set is loaded
    yes_rate = float(y_test.mean())
    print(f"Test set: {len(y_test)} rows, Yes rate {yes_rate:.1%}")
    cv = load_cv_results()
    if cv is None:
        print("  note    reports/results/cv_results.csv not found; CV columns and success check skipped")

    rows = []
    for feature_set in C.FEATURE_SETS:
        entries, missing = load_models(feature_set)
        if missing:
            print(f"  WARNING {feature_set}: {len(missing)} tuned model(s) missing, "
                  f"run python -m src.train: {', '.join(missing)}")
        preds = []
        for e in entries:
            with warnings.catch_warnings():
                # The baseline gives every student P(Yes) = 0; that is expected.
                warnings.simplefilter("ignore", UserWarning)
                y_pred, y_prob = predict(e["pipe"], X_test)
            metrics = score(y_test, y_pred, y_prob)
            info = {"feature_set": feature_set, "model": e["model"], "imbalance": e["imbalance"],
                    "threshold": threshold_of(e["pipe"])}
            rows.append({**info, **metrics})
            preds.append({**info, **metrics, "y_pred": y_pred, "y_prob": y_prob})
        save(plot_confusion_matrices(feature_set, preds, y_test), f"confusion_matrices_{feature_set}.png")
        save(plot_roc_curves(feature_set, preds, y_test), f"roc_curves_{feature_set}.png")

    table = add_flags(pd.DataFrame(rows), cv, yes_rate)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RESULTS_DIR / "test_results.csv"
    table.to_csv(path, index=False)
    print(f"  table   {_show(path)}")

    shown = table.assign(config=[label(m, i) for m, i in zip(table["model"], table["imbalance"])])
    cols = ["feature_set", "config", "threshold", *TEST_METRICS, "chosen_by_cv", "beats_baseline"]
    print("\nTest-set scores (recall is the priority metric; accuracy only alongside the others):")
    print(shown[cols].to_string(index=False, float_format="{:.3f}".format))

    for _, row in table[table["low_precision"]].iterrows():
        print(f"  WARNING {row['feature_set']} {label(row['model'], row['imbalance'])}: test precision "
              f"{row['precision']:.3f} is no better than saying Yes to everyone. Flag to the team.")

    print("\nSuccess criterion (CV-chosen model beats the baseline on test F1 and recall):")
    print("\n".join(success_summary(table)))
    print(f"\n{C.SCREENING_NOTE}.")


if __name__ == "__main__":
    main()
