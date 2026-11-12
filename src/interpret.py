"""Interpretation: which factors drive the tuned models?

For each feature set (full, no_health) and each model type, explain the
imbalance variant with the best CV F1 (reports/results/cv_results.csv). The
test set plays no part: it is never loaded here and never picks a model.

Three views:
1. Logistic Regression coefficients, one per encoded column, mapped back to
   readable names ("Health condition: Critical"). Numeric features were
   standardised, so their coefficient is per 1 standard deviation; Family
   background is ordinal, so per step (lower middle -> middle -> rich). Text
   categories keep every level (no reference level is dropped; the elastic-net
   penalty makes them identifiable), so compare sign and size, not exact odds.
2. Built-in importances: Random Forest impurity decrease and XGBoost gain,
   with the one-hot levels summed back to the original feature. Impurity
   importance tends to favour numeric features with many split points.
3. Permutation importance for all three models on the same scale: the drop in
   ROC-AUC when one raw feature is shuffled. Measured on held-out folds of the
   TRAINING set (stratified 5-fold, seed 42; the pipeline is refitted on each
   fold's training part), so a deep forest that memorises its training rows
   does not inflate the result.

None of this is causal: the model learned associations in self-reported,
single-item survey data.

Run:  python -m src.interpret              (after python -m src.train)
      python -m src.interpret --repeats 5  (fewer shuffles, faster)
Figures -> reports/figures/lr_coefficients_<feature_set>.png
           reports/figures/tree_importances_<feature_set>.png
           reports/figures/permutation_importance_<feature_set>.png
Tables  -> reports/results/lr_coefficients.csv, reports/results/importances.csv
Summary -> reports/interpretation_summary.md (generated; do not edit by hand)

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

from src import config as C  # noqa: E402
from src import evaluate, train  # noqa: E402
from src.data import flag_columns, load_train  # noqa: E402

N_REPEATS = 10  # shuffles per feature per CV fold
TOP_N = 5       # factors listed per model in the summary
TREE_MODELS = ("random_forest", "xgboost")

# Diverging pair from the dataviz reference palette (validated: CVD and
# normal-vision separation pass). Red raises the predicted risk, blue lowers it.
RAISES_COLOR = "#e34948"
LOWERS_COLOR = "#2a78d6"
ZERO_COLOR = evaluate.BASELINE_COLOR
GRID_COLOR = "#e5e4df"
NOTE_COLOR = "#5f5e5a"


def pretty(col: str) -> str:
    """Readable feature name: 'friend_circle' -> 'Friend circle', 'CGPA' stays."""
    s = col.replace("_", " ")
    return s[0].upper() + s[1:]


def summary_path():
    return C.REPORTS_DIR / "interpretation_summary.md"


# ---------------------------------------------------------------- choosing models
def pick_per_model(cv: pd.DataFrame) -> pd.DataFrame:
    """Best imbalance variant by CV F1 for every (feature set, model). Never a test score."""
    best = cv.loc[cv.groupby(["feature_set", "model"])["f1_mean"].idxmax()]
    rank = {**{fs: i for i, fs in enumerate(C.FEATURE_SETS)}, **{m: i for i, m in enumerate(train.MODELS)}}
    return best.sort_values(["feature_set", "model"], key=lambda s: s.map(rank)).reset_index(drop=True)


def unwrap(final):
    """(fitted preprocessor, fitted classifier) inside a TunedThresholdClassifierCV pipeline."""
    pipe = final.estimator_
    return pipe.named_steps["prep"], pipe.named_steps["model"]


def readable_names(prep) -> pd.DataFrame:
    """One row per output column of a fitted preprocessor, in output order.

    column:  encoded name (as get_feature_names_out)
    feature: original CSV column it came from (for grouping)
    label:   readable name, e.g. 'Health condition: Critical', 'Gender: Male (vs Female)'
    Built from the fitted encoders, not by parsing the column strings.
    """
    rows = []
    for _, trans, cols in prep.transformers_:
        if trans == "drop" or len(cols) == 0:
            continue
        if isinstance(trans, OneHotEncoder):
            drop_idx = trans.drop_idx_ if trans.drop_idx_ is not None else [None] * len(cols)
            for col, cats, dropped in zip(cols, trans.categories_, drop_idx):
                for i, cat in enumerate(cats):
                    if dropped is not None and i == dropped:
                        continue
                    text = f"{pretty(col)}: {cat}"
                    if dropped is not None:
                        text += f" (vs {cats[dropped]})"
                    rows.append({"feature": col, "label": text})
        else:
            rows += [{"feature": col, "label": pretty(col)} for col in cols]
    out = pd.DataFrame(rows)
    out.insert(0, "column", list(prep.get_feature_names_out()))
    return out


# ---------------------------------------------------------------- the three views
def lr_coefficients(final) -> pd.DataFrame:
    """Signed LR coefficients (log-odds of Yes) and odds ratios, readable names."""
    prep, model = unwrap(final)
    out = readable_names(prep)
    out["coefficient"] = model.coef_[0]
    out["odds_ratio"] = np.exp(out["coefficient"])
    return out.sort_values("coefficient", ascending=False, ignore_index=True)


def builtin_importance(final) -> pd.DataFrame:
    """RF impurity / XGBoost gain importance, summed to original features, shares of 1."""
    prep, model = unwrap(final)
    names = readable_names(prep)
    if hasattr(model, "get_booster"):
        # Booster keys are f0, f1, ... (the preprocessor outputs a plain array);
        # columns never used in a split are absent and count as 0.
        gain = model.get_booster().get_score(importance_type="gain")
        raw = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
    else:
        raw = np.asarray(model.feature_importances_, dtype=float)
    names["importance"] = raw / raw.sum() if raw.sum() > 0 else raw
    grouped = names.groupby("feature", as_index=False)["importance"].sum()
    return grouped.sort_values("importance", ascending=False, ignore_index=True)


def cv_permutation_importance(final, X: pd.DataFrame, y: pd.Series, feature_set: str,
                              n_repeats: int = N_REPEATS) -> pd.DataFrame:
    """Drop in ROC-AUC when one raw feature is shuffled, on held-out training folds.

    The tuned pipeline (same hyperparameters) is refitted on each fold's training
    part and scored on its held-out part. ROC-AUC ignores the threshold, so the
    threshold step is not needed. Only the feature set's columns are permuted.
    """
    cols = C.FEATURE_SETS[feature_set] + flag_columns(X)
    X = X[cols]
    drops = []
    for train_idx, val_idx in C.make_cv().split(X, y):
        est = clone(final.estimator_).fit(X.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            est, X.iloc[val_idx], y.iloc[val_idx], scoring="roc_auc",
            n_repeats=n_repeats, random_state=C.RANDOM_STATE,
        )
        drops.append(result.importances)  # (n_features, n_repeats)
    drops = np.hstack(drops)
    out = pd.DataFrame({"feature": cols, "importance": drops.mean(axis=1),
                        "importance_std": drops.std(axis=1)})
    return out.sort_values("importance", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- figures
def _style(ax) -> None:
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _footer(fig, n_train: int, what: str) -> None:
    """Bottom-right note; `what` says how the numbers were made (ends before the n)."""
    fig.text(0.99, 0.005, f"{C.SCREENING_NOTE}. {what} (training set, n = {n_train}). Associations, not causes.",
             ha="right", va="bottom", fontsize=8, color=NOTE_COLOR)


def plot_lr_coefficients(feature_set: str, coefs: pd.DataFrame, imbalance: str, n_train: int) -> plt.Figure:
    """Signed horizontal bars: red raises risk, blue lowers it (sign also in the legend)."""
    data = coefs.iloc[::-1]  # largest positive at the top
    colors = [RAISES_COLOR if c > 0 else LOWERS_COLOR if c < 0 else ZERO_COLOR for c in data["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(data) + 1.6))
    ax.barh(data["label"], data["coefficient"], color=colors, height=0.65)
    zero = data["coefficient"] == 0
    ax.plot(np.zeros(zero.sum()), data.loc[zero, "label"], "o", color=ZERO_COLOR, markersize=6, zorder=3)
    ax.axvline(0, color="#3d3d3a", linewidth=1)
    ax.set_xlabel("Coefficient (change in log-odds of Yes; numeric features per 1 SD)")
    ax.set_title(f"Logistic Regression coefficients: {feature_set} features\n"
                 f"{evaluate.IMBALANCE_LABELS[imbalance].capitalize()}, best CV F1", fontsize=11)
    _style(ax)
    handles = [Patch(color=RAISES_COLOR, label="+ raises predicted risk"),
               Patch(color=LOWERS_COLOR, label="− lowers predicted risk")]
    if (data["coefficient"] == 0).any():
        handles.append(Line2D([], [], marker="o", linestyle="", color=ZERO_COLOR, label="0, dropped by the penalty"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    _footer(fig, n_train, "Coefficients of the fitted model")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _feature_panels(feature_set: str, tables: dict, xlabel: str, title: str, footer: str,
                    n_train: int, order: list[str] | None = None, errors: bool = False) -> plt.Figure:
    """One panel per model (shared feature order), one colour per model."""
    models = list(tables)
    if order is None:
        order = (pd.concat(tables.values()).groupby("feature")["importance"].mean()
                 .sort_values().index.tolist())  # ascending: top of the chart = most important
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 1.6, 0.36 * len(order) + 1.8),
                             sharey=True, squeeze=False)
    for ax, (model, imbalance) in zip(axes[0], models):
        t = tables[(model, imbalance)].set_index("feature").reindex(order)
        ax.barh([pretty(f) for f in order], t["importance"], color=evaluate.MODEL_COLORS[model], height=0.65,
                xerr=t["importance_std"] if errors else None,
                error_kw={"ecolor": "#3d3d3a", "elinewidth": 1, "capsize": 2})
        ax.set_title(evaluate.label(model, imbalance), fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.axvline(0, color="#3d3d3a", linewidth=1)
        _style(ax)
    fig.suptitle(f"{title}: {feature_set} features", fontsize=12)
    _footer(fig, n_train, footer)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def plot_tree_importances(feature_set, tables, n_train) -> plt.Figure:
    return _feature_panels(feature_set, tables, "Share of importance (RF impurity / XGBoost gain)",
                           "Built-in feature importance", "One-hot levels summed per feature",
                           n_train)


def plot_permutation(feature_set, tables, n_train) -> plt.Figure:
    return _feature_panels(feature_set, tables, "Drop in ROC-AUC when shuffled",
                           "Permutation importance", "Mean ± SD over held-out 5-fold CV folds", n_train, errors=True)


# ---------------------------------------------------------------- summary
def health_dominance(builtin: pd.DataFrame, perm: pd.DataFrame, cv: pd.DataFrame) -> dict:
    """Numbers for 'how much Health condition dominates' (full feature set + CV gap)."""
    out = {"builtin_share": {}, "perm": {}}
    for (model, imbalance), t in builtin[builtin["feature_set"] == "full"].groupby(["model", "imbalance"]):
        share = t.loc[t["feature"] == C.HEALTH_COL, "importance"]
        out["builtin_share"][model] = float(share.iloc[0]) if len(share) else 0.0
    for (model, imbalance), t in perm[perm["feature_set"] == "full"].groupby(["model", "imbalance"]):
        t = t.sort_values("importance", ascending=False)
        health = t.loc[t["feature"] == C.HEALTH_COL, "importance"]
        others = t.loc[t["feature"] != C.HEALTH_COL]
        out["perm"][model] = {
            "health": float(health.iloc[0]) if len(health) else 0.0,
            "rank": int((t["feature"] == C.HEALTH_COL).to_numpy().argmax()) + 1 if len(health) else None,
            "next_feature": others["feature"].iloc[0], "next": float(others["importance"].iloc[0]),
        }
    out["cv"] = {fs: row for fs, row in train.pick_demo(cv).items()}
    return out


def _top(table: pd.DataFrame, n: int = TOP_N) -> str:
    t = table.sort_values("importance", ascending=False).head(n)
    return ", ".join(f"{pretty(f)} ({v:.3f})" for f, v in zip(t["feature"], t["importance"]))


def summary_markdown(picks: pd.DataFrame, coefs: pd.DataFrame, builtin: pd.DataFrame,
                     perm: pd.DataFrame, cv: pd.DataFrame, n_train: int, n_repeats: int) -> str:
    """The takeaway, generated from the numbers (nothing typed by hand)."""
    dom = health_dominance(builtin, perm, cv)
    lines = [
        "# Interpretation summary",
        "",
        "Generated by `python -m src.interpret`. Do not edit by hand; re-run the script instead.",
        "",
        f"> **{C.SCREENING_NOTE}.** The label is self-reported and single-item, and these are "
        "associations the models learned, not causes.",
        "",
        "## Models explained",
        "",
        "Per feature set and model type, the imbalance variant with the best CV F1 "
        "(never a test score). Training set only.",
        "",
        "| Feature set | Model | Imbalance | CV F1 | CV ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    for _, r in picks.iterrows():
        lines.append(f"| {r['feature_set']} | {evaluate.MODEL_LABELS[r['model']]} | "
                     f"{evaluate.IMBALANCE_LABELS[r['imbalance']]} | {r['f1_mean']:.3f} | {r['roc_auc_mean']:.3f} |")

    lines += ["", "## How much Health condition dominates", ""]
    if "full" in dom["cv"] and "no_health" in dom["cv"]:
        f, n = dom["cv"]["full"], dom["cv"]["no_health"]
        lines.append(
            f"- **Removing it costs most of the performance.** Best CV F1 falls from {f['f1_mean']:.3f} "
            f"({evaluate.label(f['model'], f['imbalance'])}, `full`) to {n['f1_mean']:.3f} "
            f"({evaluate.label(n['model'], n['imbalance'])}, `no_health`); CV ROC-AUC from "
            f"{f['roc_auc_mean']:.3f} to {n['roc_auc_mean']:.3f}.")
    for model, p in dom["perm"].items():
        lines.append(
            f"- **{evaluate.MODEL_LABELS[model]}** (`full`): shuffling Health condition drops ROC-AUC by "
            f"{p['health']:.3f} (rank {p['rank']}); the next factor, {pretty(p['next_feature'])}, by {p['next']:.3f}.")
    for model, share in dom["builtin_share"].items():
        lines.append(f"- **{evaluate.MODEL_LABELS[model]}** built-in importance (`full`): "
                     f"Health condition holds {share:.0%} of the total.")

    lines += ["", "## Which factors matter most", "",
              f"Permutation importance: mean drop in ROC-AUC when the factor is shuffled "
              f"(held-out 5-fold CV of the training set, {n_repeats} shuffles per fold). Top {TOP_N}:", ""]
    for fs in C.FEATURE_SETS:
        sub = perm[perm["feature_set"] == fs]
        if sub.empty:
            continue
        lines.append(f"**`{fs}`**")
        lines.append("")
        for (model, imbalance), t in sub.groupby(["model", "imbalance"], sort=False):
            lines.append(f"- {evaluate.label(model, imbalance)}: {_top(t)}")
        consensus = sub.groupby("feature")["importance"].mean().sort_values(ascending=False)
        lines.append(f"- Averaged over the three models: "
                     f"{', '.join(pretty(f) for f in consensus.index[:TOP_N])}")
        lines.append("")

    lines += ["## Logistic Regression: direction of effect", "",
              "Largest coefficients (log-odds of Yes; numeric features per 1 SD). "
              "Positive raises the predicted risk, negative lowers it.", ""]
    for fs in C.FEATURE_SETS:
        sub = coefs[coefs["feature_set"] == fs]
        if sub.empty:
            continue
        top = sub.reindex(sub["coefficient"].abs().sort_values(ascending=False).index).head(TOP_N)
        lines.append(f"- `{fs}`: " + "; ".join(f"{l} ({c:+.2f})" for l, c in zip(top["label"], top["coefficient"])))

    lines += [
        "", "## Caveats", "",
        "- Built-in RF / XGBoost importance tends to favour numeric features with many split points "
        "(Age, CGPA, cost); permutation importance is the fairer comparison across models.",
        "- Text categories keep every level in Logistic Regression, so compare coefficient sign and size, "
        "not exact odds ratios.",
        "- Features that carry the same information share importance; a low score does not prove a factor is unrelated.",
        f"- Computed on the training set only (n = {n_train}); the test set is used only in `src/evaluate.py`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def _tag(df: pd.DataFrame, **info) -> pd.DataFrame:
    return df.assign(**info)[[*info, *df.columns]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=N_REPEATS, help="shuffles per feature per CV fold")
    args = parser.parse_args(argv)

    cv = evaluate.load_cv_results()
    if cv is None:
        raise FileNotFoundError("reports/results/cv_results.csv not found. Run `python -m src.train` first.")
    X, y = load_train()  # training set only; the test set belongs to src/evaluate.py
    print(f"Training set: {len(y)} rows, Yes rate {y.mean():.1%}")

    picks = pick_per_model(cv)
    coef_rows, builtin_rows, perm_rows, kept = [], [], [], []
    for feature_set in C.FEATURE_SETS:
        tree_tables, perm_tables = {}, {}
        for _, row in picks[picks["feature_set"] == feature_set].iterrows():
            model, imbalance = row["model"], row["imbalance"]
            path = train.model_path(model, imbalance, feature_set)
            if not path.exists():
                print(f"  WARNING {path.name} missing (run python -m src.train); skipped")
                continue
            kept.append(row)
            # joblib unpickles: only load files this repo's src/train.py wrote locally.
            final = joblib.load(path)
            info = {"feature_set": feature_set, "model": model, "imbalance": imbalance}
            print(f"  {feature_set:9s} {evaluate.label(model, imbalance)}")

            if model == "logreg":
                coefs = lr_coefficients(final)
                coef_rows.append(_tag(coefs, **info))
                evaluate.save(plot_lr_coefficients(feature_set, coefs, imbalance, len(y)),
                              f"lr_coefficients_{feature_set}.png")
            if model in TREE_MODELS:
                tree_tables[(model, imbalance)] = builtin_importance(final)
                builtin_rows.append(_tag(tree_tables[(model, imbalance)], **info, method="builtin"))
            perm_tables[(model, imbalance)] = cv_permutation_importance(final, X, y, feature_set, args.repeats)
            perm_rows.append(_tag(perm_tables[(model, imbalance)], **info, method="permutation_roc_auc"))

        if tree_tables:
            evaluate.save(plot_tree_importances(feature_set, tree_tables, len(y)),
                          f"tree_importances_{feature_set}.png")
        if perm_tables:
            evaluate.save(plot_permutation(feature_set, perm_tables, len(y)),
                          f"permutation_importance_{feature_set}.png")

    coefs = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame(
        columns=["feature_set", "model", "imbalance", "label", "coefficient"])
    builtin = pd.concat(builtin_rows, ignore_index=True) if builtin_rows else pd.DataFrame(
        columns=["feature_set", "model", "imbalance", "method", "feature", "importance"])
    perm = pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame(
        columns=["feature_set", "model", "imbalance", "method", "feature", "importance", "importance_std"])

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in [("lr_coefficients.csv", coefs), ("importances.csv", pd.concat([builtin, perm]))]:
        table.to_csv(C.RESULTS_DIR / name, index=False)
        print(f"  table   {evaluate._show(C.RESULTS_DIR / name)}")

    text = summary_markdown(pd.DataFrame(kept), coefs, builtin, perm, cv, len(y), args.repeats)
    path = summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  summary {evaluate._show(path)}\n")
    start = text.index("## How much")
    print(text[start:text.index("## Caveats")].rstrip())
    print(f"\n{C.SCREENING_NOTE}.")


if __name__ == "__main__":
    # Run main() from the importable module, as src/train.py does.
    from src.interpret import main as _main

    _main()
