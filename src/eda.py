"""Exploratory data analysis: plots and summary tables for the slides.

Uses the full cleaned dataset (team decision: EDA describes the data and does
not select any model settings). Run `python -m src.data` first.

Run:  python -m src.eda
Figures -> reports/figures/eda_*.png, tables -> reports/results/eda_*.csv.
Only aggregates are written; no individual rows are exported.

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config as C
from src.data import load_clean, load_raw, odd_value_report

# ---------------------------------------------------------------- style
COLOR_NO = "#2a78d6"   # categorical slot 1 (blue)
COLOR_YES = "#eb6834"  # categorical slot 2 (orange)
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
# Diverging: blue pole, neutral gray midpoint, orange pole.
DIVERGING = LinearSegmentedColormap.from_list("div", [COLOR_NO, "#f0efec", COLOR_YES])

# Display order only (for readable charts); the model one-hot encodes this column.
HEALTH_DISPLAY_ORDER = ["Critical", "Abnormal", "Normal", "Good"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "legend.frameon": False,
    "font.size": 10,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, name: str) -> None:
    # Place the note just below everything already drawn (axis labels, rotated ticks).
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())  # in inches
    fig_w, fig_h = fig.get_size_inches()
    fig.text(content.x0 / fig_w, (content.y0 - 0.12) / fig_h,
             f"{C.SCREENING_NOTE}. Self-reported, single-item label.",
             fontsize=8, color=TEXT_MUTED, ha="left", va="top")
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure  {path.relative_to(C.ROOT)}")


def _save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    path = C.RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  table   {path.relative_to(C.ROOT)}")


def _with_labels(df: pd.DataFrame) -> pd.Series:
    """Target as 'Yes'/'No' text for plotting."""
    return df[C.TARGET].map({v: k for k, v in C.TARGET_MAP.items()})


# ---------------------------------------------------------------- tables
def odd_values_table() -> None:
    """Odd-value counts on the raw data (before cleaning) for the slides."""
    report = odd_value_report(load_raw())
    _save_table(report, "eda_odd_values")
    print(report.to_string(index=False, formatters={"share_of_data": "{:.2%}".format}))


def summary_tables(df: pd.DataFrame) -> None:
    summary = df[C.NUMERIC_COLS].describe().T.round(2)
    _save_table(summary, "eda_summary_numeric", index=True)

    by_target = df.groupby(_with_labels(df))[C.NUMERIC_COLS].agg(["mean", "median"]).round(2)
    by_target.columns = [f"{col}_{stat}" for col, stat in by_target.columns]
    _save_table(by_target, "eda_numeric_by_target", index=True)

    counts = _with_labels(df).value_counts().reindex([C.NEGATIVE_LABEL, C.POSITIVE_LABEL])
    balance = pd.DataFrame({"count": counts, "share": (counts / counts.sum()).round(4)})
    balance.index.name = C.TARGET
    _save_table(balance, "eda_class_balance", index=True)


# ---------------------------------------------------------------- figures
def plot_class_balance(df: pd.DataFrame) -> None:
    counts = _with_labels(df).value_counts().reindex([C.NEGATIVE_LABEL, C.POSITIVE_LABEL])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(counts.index, counts.values, color=[COLOR_NO, COLOR_YES], width=0.55)
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,} ({n / counts.sum():.0%})", (bar.get_x() + bar.get_width() / 2, n),
                    xytext=(0, 4), textcoords="offset points", ha="center", color=TEXT)
    ax.set_title("Class balance: 27% of students at risk")
    ax.set_xlabel("Depression (self-reported)")
    ax.set_ylabel("Students")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "eda_class_balance")


def plot_numeric_by_target(df: pd.DataFrame) -> None:
    labels = _with_labels(df)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, col in zip(axes.flat, C.NUMERIC_COLS):
        bins = np.histogram_bin_edges(df[col], bins=20 if df[col].nunique() > 20 else df[col].nunique())
        for label, color in [(C.NEGATIVE_LABEL, COLOR_NO), (C.POSITIVE_LABEL, COLOR_YES)]:
            ax.hist(df.loc[labels == label, col], bins=bins, density=True, histtype="step",
                    linewidth=2, color=color, label=label)
        ax.set_title(col.replace("_", " "), fontsize=11)
        ax.set_ylabel("Density")
    axes.flat[0].legend(title="Depression")
    fig.suptitle("Numeric features by depression label", fontweight="bold", color=TEXT)
    fig.tight_layout()
    _save(fig, "eda_numeric_by_target")


def yes_rate_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """% Yes and n per category, sorted for display."""
    rate = df.groupby(col)[C.TARGET].agg(yes_rate="mean", n="size")
    if col == C.HEALTH_COL:
        return rate.reindex(HEALTH_DISPLAY_ORDER)
    if col in C.ORDINAL_ORDERS:
        return rate.reindex(C.ORDINAL_ORDERS[col])
    return rate.sort_values("yes_rate", ascending=False)


def plot_categorical_rates(df: pd.DataFrame) -> pd.DataFrame:
    overall = df[C.TARGET].mean()
    tables = []
    for col in C.CATEGORICAL_COLS:
        rate = yes_rate_by_category(df, col)
        tables.append(rate.assign(feature=col).rename_axis("category").reset_index())

        fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rate) + 1.4))
        y = np.arange(len(rate))[::-1]
        ax.barh(y, rate["yes_rate"], color=COLOR_NO, height=0.6)
        ax.axvline(overall, color=TEXT_MUTED, linestyle="--", linewidth=1)
        ax.annotate(f"overall {overall:.0%}", (overall, y.max() + 0.5), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_MUTED, va="center")
        for yi, (r, n) in zip(y, rate[["yes_rate", "n"]].itertuples(index=False)):
            ax.annotate(f"{r:.0%}  (n={n:,})", (r, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yticks(y, rate.index)
        ax.set_xlim(0, max(1.0, rate["yes_rate"].max() * 1.25))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Share of students labelled 'Yes'")
        ax.set_title(f"At-risk rate by {col.replace('_', ' ').lower()}")
        _save(fig, f"eda_categorical_rate_{col.lower()}")

    table = pd.concat(tables)[["feature", "category", "n", "yes_rate"]]
    table["yes_rate"] = table["yes_rate"].round(4)
    _save_table(table, "eda_categorical_rates")
    return table


def health_rule(df: pd.DataFrame) -> pd.DataFrame:
    """How well Health_condition alone predicts the label.

    Rule: for each health category, predict that category's majority label.
    Explains why every experiment is also run on the 'no_health' feature set.
    """
    majority = df.groupby(C.HEALTH_COL)[C.TARGET].mean().ge(0.5).astype(int)
    pred = df[C.HEALTH_COL].map(majority)
    y = df[C.TARGET]
    tp = int(((pred == 1) & (y == 1)).sum())
    result = pd.DataFrame([{
        "rule": "Yes if Health_condition in {" + ", ".join(sorted(majority[majority == 1].index)) + "}",
        "accuracy": (pred == y).mean(),
        "recall": tp / (y == 1).sum(),
        "precision": tp / (pred == 1).sum(),
    }]).round(4)
    _save_table(result, "eda_health_rule")
    return result


def plot_health_vs_depression(df: pd.DataFrame, rule: pd.DataFrame) -> None:
    ct = pd.crosstab(df[C.HEALTH_COL], _with_labels(df)).reindex(HEALTH_DISPLAY_ORDER)
    ct = ct[[C.NEGATIVE_LABEL, C.POSITIVE_LABEL]]
    _save_table(ct, "eda_health_crosstab", index=True)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(ct))[::-1]
    gap = 4  # small surface gap between stacked segments
    ax.barh(y, ct[C.NEGATIVE_LABEL], color=COLOR_NO, height=0.6, label=C.NEGATIVE_LABEL)
    ax.barh(y, ct[C.POSITIVE_LABEL] - np.where(ct[C.POSITIVE_LABEL] > 0, gap, 0),
            left=ct[C.NEGATIVE_LABEL] + np.where(ct[C.NEGATIVE_LABEL] > 0, gap, 0),
            color=COLOR_YES, height=0.6, label=C.POSITIVE_LABEL)
    for yi, (no, yes) in zip(y, ct.itertuples(index=False)):
        ax.annotate(f"{yes / (no + yes):.0%} Yes", (no + yes, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, ct.index)
    ax.set_xlim(0, ct.sum(axis=1).max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Students")
    ax.legend(title="Depression", loc="lower right")
    acc = rule["accuracy"].iloc[0]
    ax.set_title(f"Health condition nearly decides the label (rule alone: {acc:.0%} accuracy)")
    _save(fig, "eda_health_vs_depression")


def plot_correlation(df: pd.DataFrame) -> None:
    data = df[C.NUMERIC_COLS].copy()
    for col, order in C.ORDINAL_ORDERS.items():
        data[f"{col} (ordinal)"] = df[col].map({v: i for i, v in enumerate(order)})
    data["Depression (Yes=1)"] = df[C.TARGET]
    corr = data.corr(method="spearman")
    _save_table(corr.round(3), "eda_correlation", index=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color=TEXT)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title("Correlation: numeric features and label")
    _save(fig, "eda_correlation")


# ---------------------------------------------------------------- main
def main() -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Odd values (raw data):")
    odd_values_table()

    df = load_clean()
    print(f"\nEDA on cleaned data: {len(df)} rows")
    summary_tables(df)
    plot_class_balance(df)
    plot_numeric_by_target(df)
    plot_categorical_rates(df)
    rule = health_rule(df)
    plot_health_vs_depression(df, rule)
    plot_correlation(df)

    print("\nHealth-condition-only rule:")
    print(rule.to_string(index=False))


if __name__ == "__main__":
    main()
