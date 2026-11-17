"""Streamlit demo: enter a student profile, see the predicted risk.

Run:  streamlit run app/streamlit_app.py
Needs: python -m src.data, python -m src.baseline, python -m src.train
       (writes models/demo_no_health.joblib, demo_full.joblib, baseline_*.joblib)

Loads saved pipelines only; nothing is trained here. The demo models are the
best CV-F1 configuration per feature set (never a test-set score). Their
decision threshold is tuned for F1 and already inside the pipeline, so
predict() applies it.

Reminder: the model is a screening aid, not a diagnosis.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when launched as `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import config as C
from src.data import load_clean

MODEL_CHOICES = {
    "no_health": "Without Health condition (default)",
    "full": "With Health condition",
}


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_pipeline(path: Path):
    return joblib.load(path)


@st.cache_data
def load_options() -> pd.DataFrame:
    """Cleaned data, used only to read valid category values and numeric ranges."""
    return load_clean()


def missing_files() -> list[Path]:
    needed = [C.MODELS_DIR / f"demo_{fs}.joblib" for fs in C.FEATURE_SETS]
    needed += [C.MODELS_DIR / f"baseline_{fs}.joblib" for fs in C.FEATURE_SETS]
    needed.append(C.CLEAN_CSV)
    return [p for p in needed if not p.exists()]


# ---------------------------------------------------------------- form
def categories(df: pd.DataFrame, col: str) -> list[str]:
    if col in C.ORDINAL_ORDERS:  # keep the documented low -> high order
        return list(C.ORDINAL_ORDERS[col])
    return sorted(df[col].unique())


def profile_form(df: pd.DataFrame, use_health: bool) -> pd.DataFrame:
    """Widgets for the 12 features; returns a one-row DataFrame with raw values."""
    def select(label, col, key=None):
        return st.selectbox(label, categories(df, col), key=key or col)

    def number(label, col):
        return st.number_input(
            label, int(df[col].min()), int(df[col].max()), int(df[col].median()), key=col
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Personal")
        gender = select("Gender", C.GENDER_COL)
        age = number("Age", C.AGE_COL)
        marital = select("Marital status", C.MARITAL_COL)
        relationship = select("Relationship status", C.RELATIONSHIP_COL)
        st.subheader("Academic")
        semester = number("Current semester", C.SEMESTER_COL)
        cgpa = st.slider(
            "CGPA", float(df[C.CGPA_COL].min()), float(df[C.CGPA_COL].max()),
            round(float(df[C.CGPA_COL].median()), 2), 0.01, key=C.CGPA_COL,
        )
    with right:
        st.subheader("Life and support")
        family = select("Family background", C.FAMILY_COL)
        living = select("Living style", C.LIVING_COL)
        friend = select("Has a friend circle", C.FRIEND_COL)
        cost = st.number_input(
            "Monthly cost (without study)", int(df[C.COST_COL].min()),
            int(df[C.COST_COL].max()), int(df[C.COST_COL].median()),
            step=500, key=C.COST_COL,
        )
        home = select("Home division", C.HOME_COL)
        if use_health:
            health = select("Health condition", C.HEALTH_COL)
        else:
            health = None
            st.caption("Health condition is not used by this model.")

    row = {
        C.GENDER_COL: gender, C.AGE_COL: age, C.FAMILY_COL: family,
        C.RELATIONSHIP_COL: relationship, C.MARITAL_COL: marital,
        C.LIVING_COL: living, C.SEMESTER_COL: semester, C.CGPA_COL: cgpa,
        C.FRIEND_COL: friend, C.COST_COL: cost, C.HOME_COL: home,
        # Not used by the no_health pipeline; any valid value is ignored there.
        C.HEALTH_COL: health if health is not None else categories(df, C.HEALTH_COL)[0],
    }
    return pd.DataFrame([row])[C.ALL_FEATURES]


# ---------------------------------------------------------------- output
def label(pred: int) -> str:
    return C.POSITIVE_LABEL if pred == 1 else C.NEGATIVE_LABEL


def show_prediction(feature_set: str, X: pd.DataFrame) -> None:
    model = load_pipeline(C.MODELS_DIR / f"demo_{feature_set}.joblib")
    baseline = load_pipeline(C.MODELS_DIR / f"baseline_{feature_set}.joblib")

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    base_pred = int(baseline.predict(X)[0])

    col_model, col_base = st.columns(2)
    with col_model:
        st.markdown("### Model prediction")
        st.metric("At risk of depression?", label(pred))
        st.metric("Estimated probability of Yes", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
    with col_base:
        st.markdown("### Baseline (majority class)")
        st.metric("At risk of depression?", label(base_pred))
        st.caption(
            "The baseline says No for every student: about 73% accuracy, "
            "but it catches no one at risk."
        )


# ---------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="Student Depression Risk Screening", page_icon="🎓")
    st.title("Student Depression Risk Screening")
    st.warning(
        f"**{C.SCREENING_NOTE}.** The result is a risk estimate from a self-reported "
        "survey, not a clinical assessment. If you are worried about someone's "
        "wellbeing, please talk to a counsellor."
    )

    missing = missing_files()
    if missing:
        st.error(
            "Missing files: " + ", ".join(str(p.relative_to(C.ROOT)) for p in missing)
            + ". Run `python -m src.data`, `python -m src.baseline` and "
            "`python -m src.train` first."
        )
        st.stop()

    feature_set = st.radio(
        "Model",
        list(MODEL_CHOICES),
        format_func=MODEL_CHOICES.get,
        horizontal=True,
        help="Health condition nearly decides the label on its own. "
             "Switch to see how much it changes the prediction.",
    )
    if feature_set == "full":
        st.info(
            "This model uses Health condition, which is the dominant factor. "
            "Without it the other factors alone give much weaker predictions."
        )

    X = profile_form(load_options(), use_health=(feature_set == "full"))
    st.divider()
    show_prediction(feature_set, X)

    st.caption("Anonymised survey data (Kaggle student depression dataset). CSC340, KMUTT.")


main()
