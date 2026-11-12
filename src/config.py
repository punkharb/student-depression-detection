"""Shared constants: paths, seed, column names, feature sets, target.

Every other script imports column names from here. Do not hard-code or rename
columns anywhere else.

Reminder: the model is a screening aid, not a diagnosis.
"""
from pathlib import Path

from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Student_Depression_Dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CSV = PROCESSED_DIR / "clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
MODELS_DIR = ROOT / "models"

# The raw CSV header starts with a UTF-8 BOM.
RAW_ENCODING = "utf-8-sig"

# ---------------------------------------------------------------- randomness / split
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------- cross-validation
# Shared by src/baseline.py and src/train.py so every model is scored the same
# way. train.py selects hyperparameters by ROC-AUC, tunes the threshold for F1,
# and reports every CV_SCORING metric.
# Recall is the priority metric; accuracy is only ever reported next to the others.
CV_METRICS = ["recall", "precision", "f1", "roc_auc", "accuracy"]
CV_SCORING = {
    "recall": "recall",
    # zero_division=0: a model that never predicts Yes (e.g. the majority-class
    # baseline) scores 0 instead of raising UndefinedMetricWarning.
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def make_cv() -> StratifiedKFold:
    """Stratified 5-fold CV splitter with the fixed seed (training set only)."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- target
TARGET = "Depression"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
TARGET_MAP = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

# ---------------------------------------------------------------- columns
# Names exactly as in the CSV header (do not rename).
GENDER_COL = "Gender"
AGE_COL = "Age"
FAMILY_COL = "Family_background"
RELATIONSHIP_COL = "Relationship_status"
MARITAL_COL = "Marital_status"
LIVING_COL = "Living_style"
SEMESTER_COL = "Current_semester"
CGPA_COL = "CGPA"
FRIEND_COL = "friend_circle"
HEALTH_COL = "Health_condition"
COST_COL = "Cost_without_study"
HOME_COL = "Home_division"

ALL_FEATURES = [
    GENDER_COL, AGE_COL, FAMILY_COL, RELATIONSHIP_COL, MARITAL_COL, LIVING_COL,
    SEMESTER_COL, CGPA_COL, FRIEND_COL, HEALTH_COL, COST_COL, HOME_COL,
]
RAW_COLUMNS = ALL_FEATURES + [TARGET]

NUMERIC_COLS = [AGE_COL, SEMESTER_COL, CGPA_COL, COST_COL]

# Ordinal features: categories listed lowest -> highest, encoded 0, 1, 2, ...
# Family background: lowest to highest socio-economic level.
ORDINAL_ORDERS = {
    FAMILY_COL: ["Lower Middle class", "Middle class", "Rich class"],
}
ORDINAL_COLS = list(ORDINAL_ORDERS)

# One-hot features.
# NOTE: Health_condition is one-hot, not ordinal (team decision, 2026-11-08).
# Its labels are Critical / Abnormal / Normal / Good, and "Normal" vs "Good"
# has no clear worst -> best order, so CLAUDE.md's fallback rule applies.
ONEHOT_COLS = [
    GENDER_COL, RELATIONSHIP_COL, MARITAL_COL, LIVING_COL,
    FRIEND_COL, HEALTH_COL, HOME_COL,
]

CATEGORICAL_COLS = ORDINAL_COLS + ONEHOT_COLS

# ---------------------------------------------------------------- feature sets
# Health condition nearly decides the label, so every experiment is run on both.
FEATURE_SETS = {
    "full": list(ALL_FEATURES),
    "no_health": [c for c in ALL_FEATURES if c != HEALTH_COL],
}

# ---------------------------------------------------------------- odd values
# Values treated as "odd" (not plausible). See CLAUDE.md > Known data issues.
ODD_VALUE_RULES = {SEMESTER_COL: 0, COST_COL: 0}
# Flag column names, used only if the odd rows exceed the threshold.
ODD_VALUE_FLAGS = {SEMESTER_COL: "semester_was_zero", COST_COL: "cost_was_zero"}
# Drop the odd rows if they are this share of the data or less; otherwise flag + impute.
ODD_VALUE_THRESHOLD = 0.02

SCREENING_NOTE = "Screening aid, not a diagnosis"
