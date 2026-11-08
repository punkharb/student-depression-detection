# Student Depression Detection Using Machine Learning

CSC340 Artificial Intelligence · School of Information Technology, KMUTT

Binary classification: predict whether a student is at risk of depression (`Yes` / `No`) from 12 survey features, compare Logistic Regression, Random Forest and XGBoost against a majority-class baseline, and explain which factors matter most.

> **Screening aid, not a diagnosis.** The label is self-reported and single-item; results describe risk only.

Full project rules (methodology, leakage rules, experiment grid, decision log) live in [CLAUDE.md](CLAUDE.md). Read it before you start.

## Team

| Member | Area |
|---|---|
| Kawin K. | Data preprocessing and EDA |
| Punnatorn B. | Model training and tuning |
| Chavaphon I. | Evaluation and interpretation |
| Sittipon P. | Demo |

## Setup

Requires Python 3.11 or newer (the pinned pandas 3.x does not support 3.10).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

The raw CSV from [Kaggle: Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset) is included at `data/raw/Student_Depression_Dataset.csv`, so no download is needed. Do not edit it. The cleaned data and the fixed train/test split are also committed in `data/processed/` (`clean.csv`, `train.csv`, `test.csv`). If `src/data.py` or `src/config.py` changes, re-run `python -m src.data` and commit the updated files.

## Run

Run `python -m src.data` first: it cleans the raw CSV and writes the fixed train/test split to `data/processed/`.

```bash
python -m src.data
python -m src.eda
python -m src.baseline
python -m src.train
python -m src.evaluate
python -m src.interpret
streamlit run app/streamlit_app.py
```

## Layout

```
data/raw/            original Kaggle CSV (committed, never edited)
data/processed/      cleaned data and fixed train/test split (committed)
src/                 config, data, features, eda, baseline, train, evaluate, interpret
models/              saved pipelines (not committed)
reports/figures/     generated plots
reports/results/     metric tables
app/                 Streamlit demo
```

## Timeline

| Dates | Work |
|---|---|
| 4–12 Nov | Data cleaning, EDA, baseline |
| 13–19 Nov | Train LR / RF / XGBoost, tune |
| 20–22 Nov | Evaluate, interpret, demo |
| **23 Nov 23:59** | **Deadline** |

## Workflow

- One issue → one branch → one pull request into `main`. Do not push to `main` directly.
- Reference the issue in the PR (`Closes #N`).
- New decisions go in the CLAUDE.md decision log.
