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
| Sittipon P. | Demo and presentation |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset

Download from [Kaggle: Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset) and put the CSV in `data/raw/`. Do not edit it and do not commit it (`data/` is git-ignored).

## Run

```bash
python -m src.eda
python -m src.baseline
python -m src.train
python -m src.evaluate
python -m src.interpret
streamlit run app/streamlit_app.py
```

## Layout

```
data/raw/            original Kaggle CSV (not committed)
data/processed/      cleaned data
src/                 config, data, features, eda, baseline, train, evaluate, interpret
models/              saved pipelines (not committed)
reports/figures/     plots for slides
reports/results/     metric tables
app/                 Streamlit demo
```

## Timeline

| Dates | Work |
|---|---|
| 4–12 Nov | Data cleaning, EDA, baseline |
| 13–19 Nov | Train LR / RF / XGBoost, tune |
| 20–22 Nov | Evaluate, interpret, demo, slides |
| **23 Nov 23:59** | **Submit slides and demo** |
| 24 Nov | Presentation |

## Workflow

- One issue → one branch → one pull request into `main`. Do not push to `main` directly.
- Reference the issue in the PR (`Closes #N`).
- New decisions go in the CLAUDE.md decision log.
