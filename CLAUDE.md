# CLAUDE.md

Guidance for Claude when working in this repository.

## Project

**Student Depression Detection Using Machine Learning**
CSC340 Artificial Intelligence, School of Information Technology, KMUTT (Lecturer: Punnarust Silparattanawong).

Binary classification: predict whether a student is at risk of depression (`Yes` / `No`) from survey features, compare three ML models against a majority-class baseline, and explain which factors matter most.

Research question: *Can machine learning identify students at risk of depression better than a majority-class baseline, and which factors matter most?*

The model is a **screening aid, not a diagnosis**. Keep this framing in all code comments, outputs, plots and the demo.

### Team and ownership

| Member | ID | Area |
|---|---|---|
| Kawin K. | 67130500836 | Data preprocessing and EDA |
| Punnatorn B. | 67130500842 | Model training and tuning |
| Chavaphon I. | 67130500805 | Evaluation and interpretation |
| Sittipon P. | 67130500867 | Demo and presentation |

### Deadlines

- 4–12 Nov: data cleaning, EDA, baseline
- 13–19 Nov: train Logistic Regression, Random Forest, XGBoost; tune and re-run
- 20–22 Nov: evaluate, interpret, build demo and slides
- **23 Nov 23:59: submit slides and demo**
- 24 Nov: project presentation (exam)

## Dataset

- Source: https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset/data
- 4,464 students, 12 input features, target `Depression` (Yes 1,201 / No 3,263, about 27% / 73%)
- Feature groups:
  - Personal: gender, age, marital status, relationship status
  - Academic: current semester, CGPA
  - Life and support: family background, living style, friend circle, health condition, monthly cost, home division
- Exact column names must be read from the CSV header and defined once in `src/config.py`. Do not guess or rename columns elsewhere.
- Never modify the raw CSV. Write cleaned data to `data/processed/`.

### Known data issues

- **Odd values** (semester 0, zero monthly cost): apply this rule in `src/data.py` and report the row counts in the EDA output and the slides.
  1. Count the rows with semester 0 and the rows with monthly cost 0, separately and combined, and check their Yes/No split.
  2. If the combined rows are **2% of the dataset or less**, drop them.
  3. If they are **more than 2%**, keep the rows: set those values to missing, add a binary flag column for each (`semester_was_zero`, `cost_was_zero`), and impute with the median inside the pipeline so it is fitted on training folds only.
  4. Never drop or change rows based on the target value.
- **Health condition nearly decides the label** (a rule on it alone reaches about 89% accuracy in EDA). Every experiment runs on two feature sets: `full` and `no_health`.
- **Self-reported, single-item label**: results describe risk, not diagnosis.
- **Sensitive data**: use anonymised data only. Never print or export individual rows beyond what is needed for debugging.

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (for SMOTE), XGBoost, Matplotlib / Seaborn, Streamlit.

Environment: **pip + venv**, dependencies pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

When adding a dependency, add it to `requirements.txt` with a pinned version in the same change.

## Repository layout

Code lives in Python scripts under `src/`, not notebooks.

```
data/
  raw/                 # original Kaggle CSV, never edited
  processed/           # cleaned data
src/
  config.py            # paths, random seed, column names, feature sets, target
  data.py              # loading and cleaning (odd-value rules live here)
  features.py          # preprocessing (encoding, scaling) as a ColumnTransformer
  eda.py               # EDA plots and summary tables
  baseline.py          # majority-class baseline
  train.py             # model pipelines and CV tuning
  evaluate.py          # test-set metrics, confusion matrices, ROC curves
  interpret.py         # coefficients and feature importance
models/                # saved fitted pipelines
reports/
  figures/             # all generated plots
  results/             # metric tables (CSV)
app/
  streamlit_app.py     # demo
requirements.txt
```

Intended entry points (keep these working once created):

```bash
python -m src.eda
python -m src.baseline
python -m src.train
python -m src.evaluate
python -m src.interpret
streamlit run app/streamlit_app.py
```

## Methodology rules

Follow this pipeline. Do not change a step without asking the team.

1. **Clean**: apply the documented odd-value rules.
2. **Encode**: one-hot for text categories; **ordinal for Health condition and Family background**. The category order for each ordinal feature is written explicitly in `src/config.py`, following these rules:
   - Health condition: worst to best (e.g. poor → average → good), encoded 0, 1, 2, …
   - Family background: lowest to highest socio-economic level (e.g. lower → middle → upper), encoded 0, 1, 2, …
   - Map the actual labels in the CSV onto these orders. If a feature's labels have no clear order (or include values like "other"), use one-hot for that feature instead and note the change in `src/config.py`.
3. **Scale**: standardise numeric features for Logistic Regression only. Tree models use unscaled features.
4. **Split**: stratified 80% train / 20% test, with the single fixed seed from `src/config.py`. The same test set is used for every model and feature set.
5. **Tune**: stratified 5-fold CV with grid or randomized search, **optimising recall**.

The random seed is **42** (`RANDOM_STATE = 42` in `src/config.py`).

### Leakage rules (must hold everywhere)

- All preprocessing (encoding, scaling, SMOTE) goes inside a pipeline so it is fitted on training folds only.
- SMOTE must use `imblearn.pipeline.Pipeline`, never be applied to the full dataset or the test set.
- The test set is touched only in `src/evaluate.py`, after tuning is finished. Never use it for model selection.
- Use the seed from `src/config.py` for every source of randomness (split, CV, models, SMOTE).

### Imbalance handling

Try both methods and compare them:

- `class_weight="balanced"` (or `scale_pos_weight` for XGBoost)
- SMOTE on training folds only

### Recall as the tuning objective

Recall is the priority metric, but a model that predicts `Yes` for everyone gets 100% recall. To guard against this:

- Use multi-metric scoring in CV (recall, precision, F1, ROC-AUC) with `refit="recall"`.
- Always log precision and F1 next to recall in CV results.
- If the chosen configuration has very low precision, flag it to the team instead of silently accepting it.

## Experiment grid

Every combination below is trained and evaluated on the same test set:

- Models: Logistic Regression, Random Forest, XGBoost
- Imbalance: class weights, SMOTE
- Feature sets: `full`, `no_health`

Plus the majority-class baseline (predicts `No` for everyone: about 73% accuracy, 0% recall) for each feature set.

## Evaluation

- Metrics: **recall (priority)**, precision, F1, ROC-AUC. Report accuracy only alongside these, never on its own.
- Outputs: one results table in `reports/results/`, confusion matrices, ROC curves, feature importance (LR coefficients, RF / XGBoost importances), each for `full` and `no_health`.
- Success criterion: the best model beats the baseline on F1 and recall.

## Demo (Streamlit)

- Input: a student profile through form widgets matching the 12 features.
- Output: predicted risk (Yes / No) and probability, shown next to the baseline prediction.
- Load a saved pipeline from `models/`; do not retrain inside the app.
- Show a clear note that this is a screening aid, not a diagnosis.
- Model choice: for each feature set, use the configuration with the best **CV recall** (from `src/train.py`), never the best test-set score.
- The app has a toggle between the `no_health` model (default) and the `full` model, so the presentation can show how much Health condition changes the prediction.
- Save both chosen pipelines to `models/` as `demo_no_health.joblib` and `demo_full.joblib`.

## Working conventions

- When something in this file is unclear or a decision is not recorded here, ask instead of assuming.
- Keep functions small and reusable across scripts; shared constants go in `src/config.py`.
- Save every figure to `reports/figures/` with a descriptive filename, so it can go straight into the slides.
- Do not commit fitted models or large generated files unless the team asks to.

## Decision log

- Random seed: 42
- Odd values: drop if 2% or less of rows, otherwise flag, set to missing and impute (see Known data issues)
- Ordinal order: Health condition worst → best, Family background lowest → highest; fall back to one-hot if labels have no clear order
- Demo: best CV-recall model per feature set, `no_health` by default with a toggle to `full`
- Health_condition is one-hot, not ordinal: its labels (Critical / Abnormal / Normal / Good) have no clear order between Normal and Good
- Odd values: 42 rows with semester 0 + 2 with cost 0 = 44 rows (0.99%, 10 Yes / 34 No), under 2%, so they are dropped (4,420 rows remain)
- EDA runs on the full cleaned dataset: it only describes the data and selects no model settings
- The fixed split is saved to `data/processed/train.csv` / `test.csv` (via `python -m src.data`) so every member uses the same rows
- Python 3.11+ is required (pinned pandas 3.x); tested on 3.14
- The raw CSV is committed in `data/raw/` (Kawin's request) so every member has the same file without a Kaggle download; `data/processed/` stays git-ignored

Add new decisions here with a one-line reason.
