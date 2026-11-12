"""Checks for src/evaluate.py. Run:  python -m unittest discover tests"""
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from src import baseline, evaluate, train
from src import config as C


def _row(feature_set, model, imbalance, recall, precision, f1):
    return {"feature_set": feature_set, "model": model, "imbalance": imbalance,
            "recall": recall, "precision": precision, "f1": f1}


class ScoreTest(unittest.TestCase):
    def test_all_no_predictions_score_like_the_baseline(self):
        y = np.array([0, 0, 0, 1])
        s = evaluate.score(y, np.zeros(4, dtype=int), np.zeros(4))
        self.assertEqual((s["recall"], s["precision"], s["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(s["accuracy"], 0.75)  # high accuracy, zero recall
        self.assertEqual(s["roc_auc"], 0.5)
        self.assertEqual((s["tn"], s["fp"], s["fn"], s["tp"]), (3, 0, 1, 0))

    def test_confusion_counts_match_metrics(self):
        y = np.array([0, 0, 1, 1, 1])
        s = evaluate.score(y, np.array([0, 1, 1, 1, 0]), np.array([.1, .6, .9, .8, .4]))
        self.assertEqual((s["tn"], s["fp"], s["fn"], s["tp"]), (1, 1, 1, 2))
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["precision"], 2 / 3)


class AddFlagsTest(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame([
            _row("full", evaluate.BASELINE, evaluate.NO_IMBALANCE, 0.0, 0.0, 0.0),
            _row("full", "logreg", "smote", 0.80, 0.60, 0.69),
            _row("full", "xgboost", "class_weight", 0.95, 0.90, 0.92),  # best on test
            _row("no_health", evaluate.BASELINE, evaluate.NO_IMBALANCE, 0.0, 0.0, 0.0),
            _row("no_health", "random_forest", "smote", 0.90, 0.20, 0.33),  # precision < Yes rate
        ])
        self.cv = pd.DataFrame({
            "model": ["logreg", "xgboost", "random_forest"],
            "imbalance": ["smote", "class_weight", "smote"],
            "feature_set": ["full", "full", "no_health"],
            "recall_mean": [0.7, 0.9, 0.8],
            "f1_mean": [0.80, 0.70, 0.40],  # CV prefers logreg on full
        })

    def test_chosen_model_comes_from_cv_f1_not_test_scores(self):
        out = evaluate.add_flags(self.table, self.cv, yes_rate=0.27)
        chosen = out[out["chosen_by_cv"]].set_index("feature_set")
        self.assertEqual(chosen.loc["full", "model"], "logreg")  # not the test-best xgboost
        self.assertEqual(chosen.loc["no_health", "model"], "random_forest")
        self.assertIn("cv_f1", out.columns)
        self.assertIn("cv_recall", out.columns)

    def test_beats_baseline_and_low_precision_flags(self):
        out = evaluate.add_flags(self.table, self.cv, yes_rate=0.27)
        self.assertFalse(out.loc[out["model"] == evaluate.BASELINE, "beats_baseline"].any())
        self.assertFalse(out.loc[out["model"] == evaluate.BASELINE, "low_precision"].any())
        self.assertTrue(out.loc[out["model"] != evaluate.BASELINE, "beats_baseline"].all())
        low = out[out["low_precision"]]
        self.assertEqual(list(low["model"]), ["random_forest"])

    def test_without_cv_results_nothing_is_chosen(self):
        out = evaluate.add_flags(self.table, None, yes_rate=0.27)
        self.assertFalse(out["chosen_by_cv"].any())
        for line in evaluate.success_summary(out):
            self.assertIn("not checked", line)


class SuccessSummaryTest(unittest.TestCase):
    def test_reports_fail_when_chosen_model_does_not_beat_baseline(self):
        table = pd.DataFrame([
            {**_row("full", evaluate.BASELINE, evaluate.NO_IMBALANCE, 0.0, 0.0, 0.0),
             "chosen_by_cv": False, "beats_baseline": False},
            {**_row("full", "logreg", "smote", 0.0, 0.0, 0.0),
             "chosen_by_cv": True, "beats_baseline": False},
        ])
        (line,) = evaluate.success_summary(table)
        self.assertIn("FAIL", line)


class LoadModelsTest(unittest.TestCase):
    def test_missing_baseline_raises(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C, "MODELS_DIR", Path(tmp)):
            with self.assertRaises(FileNotFoundError):
                evaluate.load_models("full")

    def test_missing_tuned_models_are_listed_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C, "MODELS_DIR", Path(tmp)):
            joblib.dump("baseline", baseline.model_path("full"))
            joblib.dump("tuned", train.model_path("logreg", "smote", "full"))
            entries, missing = evaluate.load_models("full")
        self.assertEqual([(e["model"], e["imbalance"]) for e in entries],
                         [(evaluate.BASELINE, evaluate.NO_IMBALANCE), ("logreg", "smote")])
        self.assertEqual(len(missing), len(train.MODELS) * len(train.IMBALANCE_METHODS) - 1)
        self.assertNotIn("logreg_smote_full.joblib", missing)


class MainOnBaselinesTest(unittest.TestCase):
    def test_runs_end_to_end_with_only_the_baselines(self):
        from src.data import flag_columns, load_train

        X, y = load_train()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            tmp = Path(tmp)
            for attr, sub in [("MODELS_DIR", "models"), ("RESULTS_DIR", "results"),
                              ("FIGURES_DIR", "figures")]:
                (tmp / sub).mkdir()
                stack.enter_context(mock.patch.object(C, attr, tmp / sub))
            for fs in C.FEATURE_SETS:
                joblib.dump(baseline.build_baseline(fs, flag_columns(X)).fit(X, y),
                            baseline.model_path(fs))
            stack.enter_context(redirect_stdout(StringIO()))
            evaluate.main()

            table = pd.read_csv(tmp / "results" / "test_results.csv")
            figures = sorted(p.name for p in (tmp / "figures").iterdir())

        self.assertEqual(sorted(table["feature_set"]), sorted(C.FEATURE_SETS))
        self.assertTrue((table["recall"] == 0).all())  # baseline always says No
        for metric in evaluate.TEST_METRICS:
            self.assertIn(metric, table.columns)
        self.assertEqual(figures, sorted(
            f"{kind}_{fs}.png" for kind in ("confusion_matrices", "roc_curves") for fs in C.FEATURE_SETS))


if __name__ == "__main__":
    unittest.main()
