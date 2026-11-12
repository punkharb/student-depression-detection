"""Checks for src/interpret.py. Run:  python -m unittest discover tests"""
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV

from src import config as C
from src import interpret, train
from src.data import load_train

N_ROWS = 400  # small slice of the training set keeps the tests fast


def _fit_small(model: str, imbalance: str, feature_set: str, X, y):
    """A threshold-tuned pipeline like src/train.py saves, but tiny and untuned."""
    pipe = train.build_pipeline(model, imbalance, feature_set)
    if model != "logreg":
        pipe.set_params(model__n_estimators=20)
    return TunedThresholdClassifierCV(pipe, scoring=C.CV_SCORING["f1"], cv=3).fit(X, y)


def _cv_row(feature_set, model, imbalance, f1):
    return {"feature_set": feature_set, "model": model, "imbalance": imbalance,
            "f1_mean": f1, "recall_mean": 0.5, "precision_mean": 0.5, "roc_auc_mean": 0.7}


class Data:
    X, y = None, None

    @classmethod
    def get(cls):
        if cls.X is None:
            X, y = load_train()
            cls.X, cls.y = X.iloc[:N_ROWS].reset_index(drop=True), y.iloc[:N_ROWS].reset_index(drop=True)
        return cls.X, cls.y


class PickPerModelTest(unittest.TestCase):
    def test_best_cv_f1_variant_per_feature_set_and_model(self):
        cv = pd.DataFrame([
            _cv_row("full", "logreg", "class_weight", 0.90),
            _cv_row("full", "logreg", "smote", 0.95),
            _cv_row("no_health", "logreg", "class_weight", 0.55),
            _cv_row("no_health", "logreg", "smote", 0.50),
            _cv_row("full", "xgboost", "class_weight", 0.97),
            _cv_row("full", "xgboost", "smote", 0.96),
        ])
        picks = interpret.pick_per_model(cv)
        got = {(r.feature_set, r.model): r.imbalance for r in picks.itertuples()}
        self.assertEqual(got, {("full", "logreg"): "smote", ("no_health", "logreg"): "class_weight",
                               ("full", "xgboost"): "class_weight"})
        self.assertEqual(list(picks["feature_set"]), ["full", "full", "no_health"])


class ReadableNamesTest(unittest.TestCase):
    def test_one_row_per_output_column_with_original_feature(self):
        X, y = Data.get()
        for fs in C.FEATURE_SETS:
            with self.subTest(feature_set=fs):
                prep, _ = interpret.unwrap(_fit_small("logreg", "class_weight", fs, X, y))
                names = interpret.readable_names(prep)
                self.assertEqual(list(names["column"]), list(prep.get_feature_names_out()))
                self.assertEqual(set(names["feature"]), set(C.FEATURE_SETS[fs]))
                self.assertEqual(C.HEALTH_COL in set(names["feature"]), fs == "full")
        gender = names.loc[names["feature"] == C.GENDER_COL, "label"]
        self.assertEqual(len(gender), 1)  # binary: one column, labelled against the dropped level
        self.assertRegex(gender.iloc[0], r"^Gender: \w+ \(vs \w+\)$")


class ImportanceTest(unittest.TestCase):
    def test_lr_coefficients_match_the_model(self):
        X, y = Data.get()
        final = _fit_small("logreg", "class_weight", "full", X, y)
        coefs = interpret.lr_coefficients(final)
        _, model = interpret.unwrap(final)
        self.assertEqual(len(coefs), model.coef_.shape[1])
        self.assertTrue(np.allclose(sorted(coefs["coefficient"]), sorted(model.coef_[0])))
        self.assertTrue(np.allclose(coefs["odds_ratio"], np.exp(coefs["coefficient"])))

    def test_builtin_importance_grouped_to_original_features(self):
        X, y = Data.get()
        for model in interpret.TREE_MODELS:
            for fs in C.FEATURE_SETS:
                with self.subTest(model=model, feature_set=fs):
                    imp = interpret.builtin_importance(_fit_small(model, "class_weight", fs, X, y))
                    self.assertEqual(set(imp["feature"]), set(C.FEATURE_SETS[fs]))
                    self.assertAlmostEqual(imp["importance"].sum(), 1.0)

    def test_permutation_uses_only_the_feature_set_and_is_repeatable(self):
        X, y = Data.get()
        final = _fit_small("logreg", "class_weight", "no_health", X, y)
        a = interpret.cv_permutation_importance(final, X, y, "no_health", n_repeats=2)
        b = interpret.cv_permutation_importance(final, X, y, "no_health", n_repeats=2)
        self.assertEqual(set(a["feature"]), set(C.FEATURE_SETS["no_health"]))
        self.assertNotIn(C.HEALTH_COL, set(a["feature"]))
        pd.testing.assert_frame_equal(a, b)  # seeded


class MainTest(unittest.TestCase):
    def test_runs_end_to_end_on_training_data_only(self):
        X, y = Data.get()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            tmp = Path(tmp)
            for attr, sub in [("MODELS_DIR", "models"), ("RESULTS_DIR", "results"),
                              ("FIGURES_DIR", "figures"), ("REPORTS_DIR", "reports")]:
                (tmp / sub).mkdir()
                stack.enter_context(mock.patch.object(C, attr, tmp / sub))
            rows = []
            for fs in C.FEATURE_SETS:
                for model in train.MODELS:
                    joblib.dump(_fit_small(model, "class_weight", fs, X, y),
                                train.model_path(model, "class_weight", fs))
                    rows.append(_cv_row(fs, model, "class_weight", 0.6 + 0.1 * (fs == "full")))
            pd.DataFrame(rows).to_csv(tmp / "results" / "cv_results.csv", index=False)
            stack.enter_context(mock.patch.object(interpret, "load_train", return_value=(X, y)))
            stack.enter_context(mock.patch("src.data.load_test", side_effect=AssertionError("test set loaded")))
            stack.enter_context(redirect_stdout(StringIO()))
            interpret.main(["--repeats", "1"])

            figures = sorted(p.name for p in (tmp / "figures").iterdir())
            imp = pd.read_csv(tmp / "results" / "importances.csv")
            coefs = pd.read_csv(tmp / "results" / "lr_coefficients.csv")
            summary = (tmp / "reports" / "interpretation_summary.md").read_text(encoding="utf-8")

        self.assertEqual(figures, sorted(
            f"{kind}_{fs}.png" for kind in ("lr_coefficients", "tree_importances", "permutation_importance")
            for fs in C.FEATURE_SETS))
        self.assertEqual(set(imp["method"]), {"builtin", "permutation_roc_auc"})
        self.assertFalse(((imp["feature_set"] == "no_health") & (imp["feature"] == C.HEALTH_COL)).any())
        self.assertEqual(set(coefs["feature_set"]), set(C.FEATURE_SETS))
        self.assertIn("How much Health condition dominates", summary)
        self.assertIn(C.SCREENING_NOTE, summary)


if __name__ == "__main__":
    unittest.main()
