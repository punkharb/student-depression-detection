"""Checks for src/train.py. Run:  python -m unittest discover tests"""
import unittest

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline

from src import config as C
from src import train


def _scales_numeric(pipe) -> bool:
    num = dict((name, t) for name, t, _ in pipe.named_steps["prep"].transformers)["num"]
    return "scale" in num.named_steps


class BuildPipelineTest(unittest.TestCase):
    def test_class_weight_variant_has_no_smote_and_balances_classes(self):
        for model in train.MODELS:
            pipe = train.build_pipeline(model, "class_weight", "full")
            self.assertIsInstance(pipe, ImbPipeline)
            self.assertNotIn("smote", pipe.named_steps)
            clf = pipe.named_steps["model"]
            if model != "xgboost":
                self.assertEqual(clf.class_weight, "balanced")
            self.assertEqual(clf.get_params()["random_state"], C.RANDOM_STATE)

    def test_smote_variant_resamples_inside_pipeline_with_seed(self):
        for model in train.MODELS:
            pipe = train.build_pipeline(model, "smote", "no_health")
            self.assertEqual(list(pipe.named_steps), ["select", "smote", "prep", "model"])
            self.assertIsInstance(pipe.named_steps["smote"], SMOTENC)
            self.assertEqual(pipe.named_steps["smote"].random_state, C.RANDOM_STATE)
            clf = pipe.named_steps["model"]
            # No double correction: SMOTE runs alone, without class weights.
            self.assertIn(clf.get_params().get("class_weight"), (None,))
            self.assertIn(clf.get_params().get("scale_pos_weight"), (None, 1, 1.0))

    def test_only_logistic_regression_is_scaled(self):
        self.assertTrue(_scales_numeric(train.build_pipeline("logreg", "smote", "full")))
        self.assertFalse(_scales_numeric(train.build_pipeline("random_forest", "smote", "full")))
        self.assertFalse(_scales_numeric(train.build_pipeline("xgboost", "smote", "full")))


class SmoteNcTest(unittest.TestCase):
    def test_synthetic_rows_keep_real_category_labels_and_drop_unused_columns(self):
        from src.data import load_train

        X, y = load_train()
        X, y = X.iloc[:600], y.iloc[:600]
        pipe = train.build_pipeline("random_forest", "smote", "no_health")
        Xs = pipe.named_steps["select"].fit_transform(X)
        self.assertNotIn(C.HEALTH_COL, Xs.columns)
        Xr, yr = pipe.named_steps["smote"].fit_resample(Xs, y)
        self.assertEqual(int((yr == 1).sum()), int((yr == 0).sum()))  # balanced
        for col in C.ONEHOT_COLS + C.ORDINAL_COLS:
            if col in Xs.columns:
                self.assertTrue(set(Xr[col]) <= set(X[col]), col)  # no made-up categories


class FlagAndImputeTest(unittest.TestCase):
    def test_smote_variant_handles_missing_values_and_flag_columns(self):
        # The >2% odd-value branch sets odd values to NaN and adds flag columns.
        from src.data import load_train

        X, y = load_train()
        X, y = X.iloc[:600].copy(), y.iloc[:600]
        odd = X.index[:30]
        X["semester_was_zero"] = 0
        X.loc[odd, "semester_was_zero"] = 1
        X[C.SEMESTER_COL] = X[C.SEMESTER_COL].astype(float)
        X.loc[odd, C.SEMESTER_COL] = float("nan")
        pipe = train.build_pipeline("logreg", "smote", "no_health", flag_cols=["semester_was_zero"])
        pipe.fit(X, y)
        self.assertEqual(pipe.predict(X.head(3)).shape, (3,))


class SavedModelsLoadTest(unittest.TestCase):
    def test_saved_models_load_and_predict_outside_the_training_script(self):
        # Models are dumped by `python -m src.train`; evaluate.py and the demo
        # must be able to load them (nothing may be pickled as __main__.*).
        import joblib
        from src.data import load_train

        paths = sorted(C.MODELS_DIR.glob("*.joblib"))
        if not paths:
            self.skipTest("no saved models; run python -m src.train")
        X, _ = load_train()
        for path in paths:
            with self.subTest(model=path.name):
                self.assertEqual(joblib.load(path).predict(X.head(3)).shape, (3,))


class XGBoostWeightTest(unittest.TestCase):
    def test_scale_pos_weight_comes_from_the_data_it_is_fitted_on(self):
        import pandas as pd
        from src.data import load_train

        X, y = load_train()
        X, y = X.iloc[:400], y.iloc[:400]  # a "fold" with its own Yes rate
        pipe = train.build_pipeline("xgboost", "class_weight", "full")
        pipe.set_params(model__n_estimators=5).fit(X, y)
        expected = (y == 0).sum() / (y == 1).sum()
        self.assertAlmostEqual(pipe.named_steps["model"].get_params()["scale_pos_weight"], expected)


class MergeResultsTest(unittest.TestCase):
    def test_rerun_replaces_only_the_requested_models(self):
        import pandas as pd

        old = pd.DataFrame({"model": ["logreg", "random_forest"], "recall_mean": [0.1, 0.2]})
        new = pd.DataFrame({"model": ["logreg"], "recall_mean": [0.9]})
        merged = train.merge_results(old, new, ["logreg"])
        self.assertEqual(sorted(zip(merged["model"], merged["recall_mean"])),
                         [("logreg", 0.9), ("random_forest", 0.2)])


class TuneTest(unittest.TestCase):
    def test_returns_threshold_tuned_model_with_nested_cv_scores(self):
        from sklearn.model_selection import TunedThresholdClassifierCV
        from src.data import load_train

        X, y = load_train()
        final, row = train.tune("logreg", "class_weight", "no_health", X, y, n_iter=2)
        self.assertIsInstance(final, TunedThresholdClassifierCV)
        self.assertTrue(0.0 < row["threshold"] < 1.0)
        self.assertAlmostEqual(row["threshold"], final.best_threshold_)
        for metric in C.CV_METRICS:
            self.assertIn(f"{metric}_mean", row)
            self.assertIn(f"{metric}_std", row)
        self.assertEqual((row["model"], row["imbalance"], row["feature_set"]),
                         ("logreg", "class_weight", "no_health"))
        self.assertIn("C", row["best_params"])
        # The tuned threshold must not collapse to "Yes for everyone" (precision
        # 0.27 = Yes rate) while still finding most Yes cases (baseline recall 0).
        self.assertGreater(row["precision_mean"], 0.35)
        self.assertGreater(row["recall_mean"], 0.6)
        self.assertEqual(final.predict(X.head(3)).shape, (3,))


if __name__ == "__main__":
    unittest.main()
