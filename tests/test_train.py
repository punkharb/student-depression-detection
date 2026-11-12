"""Checks for src/train.py. Run:  python -m unittest discover tests"""
import unittest

from imblearn.over_sampling import SMOTE
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
            self.assertEqual(list(pipe.named_steps), ["prep", "smote", "model"])
            self.assertIsInstance(pipe.named_steps["smote"], SMOTE)
            self.assertEqual(pipe.named_steps["smote"].random_state, C.RANDOM_STATE)
            clf = pipe.named_steps["model"]
            # No double correction: SMOTE runs alone, without class weights.
            self.assertIn(clf.get_params().get("class_weight"), (None,))
            self.assertIn(clf.get_params().get("scale_pos_weight"), (None, 1, 1.0))

    def test_only_logistic_regression_is_scaled(self):
        self.assertTrue(_scales_numeric(train.build_pipeline("logreg", "smote", "full")))
        self.assertFalse(_scales_numeric(train.build_pipeline("random_forest", "smote", "full")))
        self.assertFalse(_scales_numeric(train.build_pipeline("xgboost", "smote", "full")))


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
    def test_picks_best_cv_recall_and_logs_other_metrics(self):
        from src.data import load_train

        X, y = load_train()
        search, row = train.tune("logreg", "class_weight", "no_health", X, y,
                                 param_grid={"C": [0.001, 1.0]})
        self.assertEqual(search.refit, "recall")
        recalls = search.cv_results_["mean_test_recall"]
        self.assertAlmostEqual(row["recall_mean"], max(recalls))
        for metric in C.CV_METRICS:
            self.assertIn(f"{metric}_mean", row)
            self.assertIn(f"{metric}_std", row)
        self.assertEqual(row["model"], "logreg")
        self.assertEqual(row["imbalance"], "class_weight")
        self.assertEqual(row["feature_set"], "no_health")
        self.assertIn("C", row["best_params"])
        # Balanced LR on this data finds real Yes cases: well above the 0% baseline.
        self.assertGreater(row["recall_mean"], 0.3)


if __name__ == "__main__":
    unittest.main()
