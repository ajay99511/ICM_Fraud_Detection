"""
Unit tests for PredictionPipeline.
Covers: happy path, missing columns, unseen categories, batch prediction,
        missing artifact errors, and output range validation.
"""
import os
import unittest
import pandas as pd
import numpy as np
import joblib
import tempfile

from src.pipeline.prediction_pipeline import PredictionPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_artifacts(tmp_dir: str):
    """Create minimal fake artifacts so PredictionPipeline can be instantiated."""
    from sklearn.preprocessing import LabelEncoder

    # Label encoder for ProductCD
    le = LabelEncoder()
    le.fit(["W", "H", "C", "S", "R"])
    encoders = {"ProductCD": le}
    joblib.dump(encoders, os.path.join(tmp_dir, "label_encoders.joblib"))

    # Medians
    medians = {"TransactionAmt": 50.0, "card1": 1000, "Transaction_hour": 12.0}
    joblib.dump(medians, os.path.join(tmp_dir, "train_medians.joblib"))

    # Minimal schema CSV (columns the model expects)
    schema_cols = ["TransactionDT", "TransactionAmt", "ProductCD", "card1",
                   "Transaction_hour", "Transaction_day"]
    schema_df = pd.DataFrame(columns=["isFraud", "TransactionID"] + schema_cols)
    schema_path = os.path.join(tmp_dir, "schema.csv")
    schema_df.to_csv(schema_path, index=False)

    return schema_path, encoders, medians, schema_cols


class _FakeModel:
    """Minimal sklearn-compatible model stub."""
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.3)])


def _build_pipeline(tmp_dir: str) -> PredictionPipeline:
    schema_path, _, _, _ = _make_artifacts(tmp_dir)
    model_path = os.path.join(tmp_dir, "model.joblib")
    joblib.dump(_FakeModel(), model_path)

    return PredictionPipeline(
        model_path=model_path,
        encoders_path=os.path.join(tmp_dir, "label_encoders.joblib"),
        medians_path=os.path.join(tmp_dir, "train_medians.joblib"),
        schema_path=schema_path,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPredictionPipelineInit(unittest.TestCase):
    def test_missing_model_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path, _, _, _ = _make_artifacts(tmp)
            with self.assertRaises(FileNotFoundError):
                PredictionPipeline(
                    model_path=os.path.join(tmp, "nonexistent.joblib"),
                    encoders_path=os.path.join(tmp, "label_encoders.joblib"),
                    medians_path=os.path.join(tmp, "train_medians.joblib"),
                    schema_path=schema_path,
                )

    def test_missing_encoders_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema_path, _, _, _ = _make_artifacts(tmp)
            model_path = os.path.join(tmp, "model.joblib")
            joblib.dump(_FakeModel(), model_path)
            with self.assertRaises(FileNotFoundError):
                PredictionPipeline(
                    model_path=model_path,
                    encoders_path=os.path.join(tmp, "missing_encoders.joblib"),
                    medians_path=os.path.join(tmp, "train_medians.joblib"),
                    schema_path=schema_path,
                )

    def test_feature_cols_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _build_pipeline(tmp)
            self.assertIsInstance(pipeline.feature_cols, list)
            self.assertGreater(len(pipeline.feature_cols), 0)


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = _build_pipeline(self.tmp)

    def test_time_features_created(self):
        df = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0,
                             "ProductCD": "W", "card1": 1000}])
        result = self.pipeline.preprocess(df)
        self.assertIn("Transaction_hour", result.columns)
        self.assertIn("Transaction_day", result.columns)

    def test_time_feature_values(self):
        # 86400 seconds = 1 day → hour=0, day=1
        df = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0,
                             "ProductCD": "W", "card1": 1000}])
        result = self.pipeline.preprocess(df)
        self.assertEqual(result["Transaction_hour"].iloc[0], 0)
        self.assertEqual(result["Transaction_day"].iloc[0], 1)

    def test_output_columns_match_schema(self):
        df = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0,
                             "ProductCD": "W", "card1": 1000}])
        result = self.pipeline.preprocess(df)
        self.assertEqual(list(result.columns), self.pipeline.feature_cols)

    def test_unseen_category_handled(self):
        """Unseen category should not raise — mapped to first known class."""
        df = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0,
                             "ProductCD": "UNKNOWN_CATEGORY", "card1": 1000}])
        try:
            result = self.pipeline.preprocess(df)
            self.assertFalse(result.isnull().any().any(), "No NaNs expected after preprocessing")
        except Exception as exc:
            self.fail(f"Unseen category raised unexpectedly: {exc}")

    def test_missing_columns_filled(self):
        """Columns absent from input should be filled with median or 0."""
        df = pd.DataFrame([{"TransactionDT": 86400}])  # minimal input
        result = self.pipeline.preprocess(df)
        self.assertEqual(list(result.columns), self.pipeline.feature_cols)
        self.assertFalse(result.isnull().any().any())

    def test_no_mutation_of_input(self):
        """Original DataFrame must not be modified."""
        original = pd.DataFrame([{"TransactionDT": 86400, "TransactionAmt": 50.0,
                                   "ProductCD": "W", "card1": 1000}])
        original_copy = original.copy()
        self.pipeline.preprocess(original)
        pd.testing.assert_frame_equal(original, original_copy)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = _build_pipeline(self.tmp)

    def _base_transaction(self):
        return {"TransactionDT": 86400, "TransactionAmt": 50.0,
                "ProductCD": "W", "card1": 1000}

    def test_returns_float(self):
        score = self.pipeline.predict(self._base_transaction())
        self.assertIsInstance(score, float)

    def test_probability_in_range(self):
        score = self.pipeline.predict(self._base_transaction())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_stub_returns_expected_value(self):
        # _FakeModel always returns 0.3
        score = self.pipeline.predict(self._base_transaction())
        self.assertAlmostEqual(score, 0.3, places=5)

    def test_extra_fields_ignored(self):
        """Extra fields in input should not cause errors."""
        data = {**self._base_transaction(), "extra_field": "garbage", "another": 999}
        score = self.pipeline.predict(data)
        self.assertIsInstance(score, float)


class TestPredictBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = _build_pipeline(self.tmp)

    def _txn(self):
        return {"TransactionDT": 86400, "TransactionAmt": 50.0,
                "ProductCD": "W", "card1": 1000}

    def test_empty_batch_returns_empty(self):
        self.assertEqual(self.pipeline.predict_batch([]), [])

    def test_batch_length_matches_input(self):
        records = [self._txn() for _ in range(5)]
        results = self.pipeline.predict_batch(records)
        self.assertEqual(len(results), 5)

    def test_batch_all_in_range(self):
        records = [self._txn() for _ in range(10)]
        for prob in self.pipeline.predict_batch(records):
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_batch_consistent_with_single(self):
        """Batch result for one record should match single predict."""
        txn = self._txn()
        single = self.pipeline.predict(txn)
        batch = self.pipeline.predict_batch([txn])
        self.assertAlmostEqual(single, batch[0], places=6)


class TestRealArtifacts(unittest.TestCase):
    """Integration tests that run only when real trained artifacts exist."""

    MODEL_PATH = "models/fraud_model.joblib"
    ENCODERS_PATH = "models/label_encoders.joblib"
    MEDIANS_PATH = "models/train_medians.joblib"
    SCHEMA_PATH = "data/processed/train_transformed_sample.csv"

    @classmethod
    def setUpClass(cls):
        for path in [cls.MODEL_PATH, cls.ENCODERS_PATH, cls.MEDIANS_PATH, cls.SCHEMA_PATH]:
            if not os.path.exists(path):
                raise unittest.SkipTest(f"Artifact not found: {path}. Run the full pipeline first.")
        cls.pipeline = PredictionPipeline(
            model_path=cls.MODEL_PATH,
            encoders_path=cls.ENCODERS_PATH,
            medians_path=cls.MEDIANS_PATH,
            schema_path=cls.SCHEMA_PATH,
        )

    def test_predict_returns_valid_probability(self):
        score = self.pipeline.predict({
            "TransactionDT": 86400, "TransactionAmt": 50.0,
            "ProductCD": "W", "card1": 1000, "card2": 500,
        })
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_batch_predict_consistent(self):
        txn = {"TransactionDT": 86400, "TransactionAmt": 50.0,
               "ProductCD": "W", "card1": 1000}
        single = self.pipeline.predict(txn)
        batch = self.pipeline.predict_batch([txn, txn])
        self.assertAlmostEqual(single, batch[0], places=6)
        self.assertAlmostEqual(batch[0], batch[1], places=6)


if __name__ == "__main__":
    unittest.main()
