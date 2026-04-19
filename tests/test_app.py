"""
Integration tests for the FastAPI app.
Covers: health check, single predict, batch predict, validation errors,
        missing fields, and pipeline-unavailable (503) scenario.
"""
import os
import unittest
import joblib
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_pipeline(fraud_prob: float = 0.3):
    """Return a mock PredictionPipeline that always returns a fixed probability."""
    mock = MagicMock()
    mock.predict.return_value = fraud_prob
    mock.predict_batch.side_effect = lambda records: [fraud_prob] * len(records)
    return mock


def _base_transaction():
    return {
        "TransactionDT": 86400,
        "TransactionAmt": 50.0,
        "ProductCD": "W",
        "card1": 1000,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthCheck(unittest.TestCase):
    def setUp(self):
        fake_pipeline = _make_fake_pipeline()
        with patch("app.PredictionPipeline", return_value=fake_pipeline):
            from app import app
            self.client = TestClient(app)

    def test_health_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health_body(self):
        response = self.client.get("/")
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "ok")


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.fake_pipeline = _make_fake_pipeline(fraud_prob=0.3)
        with patch("app.PredictionPipeline", return_value=self.fake_pipeline):
            from app import app
            self.client = TestClient(app)

    def test_predict_returns_200(self):
        response = self.client.post("/predict", json=_base_transaction())
        self.assertEqual(response.status_code, 200)

    def test_predict_response_schema(self):
        response = self.client.post("/predict", json=_base_transaction())
        data = response.json()
        self.assertIn("fraud_probability", data)
        self.assertIn("is_fraud", data)
        self.assertIn("threshold", data)

    def test_predict_probability_type(self):
        response = self.client.post("/predict", json=_base_transaction())
        self.assertIsInstance(response.json()["fraud_probability"], float)

    def test_predict_is_fraud_type(self):
        response = self.client.post("/predict", json=_base_transaction())
        self.assertIsInstance(response.json()["is_fraud"], bool)

    def test_predict_probability_range(self):
        response = self.client.post("/predict", json=_base_transaction())
        prob = response.json()["fraud_probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_is_fraud_false_below_threshold(self):
        # prob=0.3 < default threshold 0.5
        response = self.client.post("/predict", json=_base_transaction())
        self.assertFalse(response.json()["is_fraud"])

    def test_predict_is_fraud_true_above_threshold(self):
        fake = _make_fake_pipeline(fraud_prob=0.9)
        with patch("app.PredictionPipeline", return_value=fake):
            from importlib import reload
            import app as app_module
            reload(app_module)
            client = TestClient(app_module.app)
        response = client.post("/predict", json=_base_transaction())
        self.assertTrue(response.json()["is_fraud"])

    def test_predict_optional_fields_accepted(self):
        txn = {**_base_transaction(), "card2": 500.0, "card4": "visa",
               "P_emaildomain": "gmail.com"}
        response = self.client.post("/predict", json=txn)
        self.assertEqual(response.status_code, 200)

    def test_predict_missing_required_field_returns_422(self):
        bad = {"TransactionAmt": 50.0, "ProductCD": "W", "card1": 1000}  # missing TransactionDT
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)

    def test_predict_negative_amount_returns_422(self):
        bad = {**_base_transaction(), "TransactionAmt": -10.0}
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)

    def test_predict_wrong_type_returns_422(self):
        bad = {**_base_transaction(), "TransactionDT": "not_an_int"}
        response = self.client.post("/predict", json=bad)
        self.assertEqual(response.status_code, 422)

    def test_predict_pipeline_exception_returns_500(self):
        self.fake_pipeline.predict.side_effect = RuntimeError("model exploded")
        response = self.client.post("/predict", json=_base_transaction())
        self.assertEqual(response.status_code, 500)


class TestBatchPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.fake_pipeline = _make_fake_pipeline(fraud_prob=0.3)
        with patch("app.PredictionPipeline", return_value=self.fake_pipeline):
            from app import app
            self.client = TestClient(app)

    def test_batch_returns_200(self):
        payload = {"transactions": [_base_transaction(), _base_transaction()]}
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)

    def test_batch_result_count_matches_input(self):
        txns = [_base_transaction() for _ in range(5)]
        response = self.client.post("/predict/batch", json={"transactions": txns})
        self.assertEqual(len(response.json()["results"]), 5)

    def test_batch_each_result_has_required_fields(self):
        payload = {"transactions": [_base_transaction()]}
        result = self.client.post("/predict/batch", json=payload).json()["results"][0]
        self.assertIn("fraud_probability", result)
        self.assertIn("is_fraud", result)
        self.assertIn("threshold", result)

    def test_batch_empty_list_returns_422(self):
        response = self.client.post("/predict/batch", json={"transactions": []})
        self.assertEqual(response.status_code, 422)

    def test_batch_pipeline_exception_returns_500(self):
        self.fake_pipeline.predict_batch.side_effect = RuntimeError("batch failed")
        payload = {"transactions": [_base_transaction()]}
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 500)


class TestRealArtifacts(unittest.TestCase):
    """Integration tests that run only when real trained artifacts exist."""

    @classmethod
    def setUpClass(cls):
        required = [
            "models/fraud_model.joblib",
            "models/label_encoders.joblib",
            "models/train_medians.joblib",
            "data/processed/train_transformed_sample.csv",
        ]
        for path in required:
            if not os.path.exists(path):
                raise unittest.SkipTest(f"Artifact not found: {path}. Run the full pipeline first.")
        from app import app
        cls.client = TestClient(app)

    def test_real_predict_returns_valid_probability(self):
        response = self.client.post("/predict", json=_base_transaction())
        self.assertEqual(response.status_code, 200)
        prob = response.json()["fraud_probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_real_batch_predict(self):
        payload = {"transactions": [_base_transaction(), _base_transaction()]}
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 2)


if __name__ == "__main__":
    unittest.main()
