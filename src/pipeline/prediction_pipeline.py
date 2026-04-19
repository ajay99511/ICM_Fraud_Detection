import pandas as pd
import joblib
import os
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Inference pipeline that reuses the exact encoders and medians fitted during training,
    ensuring feature distributions match what the model was trained on.
    """

    def __init__(
        self,
        model_path: str = "models/fraud_model.joblib",
        encoders_path: str = "models/label_encoders.joblib",
        medians_path: str = "models/train_medians.joblib",
        schema_path: str = "data/processed/train_transformed_sample.csv",
    ):
        for path, label in [
            (model_path, "model"),
            (encoders_path, "label encoders"),
            (medians_path, "training medians"),
            (schema_path, "schema reference"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required {label} file not found: {path}. "
                    "Run ingestion → transformation → training first."
                )

        self.model = joblib.load(model_path)
        self.label_encoders: dict = joblib.load(encoders_path)
        self.medians: dict = joblib.load(medians_path)

        # Derive expected feature columns from the saved schema (excludes target/ID)
        schema_df = pd.read_csv(schema_path, nrows=1)
        drop_cols = [c for c in ["isFraud", "TransactionID"] if c in schema_df.columns]
        self.feature_cols: list[str] = schema_df.drop(columns=drop_cols).columns.tolist()

        logger.info(
            f"Pipeline loaded: {len(self.feature_cols)} features, "
            f"{len(self.label_encoders)} encoders, {len(self.medians)} medians."
        )

    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        df = input_df.copy()

        # ── Time engineering ──────────────────────────────────────────────────
        if "TransactionDT" in df.columns:
            df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
            df["Transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7

        # ── Categorical encoding (use saved encoders — no re-fitting) ─────────
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                known = set(le.classes_)
                df[col] = (
                    df[col]
                    .astype(str)
                    .fillna("nan")
                    .apply(lambda x: x if x in known else le.classes_[0])
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0  # unseen column — default to 0

        # ── Fill numeric NAs with training medians ────────────────────────────
        for col, median_val in self.medians.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(median_val)

        # ── Align columns to training schema ─────────────────────────────────
        for col in self.feature_cols:
            if col not in df.columns:
                # Use training median if available, else 0
                df[col] = self.medians.get(col, 0)

        # Fill any remaining NAs
        df = df.fillna(0)

        return df[self.feature_cols]

    def predict(self, input_data: dict[str, Any]) -> float:
        """
        Predict fraud probability for a single transaction dict.

        Returns:
            Fraud probability in [0, 1].
        """
        df = pd.DataFrame([input_data])
        processed_df = self.preprocess(df)
        prob = self.model.predict_proba(processed_df)[:, 1]
        return float(prob[0])

    def predict_batch(self, records: list[dict[str, Any]]) -> list[float]:
        """
        Predict fraud probabilities for a batch of transaction dicts.

        Returns:
            List of fraud probabilities.
        """
        if not records:
            return []
        df = pd.DataFrame(records)
        processed_df = self.preprocess(df)
        probs = self.model.predict_proba(processed_df)[:, 1]
        return probs.tolist()
