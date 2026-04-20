import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/fraud_model.joblib"


def train_model(
    df: pd.DataFrame,
    model_path: str = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost fraud detection model.

    Args:
        df:           Transformed DataFrame with 'isFraud' and 'TransactionID' columns.
        model_path:   Where to save the trained model.
        test_size:    Fraction of data to use for validation.
        random_state: Reproducibility seed.

    Returns:
        Trained XGBClassifier.
    """
    required_cols = {"isFraud", "TransactionID"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    logger.info("Preparing data for training...")
    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]

    if y.nunique() < 2:
        raise ValueError("Target column 'isFraud' must have at least two classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} rows | Val: {X_val.shape[0]} rows")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,           # reduced from 6 — shallower trees generalize better
        learning_rate=0.05,
        subsample=0.7,         # reduced from 0.8 — more regularization
        colsample_bytree=0.7,  # reduced from 0.8
        min_child_weight=5,    # added — prevents splits on very few fraud samples
        gamma=1.0,             # added — minimum loss reduction to make a split
        reg_alpha=0.1,         # added — L1 regularization
        reg_lambda=1.5,        # added — L2 regularization
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)

    logger.info(f"ROC AUC: {auc:.4f} | F1 Score: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    input_path = "data/processed/train_transformed_sample.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Run transformation first. Expected: {input_path}")

    df = pd.read_csv(input_path)
    train_model(df)
