import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENCODERS_PATH = "models/label_encoders.joblib"
MEDIANS_PATH = "models/train_medians.joblib"
DROP_COLS_PATH = "models/drop_cols.joblib"


def transform_data(
    df: pd.DataFrame,
    fit: bool = True,
    encoders_path: str = ENCODERS_PATH,
    medians_path: str = MEDIANS_PATH,
    drop_cols_path: str = DROP_COLS_PATH,
) -> pd.DataFrame:
    """
    Transform raw merged DataFrame for model training or inference.

    Args:
        df:            Input DataFrame.
        fit:           If True, fit encoders/medians and save them (training mode).
                       If False, load saved encoders/medians and apply (inference mode).
        encoders_path: Path to save/load fitted LabelEncoders.
        medians_path:  Path to save/load training medians.
        drop_cols_path: Path to save/load columns dropped during training.

    Returns:
        Transformed DataFrame.
    """
    df = df.copy()

    # Columns that must pass through untouched — never transform target or ID
    PASSTHROUGH = [c for c in ["isFraud", "TransactionID"] if c in df.columns]

    # ── 1. Drop columns with >90% missing values ──────────────────────────────
    if fit:
        # Exclude passthrough cols from the null-drop check
        null_pct = df.drop(columns=PASSTHROUGH).isnull().sum() / len(df)
        drop_cols = null_pct[null_pct > 0.9].index.tolist()
        logger.info(f"Dropping {len(drop_cols)} columns with >90% missing values.")
        df = df.drop(columns=drop_cols)
        os.makedirs(os.path.dirname(drop_cols_path), exist_ok=True)
        joblib.dump(drop_cols, drop_cols_path)
        logger.info(f"Saved drop_cols list ({len(drop_cols)} cols) to {drop_cols_path}")
    else:
        if not os.path.exists(drop_cols_path):
            raise FileNotFoundError(f"drop_cols not found at {drop_cols_path}. Run training first.")
        drop_cols = joblib.load(drop_cols_path)
        cols_to_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    # ── 2. Time engineering ───────────────────────────────────────────────────
    if "TransactionDT" in df.columns:
        df["Transaction_hour"] = (df["TransactionDT"] // 3600) % 24
        df["Transaction_day"] = (df["TransactionDT"] // (3600 * 24)) % 7

    # ── 3. Categorical encoding ───────────────────────────────────────────────
    # Exclude passthrough cols — they're numeric/int, not categories to encode
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns
        if c not in PASSTHROUGH
    ]

    if fit:
        label_encoders: dict[str, LabelEncoder] = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("nan"))
            label_encoders[col] = le
        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        joblib.dump(label_encoders, encoders_path)
        logger.info(f"Saved {len(label_encoders)} LabelEncoders to {encoders_path}")
    else:
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"LabelEncoders not found at {encoders_path}. Run training first.")
        label_encoders = joblib.load(encoders_path)
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Map unseen categories to a dedicated 'unknown' class
                known = set(le.classes_)
                df[col] = df[col].astype(str).fillna("nan").apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
            else:
                df[col] = 0  # column not seen during training

    # ── 4. Numeric missing values ─────────────────────────────────────────────
    # Exclude passthrough cols from median imputation
    num_cols = [
        c for c in df.select_dtypes(include=["number"]).columns
        if c not in PASSTHROUGH
    ]

    if fit:
        medians = {col: df[col].median() for col in num_cols if df[col].isnull().any()}
        os.makedirs(os.path.dirname(medians_path), exist_ok=True)
        joblib.dump(medians, medians_path)
        logger.info(f"Saved medians for {len(medians)} columns to {medians_path}")
    else:
        if not os.path.exists(medians_path):
            raise FileNotFoundError(f"Training medians not found at {medians_path}. Run training first.")
        medians = joblib.load(medians_path)

    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    # Fill any remaining NAs with 0 (columns not seen during training)
    df = df.fillna(0)

    logger.info("Transformation complete.")
    return df


if __name__ == "__main__":
    input_path = "data/processed/train_sample.csv"
    output_path = "data/processed/train_transformed_sample.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Run data ingestion first. Expected: {input_path}")

    df = pd.read_csv(input_path)
    df_transformed = transform_data(df, fit=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_transformed.to_csv(output_path, index=False)
    logger.info(f"Saved transformed data to {output_path}")
