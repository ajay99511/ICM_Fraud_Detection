import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"Memory before optimization: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        if col_type == object:
            df[col] = df[col].astype("category")
        elif str(col_type).startswith("int"):
            c_min, c_max = df[col].min(), df[col].max()
            for dtype in [np.int8, np.int16, np.int32, np.int64]:
                if np.iinfo(dtype).min < c_min and c_max < np.iinfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break
        elif str(col_type).startswith("float"):
            c_min, c_max = df[col].min(), df[col].max()
            for dtype in [np.float32, np.float64]:
                if np.finfo(dtype).min < c_min and c_max < np.finfo(dtype).max:
                    df[col] = df[col].astype(dtype)
                    break

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info(f"Memory after optimization: {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)")
    return df


def ingest_data(
    trans_path: str = "data/raw/train_transaction.csv",
    id_path: str = "data/raw/train_identity.csv",
    output_path: str = "data/processed/train_sample.csv",
    sample_size: int = 10_000,
    optimize_memory: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load, merge, stratified-sample and memory-optimize the raw data.
    Uses stratified sampling to preserve the fraud class ratio in the sample.
    Returns the full merged DataFrame.
    """
    for path in [trans_path, id_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")

    logger.info("Loading transaction data...")
    train_trans = pd.read_csv(trans_path)

    logger.info("Loading identity data...")
    train_id = pd.read_csv(id_path)

    logger.info("Merging on TransactionID (left join)...")
    train = pd.merge(train_trans, train_id, on="TransactionID", how="left")

    if optimize_memory:
        train = reduce_mem_usage(train)

    logger.info(f"Merged shape: {train.shape}")
    logger.info(f"Target distribution:\n{train['isFraud'].value_counts(normalize=True).to_string()}")

    missing_top = train.isnull().sum().sort_values(ascending=False).head(10) / len(train)
    logger.info(f"Top 10 missing value rates:\n{missing_top.to_string()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Stratified sample — preserves fraud class ratio instead of taking head()
    actual_size = min(sample_size, len(train))
    frac = actual_size / len(train)
    sample = (
        train.groupby("isFraud", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
        .sample(frac=1, random_state=random_state)  # shuffle after groupby
        .reset_index(drop=True)
    )
    sample.to_csv(output_path, index=False)
    logger.info(
        f"Saved stratified {len(sample)}-row sample to {output_path} "
        f"(fraud rate: {sample['isFraud'].mean():.2%})"
    )

    return train


if __name__ == "__main__":
    ingest_data()
