# IEEE-CIS Fraud Detection

A production-grade, real-time transaction fraud detection system built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). The system exposes a REST API that scores individual or batched financial transactions and returns a calibrated fraud probability — the same architectural pattern used by payment processors like Stripe, Visa, and PayPal.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Pipeline](#pipeline)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Data Transformation](#2-data-transformation)
  - [3. Model Training](#3-model-training)
- [API Reference](#api-reference)
  - [Health Check](#get-)
  - [Single Prediction](#post-predict)
  - [Batch Prediction](#post-predictbatch)
- [Configuration](#configuration)
- [Testing](#testing)
- [Key Design Decisions](#key-design-decisions)
- [Known Issues & Roadmap](#known-issues--roadmap)

---

## Overview

Every card transaction needs a fraud decision in under 100ms. Manual review doesn't scale across billions of daily transactions — this is why every major financial institution runs an ML-based fraud scoring service in their payment stack.

This project implements that scoring service end-to-end:

- Merges raw transaction and device identity signals
- Engineers time-based and categorical features
- Trains an XGBoost classifier with class-imbalance handling
- Serves predictions via a FastAPI REST API with a configurable decision threshold

The trained model outputs a **fraud probability [0, 1]**. Downstream decision engines use this score to auto-approve, auto-decline, or route to manual review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                 │
│  Raw CSVs  ──►  data_ingestion  ──►  data_transformation  ──►  │
│                                                                 │
│  model_trainer  ──►  fraud_model.joblib                        │
│                  ──►  label_encoders.joblib                     │
│                  ──►  train_medians.joblib                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       Inference Service                         │
│                                                                 │
│  POST /predict  ──►  PredictionPipeline.preprocess()           │
│                       │  • Time engineering                     │
│                       │  • Saved LabelEncoder transform         │
│                       │  • Median imputation                    │
│                       │  • Schema alignment                     │
│                       └──►  XGBClassifier.predict_proba()      │
│                             └──►  fraud_probability [0, 1]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── app.py                          # FastAPI application entry point
├── requirements.txt
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Merge & sample raw transaction + identity data
│   │   ├── data_transformation.py  # Feature engineering, encoding, imputation
│   │   └── model_trainer.py        # XGBoost training with early stopping
│   │
│   └── pipeline/
│       └── prediction_pipeline.py  # Inference pipeline (preprocess + predict)
│
├── data/
│   ├── raw/                        # Source CSVs (git-ignored)
│   └── processed/                  # Sampled & transformed CSVs (git-ignored)
│
├── models/                         # Serialized artifacts (git-ignored)
│   ├── fraud_model.joblib
│   ├── label_encoders.joblib
│   └── train_medians.joblib
│
└── tests/
    ├── test_app.py                 # FastAPI integration tests
    └── test_pipeline.py            # PredictionPipeline unit tests
```

---

## Quickstart

### Prerequisites

- Python 3.13+
- The raw IEEE-CIS dataset CSVs placed in `data/raw/`:
  - `train_transaction.csv`
  - `train_identity.csv`

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline

Each step must be run in order. Each script saves its output for the next step.

```bash
# Step 1 — merge and sample raw data
python src/components/data_ingestion.py

# Step 2 — engineer features, fit and save encoders + medians
python src/components/data_transformation.py

# Step 3 — train XGBoost model and save artifact
python src/components/model_trainer.py
```

After this, `models/` will contain three artifacts:
- `fraud_model.joblib` — trained XGBoost classifier
- `label_encoders.joblib` — fitted `LabelEncoder` per categorical column
- `train_medians.joblib` — per-column medians for numeric imputation

### 3. Start the API server

```bash
python app.py
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Pipeline

### 1. Data Ingestion

**File:** `src/components/data_ingestion.py`

Loads `train_transaction.csv` and `train_identity.csv`, merges them on `TransactionID` via a left join, optionally downsamples to a configurable row count, and writes the result to `data/processed/train_sample.csv`.

Memory optimization is applied by downcasting numeric columns to the smallest safe dtype — a practical necessity given the dataset's 400+ feature columns.

```python
from src.components.data_ingestion import ingest_data

df = ingest_data(
    trans_path="data/raw/train_transaction.csv",
    id_path="data/raw/train_identity.csv",
    output_path="data/processed/train_sample.csv",
    sample_size=10_000,
)
```

### 2. Data Transformation

**File:** `src/components/data_transformation.py`

Transforms the merged DataFrame for model training or inference. Operates in two modes controlled by the `fit` flag.

| Mode | Behavior |
|---|---|
| `fit=True` (training) | Drops high-null columns, fits encoders and medians, saves artifacts |
| `fit=False` (inference) | Loads saved artifacts, applies identical transforms |

**Features engineered:**

| Feature | Logic | Signal |
|---|---|---|
| `Transaction_hour` | `(TransactionDT // 3600) % 24` | Fraud spikes at off-hours |
| `Transaction_day` | `(TransactionDT // 86400) % 7` | Day-of-week behavioral patterns |
| Categorical encoding | `LabelEncoder` per column, saved to disk | Card network, email domain, device type |
| Numeric imputation | Per-column training medians | Handles real-world missing fields |

```python
from src.components.data_transformation import transform_data

# Training mode — fits and saves encoders/medians
df_transformed = transform_data(df, fit=True)

# Inference mode — loads saved artifacts
df_transformed = transform_data(df_new, fit=False)
```

### 3. Model Training

**File:** `src/components/model_trainer.py`

Trains an XGBoost classifier on the transformed data with an 80/20 stratified train/validation split.

**Key choices:**

- `scale_pos_weight = neg / pos` — corrects for class imbalance (~3–5% fraud rate in real data)
- `early_stopping_rounds=20` — halts training when validation AUC stops improving
- `eval_metric="auc"` — ROC-AUC is the correct metric for imbalanced binary classification; accuracy is misleading here

```python
from src.components.model_trainer import train_model

model = train_model(df_transformed, model_path="models/fraud_model.joblib")
```

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `/docs` (Swagger UI) | `/redoc` (ReDoc)

---

### `GET /`

Health check.

**Response `200`**
```json
{
  "message": "Fraud Detection API is running",
  "status": "ok"
}
```

---

### `POST /predict`

Score a single transaction for fraud probability.

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `TransactionDT` | `integer` | ✅ | Transaction timestamp offset (seconds) |
| `TransactionAmt` | `float > 0` | ✅ | Transaction amount |
| `ProductCD` | `string` | ✅ | Product code (`W`, `H`, `C`, `S`, `R`) |
| `card1` | `integer` | ✅ | Card feature 1 |
| `card2` | `float` | — | Card feature 2 |
| `card3` | `float` | — | Card feature 3 |
| `card4` | `string` | — | Card network (e.g. `visa`, `mastercard`) |
| `card5` | `float` | — | Card feature 5 |
| `card6` | `string` | — | Card type (e.g. `credit`, `debit`) |
| `addr1` | `float` | — | Billing address feature |
| `addr2` | `float` | — | Billing address feature |
| `dist1` | `float` | — | Distance feature 1 |
| `dist2` | `float` | — | Distance feature 2 |
| `P_emaildomain` | `string` | — | Purchaser email domain |
| `R_emaildomain` | `string` | — | Recipient email domain |

**Example request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionDT": 86400,
    "TransactionAmt": 49.99,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com"
  }'
```

**Response `200`**
```json
{
  "fraud_probability": 0.031842,
  "is_fraud": false,
  "threshold": 0.5
}
```

| Field | Type | Description |
|---|---|---|
| `fraud_probability` | `float` | Model confidence score [0, 1] |
| `is_fraud` | `bool` | `true` if `fraud_probability > threshold` |
| `threshold` | `float` | Decision threshold in use |

**Error responses**

| Status | Cause |
|---|---|
| `422` | Missing required field, invalid type, or non-positive amount |
| `500` | Internal prediction error |
| `503` | Model pipeline not loaded |

---

### `POST /predict/batch`

Score up to 1000 transactions in a single request.

**Example request**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionDT": 86400, "TransactionAmt": 49.99, "ProductCD": "W", "card1": 13926},
      {"TransactionDT": 90000, "TransactionAmt": 999.0, "ProductCD": "H", "card1": 4497}
    ]
  }'
```

**Response `200`**
```json
{
  "results": [
    {"fraud_probability": 0.031842, "is_fraud": false, "threshold": 0.5},
    {"fraud_probability": 0.784201, "is_fraud": true,  "threshold": 0.5}
  ]
}
```

---

## Configuration

All runtime settings are controlled via environment variables. No code changes needed.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/fraud_model.joblib` | Path to trained XGBoost model |
| `ENCODERS_PATH` | `models/label_encoders.joblib` | Path to fitted label encoders |
| `MEDIANS_PATH` | `models/train_medians.joblib` | Path to training medians |
| `SCHEMA_PATH` | `data/processed/train_transformed_sample.csv` | Feature schema reference |
| `FRAUD_THRESHOLD` | `0.5` | Decision threshold for `is_fraud` flag |

**Example — lower threshold for a high-risk card program:**
```bash
FRAUD_THRESHOLD=0.3 python app.py
```

---

## Testing

The test suite uses `unittest` and requires Python 3.13+.

```bash
# Run all tests
py -3.13 -m pytest tests/ -v

# Run only unit tests (no artifacts required)
py -3.13 -m pytest tests/test_pipeline.py -v

# Run only API integration tests
py -3.13 -m pytest tests/test_app.py -v
```

**Test coverage by class:**

| Class | File | Requires Artifacts | What It Tests |
|---|---|---|---|
| `TestPredictionPipelineInit` | `test_pipeline.py` | No | Init, missing file errors |
| `TestPreprocess` | `test_pipeline.py` | No | Time features, encoding, schema alignment, immutability |
| `TestPredict` | `test_pipeline.py` | No | Output type, range, stub value, extra fields |
| `TestPredictBatch` | `test_pipeline.py` | No | Empty batch, length, consistency with single |
| `TestHealthCheck` | `test_app.py` | No | `GET /` status and body |
| `TestPredictEndpoint` | `test_app.py` | No | Schema, types, validation, error handling |
| `TestBatchPredictEndpoint` | `test_app.py` | No | Batch size, fields, error handling |
| `TestRealArtifacts` | Both | **Yes** | End-to-end with real trained model |

`TestRealArtifacts` tests are automatically skipped if trained artifacts are not present.

---

## Key Design Decisions

**Artifact separation**  
The model, encoders, and medians are saved as three independent files. This allows each to be versioned, audited, and redeployed independently — a standard MLOps practice.

**No re-fitting at inference**  
`LabelEncoder.fit_transform()` on a single row produces arbitrary encodings that don't match training. The pipeline loads saved encoders and uses `transform()` only. Unseen categories are mapped to the first known class rather than raising an error.

**Schema alignment**  
At startup, the pipeline reads one row of the training schema CSV to derive the exact ordered feature list the model expects. Missing columns are filled with training medians or zero. Extra columns are dropped. This prevents silent feature mismatch bugs.

**Configurable threshold**  
The `is_fraud` boolean is derived from `fraud_probability > FRAUD_THRESHOLD`. The threshold is an env var because risk tolerance is a business decision, not a model decision. Different card programs, geographies, or transaction types may warrant different thresholds.

**Class imbalance handling**  
`scale_pos_weight = negative_count / positive_count` tells XGBoost to penalize missed fraud detections more heavily. Without this, a model trained on ~3% fraud data will learn to predict "not fraud" for everything and achieve 97% accuracy while being completely useless.

---

## Known Issues & Roadmap

**Current known issue — app test mock patching**  
`TestPredictEndpoint` and `TestBatchPredictEndpoint` currently fail because `patch("app.PredictionPipeline", ...)` in `setUp` does not intercept the constructor call inside the `lifespan` async context manager. The fix is to patch `app.pipeline` (the global variable) directly within each test method. This is a test infrastructure issue and does not affect the application itself.

**Roadmap**
- [ ] Fix app test mock strategy to patch `app.pipeline` directly
- [ ] Add `models/label_encoders.joblib` and `models/train_medians.joblib` generation to CI
- [ ] Add model versioning and artifact registry integration
- [ ] Add Prometheus metrics endpoint (`/metrics`) for latency and prediction distribution monitoring
- [ ] Add input drift detection using `evidently` (already in `requirements.txt`)
- [ ] Dockerize the service for container-based deployment
- [ ] Add `/health/ready` and `/health/live` endpoints for Kubernetes probes

---

## Data Source

This project uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset, a collaboration between IEEE, the Computational Intelligence Society, and **Vesta Corporation** — a real fraud prevention company. The features reflect actual signals used in production fraud systems.

Raw data files are not included in this repository and must be downloaded separately from Kaggle.

---

## What Makes This "Production-Accepted" Specifically

Artifact persistence — encoders, medians, and model saved separately so they can be versioned and deployed independently
Schema alignment — the pipeline reads the training schema at startup and enforces it at inference, preventing silent feature mismatch bugs
Lifespan management — model loads once at startup, not on every request
Batch endpoint — real systems don't always score one transaction at a time; batch scoring is used for end-of-day reconciliation, risk reporting, and backfill jobs
Configurable threshold — business logic separated from model logic
Graceful handling of unseen categories — a new card network or email domain won't crash the service
