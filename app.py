import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.pipeline.prediction_pipeline import PredictionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "models/label_encoders.joblib")
MEDIANS_PATH = os.getenv("MEDIANS_PATH", "models/train_medians.joblib")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "data/processed/train_transformed_sample.csv")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

pipeline: Optional[PredictionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline once at startup; release on shutdown."""
    global pipeline
    try:
        pipeline = PredictionPipeline(
            model_path=MODEL_PATH,
            encoders_path=ENCODERS_PATH,
            medians_path=MEDIANS_PATH,
            schema_path=SCHEMA_PATH,
        )
        logger.info("Prediction pipeline loaded successfully.")
    except FileNotFoundError as exc:
        logger.error(f"Startup failed — missing artifact: {exc}")
        raise RuntimeError(str(exc)) from exc
    yield
    pipeline = None
    logger.info("Pipeline released.")


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="Real-time fraud probability scoring for financial transactions.",
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    TransactionDT: int = Field(..., description="Transaction timestamp offset in seconds")
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    ProductCD: str = Field(..., description="Product code")
    card1: int = Field(..., description="Card feature 1")
    card2: Optional[float] = Field(None, description="Card feature 2")
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Probability of fraud [0, 1]")
    is_fraud: bool = Field(..., description=f"True if probability exceeds threshold ({FRAUD_THRESHOLD})")
    threshold: float = Field(..., description="Decision threshold used")


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[PredictionResponse]


@app.get("/", tags=["Health"])
def health_check():
    return {"message": "Fraud Detection API is running", "status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: Transaction):
    """Score a single transaction for fraud probability."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not available.")
    try:
        prob = pipeline.predict(transaction.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=prob > FRAUD_THRESHOLD,
        threshold=FRAUD_THRESHOLD,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_fraud_batch(request: BatchRequest):
    """Score a batch of transactions (up to 1000) in a single call."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline not available.")
    try:
        records = [t.model_dump() for t in request.transactions]
        probs = pipeline.predict_batch(records)
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc

    results = [
        PredictionResponse(
            fraud_probability=round(p, 6),
            is_fraud=p > FRAUD_THRESHOLD,
            threshold=FRAUD_THRESHOLD,
        )
        for p in probs
    ]
    return BatchResponse(results=results)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
