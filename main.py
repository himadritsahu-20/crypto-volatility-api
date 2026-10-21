
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crypto_volatility_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

FEATURE_COLUMNS = metadata["feature_columns"]
THRESHOLD = float(metadata["classification_threshold"])

app = FastAPI(
    title="Crypto Volatility Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Netlify domain before production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature values matching the training pipeline"
    )


@app.get("/")
def home():
    return {
        "message": "Crypto Volatility Prediction API",
        "health": "/health",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata.get("model_name"),
        "threshold": THRESHOLD,
        "target": metadata.get("target"),
        "feature_columns": FEATURE_COLUMNS,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    supplied = request.features
    expected = set(FEATURE_COLUMNS)

    missing = sorted(expected - set(supplied))
    extra = sorted(set(supplied) - expected)

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": extra
            }
        )

    row = pd.DataFrame(
        [[supplied[name] for name in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    if row.isna().any().any():
        raise HTTPException(
            status_code=422,
            detail="Feature values cannot be missing."
        )

    try:
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}"
        )

    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "label": (
            "High volatility" if prediction
            else "Low/moderate volatility"
        ),
        "high_volatility_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "disclaimer": "Experimental model; not financial advice."
    }
