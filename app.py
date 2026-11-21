from __future__ import annotations

import json
import logging
import os
import sys

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

MODEL_URI = os.environ.get("MODEL_URI", "models:/trees@champion")

FEATURE_COLUMNS = [
    "cows",
    "buffaloes",
    "fodder_price_rs",
    "temp_c",
    "vet_visits",
    "feed_kg",
    "breed_mix",
    "village_cluster",
]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp168")
logger.setLevel(logging.INFO)
logger.handlers = [_handler]
logger.propagate = False

mlflow.set_tracking_uri(TRACKING_URI)

app = FastAPI(
    title="MP168 Milk Collection API",
    description="Predicts a village centre's daily milk collection (litres/day).",
    version="1.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model


class PredictRequest(BaseModel):
    cows: int = Field(..., le=350, description="Cows registered (max 350).")
    buffaloes: int = Field(..., description="Buffaloes registered.")
    fodder_price_rs: float = Field(..., description="Green fodder price, Rs/kg.")
    temp_c: float = Field(..., description="Maximum temperature, deg C.")
    vet_visits: int = Field(..., description="Vet visits this month.")
    feed_kg: float = Field(..., ge=50, description="Cattle feed sold, kg (min 50).")
    breed_mix: str = Field(..., description="Main breed: desi, crossbred, mixed.")
    village_cluster: str = Field(..., description="Cluster A, Cluster B or Cluster C.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cows": 120,
                "buffaloes": 30,
                "fodder_price_rs": 18.5,
                "temp_c": 32.0,
                "vet_visits": 2,
                "feed_kg": 240.0,
                "breed_mix": "crossbred",
                "village_cluster": "Cluster A",
            }
        }
    }


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted litres_per_day.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_URI}' is not available: {exc}",
        ) from exc

    row = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(row)[0])

    logger.info(
        "prediction",
        extra={
            "extra_fields": {
                "event": "prediction",
                "prediction": prediction,
                "village_cluster": payload.village_cluster,
                "breed_mix": payload.breed_mix,
            }
        },
    )

    return PredictResponse(prediction=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
