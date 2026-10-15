import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schema import FEATURE_NAMES, PredictionRequest

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")


def load_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Train it with scripts/train.py "
            "or build the Docker image."
        )
    return joblib.load(MODEL_PATH)


model = load_model()
app = FastAPI(title="Diabetes Triage ML Demo", version=MODEL_VERSION)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = []
    for error in exc.errors():
        errors.append(
            {
                "type": error.get("type", "value_error"),
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", "Invalid input"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "error": "Invalid request input"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, float]:
    values = pd.DataFrame([payload.ordered_values()], columns=FEATURE_NAMES, dtype=float)
    prediction = float(model.predict(values)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("Model returned a non-finite prediction")
    return {"prediction": prediction}
