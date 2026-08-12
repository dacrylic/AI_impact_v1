"""FastAPI service for the E0/E1/E23 AI-impact classifier."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_impact_classifier.production import predict_with_review_flags


DEFAULT_MODEL_PATH = Path("models/e0_e1_e23_sparse_svc.joblib")


class PredictionRequest(BaseModel):
    keytask_content: str = Field(min_length=1, description="Task content to classify.")
    jobrole_title: Optional[str] = Field(default=None, description="Optional job role title.")
    include_diagnostics: bool = Field(default=False, description="Include margin and vocabulary-coverage details.")


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    path = Path(os.environ.get("AI_IMPACT_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


app = FastAPI(
    title="AI Impact Classifier API",
    version="1.0.0",
    description="CPU-first E0/E1/E23 task classifier with a review-priority signal.",
)


@app.get("/health")
def health() -> dict:
    try:
        artifact = load_artifact()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "status": "ok",
        "artifact_version": artifact["artifact_version"],
        "target_labels": artifact["target_labels"],
    }


@app.post("/v1/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        artifact = load_artifact()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    frame = pd.DataFrame([{
        "keytask_content": request.keytask_content,
        "jobrole_title": request.jobrole_title or "",
    }])
    result = predict_with_review_flags(
        artifact["model"],
        frame,
        artifact["review_calibration"],
        include_diagnostics=request.include_diagnostics,
    )[0]
    return result
