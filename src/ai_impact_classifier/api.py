"""FastAPI application for the approved AI-impact classifier."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .production.contracts import compose_task_from_aop
from .production.predictor import ModelNotReadyError, ProductionPredictor


DEFAULT_MODEL_PATH = "artifacts/current/model.joblib"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", str_strip_whitespace=True)

    keytask_content: Optional[str] = Field(
        alias="keytaskContent",
        max_length=10_000,
        default=None,
        description="Optional already-composed AOP task. When omitted, the API joins supplied action, object, and purpose verbatim.",
    )
    jobrole_title: Optional[str] = Field(default=None, alias="jobroleTitle", max_length=500)
    action: Optional[str] = Field(default=None, max_length=2_000, description="Upstream AOP action. Required with object when keytaskContent is omitted.")
    object_: Optional[str] = Field(default=None, alias="object", max_length=5_000, description="Upstream AOP object. Required with action when keytaskContent is omitted.")
    purpose: Optional[str] = Field(default=None, max_length=5_000, description="Optional upstream AOP component; never inferred by this API.")


class PredictionResponse(BaseModel):
    predicted_label: str = Field(alias="predictedLabel")
    review_score: float = Field(alias="reviewScore", ge=0, le=1)
    model_version: str = Field(alias="modelVersion")

    model_config = ConfigDict(populate_by_name=True)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    items: list[PredictionRequest] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

    model_config = ConfigDict(populate_by_name=True)


def _model_path() -> Path:
    return Path(os.getenv("AI_IMPACT_MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_predictor() -> ProductionPredictor:
    return ProductionPredictor.from_path(_model_path())


def _response(request: PredictionRequest) -> PredictionResponse:
    try:
        task = request.keytask_content
        if not task:
            if not request.action or not request.object_:
                raise ValueError("Provide keytaskContent, or provide both action and object as an AOP input")
            task = compose_task_from_aop(request.action, request.object_, request.purpose)
        prediction = get_predictor().predict(
            task,
            request.jobrole_title,
            request.action,
            request.object_,
            request.purpose,
        )
    except ModelNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return PredictionResponse(
        predictedLabel=prediction.predicted_label,
        reviewScore=prediction.review_score,
        modelVersion=prediction.model_version,
    )


app = FastAPI(
    title="AI Impact Classifier API",
    version="1.0.0",
    description="CPU-only classifier for upstream-normalized atomic AOP tasks; it never decomposes raw prose.",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness endpoint that does not require model loading."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    """Readiness endpoint; returns 503 until the configured artifact loads."""
    try:
        predictor = get_predictor()
    except ModelNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return {"status": "ready", "modelVersion": predictor.model_version}


@app.get("/v1/model")
def model_info() -> dict[str, str]:
    try:
        predictor = get_predictor()
    except ModelNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return {"modelVersion": predictor.model_version, "target": "E0/E1/E23"}


@app.post("/v1/predict", response_model=PredictionResponse, response_model_by_alias=True)
def predict(request: PredictionRequest) -> PredictionResponse:
    return _response(request)


@app.post("/v1/predict:batch", response_model=BatchPredictionResponse, response_model_by_alias=True)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=[_response(item) for item in request.items])
