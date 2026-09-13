"""FastAPI application for the approved AI-impact classifier."""
from __future__ import annotations

import os
from hmac import compare_digest
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from .production.contracts import compose_task_from_aop
from .production.predictor import ModelNotReadyError, ProductionPredictor


DEFAULT_MODEL_PATH = "artifacts/current/model.joblib"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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


def require_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """Protect scoring routes while leaving platform health checks public."""
    expected = os.getenv("AI_IMPACT_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "authentication_not_configured", "message": "The API key has not been configured for this deployment."},
        )
    if not api_key or not compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "authentication_failed", "message": "Provide a valid X-API-Key header."},
        )


@lru_cache(maxsize=1)
def get_predictor() -> ProductionPredictor:
    return ProductionPredictor.from_path(_model_path())


def _response(request: PredictionRequest) -> PredictionResponse:
    try:
        task = request.keytask_content
        if not task:
            if not request.action or not request.object_:
                raise ValueError("Provide keytaskContent, or provide both action and object as an AOP input.")
            task = compose_task_from_aop(request.action, request.object_, request.purpose)
        prediction = get_predictor().predict(
            task,
            request.jobrole_title,
            request.action,
            request.object_,
            request.purpose,
        )
    except ModelNotReadyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "model_not_ready", "message": str(error)},
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_request", "message": str(error)},
        ) from error
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


def _json_safe(value: object) -> object:
    """Make validation errors safe when a non-standard JSON NaN is received."""
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": {
                "code": "validation_error",
                "message": "Invalid request. Text fields must be JSON strings; omit optional fields or use null instead of NaN or Infinity.",
                "details": _json_safe(error.errors()),
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, error: HTTPException) -> JSONResponse:
    detail = error.detail if isinstance(error.detail, dict) else {"code": "request_error", "message": str(error.detail)}
    return JSONResponse(status_code=error.status_code, content={"error": detail}, headers=error.headers)


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
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "model_not_ready", "message": str(error)}) from error
    return {"status": "ready", "modelVersion": predictor.model_version}


@app.get("/v1/model", dependencies=[Depends(require_api_key)])
def model_info() -> dict[str, str]:
    try:
        predictor = get_predictor()
    except ModelNotReadyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"code": "model_not_ready", "message": str(error)}) from error
    return {"modelVersion": predictor.model_version, "target": "E0/E1/E23"}


@app.post("/v1/predict", response_model=PredictionResponse, response_model_by_alias=True, dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest) -> PredictionResponse:
    return _response(request)


@app.post("/v1/predict:batch", response_model=BatchPredictionResponse, response_model_by_alias=True, dependencies=[Depends(require_api_key)])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=[_response(item) for item in request.items])
