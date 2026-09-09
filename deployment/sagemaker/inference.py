"""SageMaker inference handlers using the same artifact as FastAPI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_impact_classifier.production.contracts import compose_task_from_aop
from ai_impact_classifier.production.predictor import ProductionPredictor


def model_fn(model_dir: str) -> ProductionPredictor:
    return ProductionPredictor.from_path(Path(model_dir) / "model.joblib")


def input_fn(request_body: str, request_content_type: str) -> dict[str, Any]:
    if request_content_type != "application/json":
        raise ValueError("Only application/json is supported")
    payload = json.loads(request_body)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return payload


def predict_fn(payload: dict[str, Any], predictor: ProductionPredictor) -> dict[str, Any]:
    task = payload.get("keytaskContent", payload.get("keytask_content"))
    title = payload.get("jobroleTitle", payload.get("jobrole_title"))
    action = payload.get("action")
    object_ = payload.get("object")
    purpose = payload.get("purpose")
    if not task:
        if not action or not object_:
            raise ValueError("Provide keytaskContent, or provide both action and object as an AOP input")
        task = compose_task_from_aop(action, object_, purpose)
    prediction = predictor.predict(task, title, action, object_, purpose)
    return {
        "predictedLabel": prediction.predicted_label,
        "reviewScore": prediction.review_score,
        "modelVersion": prediction.model_version,
    }


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    if accept not in ("application/json", "*/*"):
        raise ValueError("Only application/json is supported")
    return json.dumps(prediction), "application/json"
