from __future__ import annotations

import json
from pathlib import Path

from ai_impact_classifier.production.predictor import ProductionPredictor


ROOT = Path(__file__).resolve().parents[1]


def test_approved_release_regression_fixture() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/inference_regression.json").read_text())
    predictor = ProductionPredictor.from_path(ROOT / "models/gpt52-current-basic-v2.joblib")

    for case in fixture["cases"]:
        task = case.get("keytaskContent") or " ".join(
            value for value in (case.get("action"), case.get("object"), case.get("purpose")) if value
        )
        prediction = predictor.predict(
            task,
            case.get("jobroleTitle"),
            case.get("action"),
            case.get("object"),
            case.get("purpose"),
        )
        actual = {
            "predictedLabel": prediction.predicted_label,
            "reviewScore": prediction.review_score,
            "modelVersion": prediction.model_version,
        }
        assert actual == case["expected"]
