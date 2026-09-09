"""Artifact loading and deterministic inference for the serving layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .contracts import CLASS_ORDER, MODEL_VERSION, compose_model_text, lexical_tokens


class ModelNotReadyError(RuntimeError):
    """Raised when the service has no compatible production artifact."""


@dataclass(frozen=True)
class Prediction:
    predicted_label: str
    review_score: float
    model_version: str
    lexical_coverage: float
    decision_margin: float


class ProductionPredictor:
    """A lightweight wrapper around the approved sparse LinearSVC artifact."""

    def __init__(self, artifact: dict[str, Any]):
        if artifact.get("artifact_schema_version") not in {1, 2}:
            raise ModelNotReadyError("Unsupported or missing model artifact schema")
        if tuple(artifact.get("classes", ())) != CLASS_ORDER:
            raise ModelNotReadyError("Model artifact does not use the approved E0/E1/E23 target")
        self._artifact = artifact
        self._features = artifact["features"]
        self._classifier = artifact["classifier"]
        self._word_vectorizer = artifact.get("word_vectorizer")
        self.model_version = str(artifact.get("model_version", MODEL_VERSION))

    @classmethod
    def from_path(cls, path: str | Path) -> "ProductionPredictor":
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise ModelNotReadyError(f"Model artifact was not found: {artifact_path}")
        return cls(joblib.load(artifact_path))

    def _lexical_coverage(self, text: str) -> float:
        tokens = lexical_tokens(text)
        if not tokens:
            return 0.0
        if self._word_vectorizer is None:
            vocabulary = self._features.task_word_vocabulary
        else:
            vocabulary = self._word_vectorizer.vocabulary_
        known = sum(token in vocabulary for token in tokens)
        return float(known / len(tokens))

    def predict(
        self,
        keytask_content: str,
        jobrole_title: str | None = None,
        action: str | None = None,
        object_: str | None = None,
        purpose: str | None = None,
    ) -> Prediction:
        text = compose_model_text(keytask_content, jobrole_title, action, object_, purpose)
        if self._artifact["artifact_schema_version"] == 1:
            matrix = self._features.transform([text])
        else:
            matrix = self._features.transform(pd.DataFrame([{
                "task": keytask_content,
                "jobrole_title": jobrole_title or "",
                "action": action or "",
                "object": object_ or "",
                "purpose": purpose or "",
            }]))
        scores = np.asarray(self._classifier.decision_function(matrix))[0]
        ordered = np.sort(scores)[::-1]
        margin = float(ordered[0] - ordered[1])
        predicted_label = str(self._classifier.classes_[int(np.argmax(scores))])
        # The release review signal uses the task vocabulary as its stable
        # lexical reference. Optional fields have separate vocabularies.
        coverage = self._lexical_coverage(keytask_content)

        # ReviewScore is an operational triage signal, not a calibrated probability.
        margin_risk = float(1.0 / (1.0 + np.exp(margin)))
        coverage_risk = 1.0 - coverage
        task_tokens = lexical_tokens(keytask_content)
        short_task_risk = 0.25 if len(task_tokens) < 4 else 0.0
        review_score = float(np.clip(max(margin_risk, coverage_risk, short_task_risk), 0.0, 1.0))

        return Prediction(
            predicted_label=predicted_label,
            review_score=round(review_score, 4),
            model_version=self.model_version,
            lexical_coverage=round(coverage, 4),
            decision_margin=round(margin, 4),
        )
