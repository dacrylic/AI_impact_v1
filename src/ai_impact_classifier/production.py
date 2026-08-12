"""Single-model production prediction and review flags for E0/E1/E23."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline


CLASS_NAMES = ("E0", "E1", "E23")
OUTPUT_SCHEMA_VERSION = "1.1"


@dataclass(frozen=True)
class ReviewThresholds:
    """Thresholds calibrated on held-out/OOF data, not training predictions."""

    minimum_decision_margin: float
    minimum_word_coverage: float
    minimum_char_coverage: float
    minimum_total_coverage: float


@dataclass(frozen=True)
class ReviewCalibration:
    """OOF-derived reference distributions and routing cutoffs."""

    thresholds: ReviewThresholds
    margin_reference: tuple[float, ...]
    word_coverage_reference: tuple[float, ...]
    char_coverage_reference: tuple[float, ...]
    total_coverage_reference: tuple[float, ...]
    medium_review_score: float = 0.80
    high_review_score: float = 0.99


def _combine_text(frame: pd.DataFrame) -> pd.Series:
    title = frame.get("jobrole_title", pd.Series("", index=frame.index)).fillna("").astype(str)
    task = frame["keytask_content"].fillna("").astype(str)
    return "ROLE " + title + " TASK " + task


def _active_features(matrix: sparse.spmatrix) -> np.ndarray:
    return np.asarray(matrix.getnnz(axis=1)).ravel().astype(int)


def _known_ngram_coverage(vectorizer: Any, texts: pd.Series) -> np.ndarray:
    """Fraction of candidate n-grams present in this fitted vectorizer."""
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    result = np.zeros(len(texts), dtype=float)
    for index, value in enumerate(texts):
        candidates = analyzer(value)
        if candidates:
            result[index] = sum(candidate in vocabulary for candidate in candidates) / len(candidates)
    return result


def _percentile_rank(values: np.ndarray, reference: tuple[float, ...]) -> np.ndarray:
    reference_array = np.asarray(reference, dtype=float)
    return np.searchsorted(reference_array, values, side="right") / len(reference_array)


def predict_with_review_flags(
    model: Pipeline,
    frame: pd.DataFrame,
    calibration: ReviewCalibration,
    *,
    include_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    """Return single-model labels and uncertainty/OOV-style review signals.

    ``model`` must be a fitted pipeline with ``features`` (a FeatureUnion with
    ``word`` and ``char`` transformers) and a LinearSVC-like ``classifier``.
    """
    required = {"keytask_content"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    text = _combine_text(frame)
    word_matrix = features.transformer_list[0][1].transform(text)
    char_matrix = features.transformer_list[1][1].transform(text)
    combined = features.transform(text)
    scores = classifier.decision_function(combined)
    if scores.ndim == 1:
        raise ValueError("Expected multiclass classifier scores.")
    order = np.argsort(scores, axis=1)
    winner = order[:, -1]
    margin = scores[np.arange(len(frame)), winner] - scores[np.arange(len(frame)), order[:, -2]]
    word_active = _active_features(word_matrix)
    char_active = _active_features(char_matrix)
    total_active = _active_features(combined)
    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]
    word_coverage = _known_ngram_coverage(word_vectorizer, text)
    char_coverage = _known_ngram_coverage(char_vectorizer, text)
    total_coverage = (word_coverage + char_coverage) / 2.0
    review_score = np.maximum.reduce((
        1.0 - _percentile_rank(margin, calibration.margin_reference),
        1.0 - _percentile_rank(word_coverage, calibration.word_coverage_reference),
        1.0 - _percentile_rank(char_coverage, calibration.char_coverage_reference),
        1.0 - _percentile_rank(total_coverage, calibration.total_coverage_reference),
    ))

    output: list[dict[str, Any]] = []
    for index in range(len(frame)):
        reasons = []
        thresholds = calibration.thresholds
        if margin[index] < thresholds.minimum_decision_margin:
            reasons.append("low_decision_margin")
        if word_coverage[index] < thresholds.minimum_word_coverage:
            reasons.append("low_word_coverage")
        if char_coverage[index] < thresholds.minimum_char_coverage:
            reasons.append("low_char_coverage")
        if total_coverage[index] < thresholds.minimum_total_coverage:
            reasons.append("low_total_feature_coverage")
        if reasons or review_score[index] >= calibration.high_review_score:
            review_level = "high"
        elif review_score[index] >= calibration.medium_review_score:
            review_level = "medium"
        else:
            review_level = "low"
        result = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "predicted_label": CLASS_NAMES[int(winner[index])],
            "needs_review": review_level == "high",
            "review_level": review_level,
            "review_score": float(review_score[index]),
            "review_reasons": reasons,
        }
        if include_diagnostics:
            result.update({
            "decision_margin": float(margin[index]),
            "class_decision_scores": {CLASS_NAMES[class_index]: float(scores[index, class_index]) for class_index in range(len(CLASS_NAMES))},
            "word_active_features": int(word_active[index]),
            "char_active_features": int(char_active[index]),
            "total_active_features": int(total_active[index]),
            "word_coverage": float(word_coverage[index]),
            "char_coverage": float(char_coverage[index]),
            "total_feature_coverage": float(total_coverage[index]),
            "review_thresholds": asdict(thresholds),
            })
        output.append(result)
    return output
