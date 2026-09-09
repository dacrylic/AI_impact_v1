"""Serializable field-aware sparse features used by the production model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import normalize_text


TEXT_FIELDS = ("task", "jobrole_title", "action", "object", "purpose")


def _clean(values: Iterable[object]) -> pd.Series:
    return pd.Series(values, dtype="object").map(normalize_text)


def _cross_action_object(frame: pd.DataFrame) -> pd.Series:
    action = _clean(frame["action"]).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    object_ = _clean(frame["object"]).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return "A_" + action + "__O_" + object_


@dataclass
class FieldAwareTfidfFeatures:
    """Word and character TF-IDF blocks with fixed, documented field weights."""

    vectorizers: dict[str, TfidfVectorizer] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: {
        "task_word": 1.0,
        "task_char_wb": 0.65,
        "task_char_raw": 0.45,
        "title_word": 0.5,
        "action_word": 0.9,
        "object_word": 0.8,
        "purpose_word": 0.5,
        "action_object_word": 0.7,
    })

    def _values(self, frame: pd.DataFrame, name: str) -> pd.Series:
        if name.startswith("task_"):
            return _clean(frame["task"])
        if name == "title_word":
            return _clean(frame["jobrole_title"])
        if name == "action_word":
            return _clean(frame["action"])
        if name == "object_word":
            return _clean(frame["object"])
        if name == "purpose_word":
            return _clean(frame["purpose"])
        if name == "action_object_word":
            return _cross_action_object(frame)
        raise KeyError(name)

    @staticmethod
    def _specifications() -> dict[str, dict[str, object]]:
        common = {"dtype": np.float32, "sublinear_tf": True, "strip_accents": "unicode"}
        return {
            "task_word": {**common, "ngram_range": (1, 2), "min_df": 2, "max_features": 220_000},
            "task_char_wb": {**common, "analyzer": "char_wb", "ngram_range": (2, 6), "min_df": 2, "max_features": 220_000},
            "task_char_raw": {**common, "analyzer": "char", "ngram_range": (3, 6), "min_df": 2, "max_features": 250_000},
            "title_word": {**common, "ngram_range": (1, 3), "min_df": 1, "max_features": 100_000},
            "action_word": {**common, "ngram_range": (1, 2), "min_df": 1, "max_features": 20_000},
            "object_word": {**common, "ngram_range": (1, 2), "min_df": 1, "max_features": 120_000},
            "purpose_word": {**common, "ngram_range": (1, 2), "min_df": 1, "max_features": 60_000},
            "action_object_word": {**common, "ngram_range": (1, 1), "token_pattern": r"(?u)\b\w+\b", "min_df": 1, "max_features": 120_000},
        }

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        self.vectorizers = {}
        parts = []
        for name, specification in self._specifications().items():
            vectorizer = TfidfVectorizer(**specification)
            self.vectorizers[name] = vectorizer
            parts.append(self.weights[name] * vectorizer.fit_transform(self._values(frame, name)))
        return sparse.hstack(parts, format="csr")

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        parts = [
            self.weights[name] * vectorizer.transform(self._values(frame, name))
            for name, vectorizer in self.vectorizers.items()
        ]
        return sparse.hstack(parts, format="csr")

    @property
    def task_word_vocabulary(self) -> dict[str, int]:
        return self.vectorizers["task_word"].vocabulary_
