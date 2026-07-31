from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error


def project_ranks(pred: np.ndarray, num_labels: int) -> np.ndarray:
    rounded = np.rint(pred).astype(int)
    return np.clip(rounded, 0, num_labels - 1)


def project_scores_to_ranks(pred_scores: np.ndarray, score_values: np.ndarray) -> np.ndarray:
    score_values = np.asarray(score_values, dtype=float)
    distances = np.abs(pred_scores[:, None] - score_values[None, :])
    return np.argmin(distances, axis=1).astype(int)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, score_values: np.ndarray) -> Dict[str, float]:
    projected_true = project_scores_to_ranks(y_true, score_values=score_values)
    projected_pred = project_scores_to_ranks(y_pred, score_values=score_values)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(np.mean((y_true - y_pred) ** 2))),
        "ordinal_mae": float(mean_absolute_error(projected_true, projected_pred)),
        "accuracy": float(accuracy_score(projected_true, projected_pred)),
        "macro_f1": float(f1_score(projected_true, projected_pred, average="macro")),
        "qwk": float(cohen_kappa_score(projected_true, projected_pred, weights="quadratic")),
    }
