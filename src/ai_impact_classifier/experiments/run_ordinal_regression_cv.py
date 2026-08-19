"""Nested group-CV benchmark that regresses the source AI-impact score.

The model sees only key-task text and, optionally, job-role title.  Numeric
impact scales are targets only; no source score or label-derived field is used
as an inference feature.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVR

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle


LABELS = ("E0", "E3", "E12")


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    parameter: float
    e3_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested group-CV sparse ordinal-regression benchmark.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/ordinal_regression_cv5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-only", action="store_true", help="Exclude job-role title; task content remains mandatory.")
    return parser.parse_args()


def _text(frame: pd.DataFrame, task_only: bool) -> pd.Series:
    task = "TASK " + frame.keytask_content.fillna("").astype(str)
    if task_only:
        return task
    return "ROLE " + frame.jobrole_title.fillna("").astype(str) + " " + task


def _features(train_text: pd.Series, evaluation_text: pd.Series) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)
    train_matrix = sparse.hstack([word.fit_transform(train_text), 0.65 * char.fit_transform(train_text)], format="csr")
    evaluation_matrix = sparse.hstack([word.transform(evaluation_text), 0.65 * char.transform(evaluation_text)], format="csr")
    return train_matrix, evaluation_matrix


def _candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for c_value in (0.02, 0.08, 0.3):
        for e3_weight in (1.0, 4.0):
            candidates.append(Candidate("linear_svr", "linear_svr", c_value, e3_weight))
    for alpha in (2.0, 20.0, 100.0):
        for e3_weight in (1.0, 4.0):
            candidates.append(Candidate("ridge", "ridge", alpha, e3_weight))
    for alpha in (0.00003, 0.0001):
        for e3_weight in (1.0, 4.0):
            candidates.append(Candidate("sgd_huber", "sgd_huber", alpha, e3_weight))
    return candidates


def _fit_predict(candidate: Candidate, x_fit: sparse.csr_matrix, target: np.ndarray, ranks: np.ndarray, x_eval: sparse.csr_matrix, seed: int) -> np.ndarray:
    sample_weight = np.where(ranks == 1, candidate.e3_weight, 1.0)
    if candidate.kind == "linear_svr":
        model = LinearSVR(C=candidate.parameter, epsilon=0.1, max_iter=20_000, random_state=seed)
        model.fit(x_fit, target, sample_weight=sample_weight)
        return model.predict(x_eval)
    if candidate.kind == "ridge":
        # Centering avoids an unstable sparse intercept computation on this matrix.
        target_mean = float(np.average(target, weights=sample_weight))
        model = Ridge(alpha=candidate.parameter, solver="lsqr", fit_intercept=False)
        model.fit(x_fit, target - target_mean, sample_weight=sample_weight)
        return model.predict(x_eval) + target_mean
    model = SGDRegressor(
        loss="huber", alpha=candidate.parameter, max_iter=750, tol=1e-3,
        average=True, random_state=seed,
    )
    model.fit(x_fit, target, sample_weight=sample_weight)
    return model.predict(x_eval)


def _decode(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(scores, thresholds, right=False).astype(int)


def _best_thresholds(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Tune two monotonic cut-points on validation rows only for macro F1."""
    values = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_thresholds = np.array([np.quantile(scores, 0.35), np.quantile(scores, 0.8)])
    best_score = f1_score(y, _decode(scores, best_thresholds), average="macro", zero_division=0)
    for _ in range(3):
        improved = False
        for index in range(2):
            for value in values:
                candidate = best_thresholds.copy()
                candidate[index] = value
                if candidate[0] >= candidate[1]:
                    continue
                metric = f1_score(y, _decode(scores, candidate), average="macro", zero_division=0)
                if metric > best_score:
                    best_thresholds, best_score, improved = candidate, metric, True
        if not improved:
            break
    return best_thresholds, float(best_score)


def _metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray, target_scores: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "rank_mae": float(mean_absolute_error(y, pred)),
        "score_mae": float(mean_absolute_error(target_scores, scores)),
        "score_rmse": float(np.sqrt(np.mean((target_scores - scores) ** 2))),
    }


def _select(development: pd.DataFrame, y: np.ndarray, source_scores: np.ndarray, candidates: list[Candidate], task_only: bool, seed: int) -> tuple[Candidate, np.ndarray, dict[str, float]]:
    groups = development[GROUP_COLUMN].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, validation_idx = next(inner.split(development, y, groups))
    fit, validation = development.iloc[fit_idx], development.iloc[validation_idx]
    x_fit, x_validation = _features(_text(fit, task_only), _text(validation, task_only))
    selected: tuple[Candidate, np.ndarray, dict[str, float]] | None = None
    for candidate in candidates:
        scores = _fit_predict(candidate, x_fit, source_scores[fit_idx], y[fit_idx], x_validation, seed)
        thresholds, _ = _best_thresholds(y[validation_idx], scores)
        pred = _decode(scores, thresholds)
        metrics = _metrics(y[validation_idx], pred, scores, source_scores[validation_idx])
        if selected is None or metrics["macro_f1"] > selected[2]["macro_f1"]:
            selected = candidate, thresholds, metrics
    assert selected is not None
    return selected


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy()
    source_scores = frame.ai_impact_score.to_numpy(dtype=float)
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    candidates = _candidates()
    predictions = np.full(len(frame), -1, dtype=int)
    regression_scores = np.full(len(frame), np.nan)
    rows: list[dict[str, object]] = []

    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        leakage = check_split_leakage(
            frame,
            {"development": development.index, "outer_test": test.index},
            group_column=GROUP_COLUMN,
            text_column="target_text",
        )
        if not leakage.passed:
            raise RuntimeError("Outer-fold leakage check failed: " + "; ".join(leakage.messages))
        candidate, thresholds, validation_metrics = _select(development, y[development_idx], source_scores[development_idx], candidates, args.task_only, args.seed + fold)
        x_development, x_test = _features(_text(development, args.task_only), _text(test, args.task_only))
        scores = _fit_predict(candidate, x_development, source_scores[development_idx], y[development_idx], x_test, args.seed + fold)
        pred = _decode(scores, thresholds)
        predictions[test_idx], regression_scores[test_idx] = pred, scores
        row = {"fold": fold, "selected_model": candidate.name, "selected_parameter": candidate.parameter,
               "selected_e3_weight": candidate.e3_weight, "thresholds": json.dumps(thresholds.tolist()),
               "inner_validation_macro_f1": validation_metrics["macro_f1"],
               "shared_jobrole_ids": len(set(groups[development_idx]) & set(groups[test_idx])),
               "duplicate_target_text_warning": leakage.warnings[0] if leakage.warnings else "",
               **_metrics(y[test_idx], pred, scores, source_scores[test_idx])}
        rows.append(row)
        print(json.dumps(row), flush=True)

    if (predictions < 0).any() or np.isnan(regression_scores).any():
        raise RuntimeError("Every row must receive exactly one outer-fold prediction.")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(frame)), "jobrole_id": groups, "y_true": y,
                  "y_pred": predictions, "regression_score": regression_scores}).to_csv(out / "oof_predictions.csv", index=False)
    matrix = confusion_matrix(y, predictions, labels=range(len(LABELS)))
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(out / "oof_confusion_matrix.csv")
    aggregate = {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "rank_mae": float(mean_absolute_error(y, predictions)),
        "score_mae": float(mean_absolute_error(source_scores, regression_scores)),
        "score_rmse": float(np.sqrt(np.mean((source_scores - regression_scores) ** 2))),
    }
    payload = {"method": "nested group-stratified 5-fold CV, sparse text regression of ai_impact_score with validation-fitted canonical thresholds",
               "source_score_target_by_raw_label": {"E0": 0.1, "E3": 0.3, "E2": 0.5, "E1": 0.7},
               "canonical_prediction_order": list(LABELS),
               "allowed_input_columns": ["keytask_content"] if args.task_only else ["keytask_content", "jobrole_title"],
               "forbidden_input_columns": ["ai_impact_score", "openai_label", "model_label", "jobrole_id"],
               "aggregate_oof_metrics": aggregate,
               "all_outer_folds_have_zero_shared_jobrole_id": bool((pd.DataFrame(rows)["shared_jobrole_ids"] == 0).all())}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
