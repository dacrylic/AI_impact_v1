"""Leak-safe model comparison for the E0/E1/E23 target.

E23 combines raw E2 and E3.  Regressors predict the deliberately documented
canonical score mapping E0=0.1, E23=0.4, E1=0.7, then use thresholds chosen
inside each outer-development partition to recover the official classes.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC, LinearSVR

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle


# Class rank follows data.py: E0=0, E1=1, E23=2.  Score order is different.
CLASS_NAMES = ("E0", "E1", "E23")
SCORE_BY_CLASS = np.array([0.1, 0.7, 0.4])
SCORE_ORDER_CLASSES = np.array([0, 2, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested group-CV method zoo for E0/E1/E23.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/e0_e1_e23_method_zoo_cv5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings", action="store_true", help="Also test frozen MiniLM and MPNet embeddings.")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser.parse_args()


def _text(frame: pd.DataFrame) -> pd.Series:
    return "ROLE " + frame.jobrole_title.fillna("").astype(str) + " TASK " + frame.keytask_content.fillna("").astype(str)


def _sparse_features(fit_text: pd.Series, eval_text: pd.Series) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=300_000, sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=250_000, sublinear_tf=True)
    fit = sparse.hstack([word.fit_transform(fit_text), 0.65 * char.fit_transform(fit_text)], format="csr")
    evaluation = sparse.hstack([word.transform(eval_text), 0.65 * char.transform(eval_text)], format="csr")
    return fit, evaluation


def _decode(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    bucket = np.digitize(scores, thresholds, right=False)
    return SCORE_ORDER_CLASSES[bucket]


def _best_thresholds(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    thresholds = np.array([np.quantile(scores, 0.5), np.quantile(scores, 0.85)])
    best = f1_score(y, _decode(scores, thresholds), average="macro", zero_division=0)
    for _ in range(3):
        improved = False
        for index in range(2):
            for value in values:
                candidate = thresholds.copy(); candidate[index] = value
                if candidate[0] >= candidate[1]:
                    continue
                metric = f1_score(y, _decode(scores, candidate), average="macro", zero_division=0)
                if metric > best:
                    thresholds, best, improved = candidate, metric, True
        if not improved:
            break
    return thresholds


def _metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    values = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    if scores is not None:
        target = SCORE_BY_CLASS[y]
        values["score_mae"] = float(mean_absolute_error(target, scores))
        values["score_rmse"] = float(np.sqrt(np.mean((target - scores) ** 2)))
    return values


def _fit_predict(method: str, x_fit: np.ndarray | sparse.csr_matrix, y_fit: np.ndarray, x_eval: np.ndarray | sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "linear_svc":
        model = LinearSVC(C=0.5, class_weight="balanced", max_iter=20_000)
        model.fit(x_fit, y_fit)
        return model.predict(x_eval), None
    target = SCORE_BY_CLASS[y_fit]
    if method == "ridge_score":
        weights = np.where(y_fit == 1, 1.5, 1.0)
        center = float(np.average(target, weights=weights))
        model = Ridge(alpha=20.0, solver="lsqr", fit_intercept=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_fit, target - center, sample_weight=weights)
            scores = model.predict(x_eval) + center
    elif method == "linear_svr_score":
        model = LinearSVR(C=0.08, epsilon=0.05, max_iter=20_000, random_state=42)
        model.fit(x_fit, target, sample_weight=np.where(y_fit == 1, 1.5, 1.0))
        scores = model.predict(x_eval)
    else:
        raise ValueError(method)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{method} produced non-finite score predictions")
    return np.empty(x_eval.shape[0], dtype=int), np.clip(scores, -0.1, 0.9)


def _select_and_predict(method: str, x_development: np.ndarray | sparse.csr_matrix, y_development: np.ndarray, groups_development: np.ndarray, x_test: np.ndarray | sparse.csr_matrix, seed: int) -> tuple[np.ndarray, dict[str, object], np.ndarray | None]:
    if method == "linear_svc":
        pred, _ = _fit_predict(method, x_development, y_development, x_test)
        return pred, {}, None
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(inner.split(x_development, y_development, groups_development))
    _, val_scores = _fit_predict(method, x_development[fit_idx], y_development[fit_idx], x_development[val_idx])
    assert val_scores is not None
    thresholds = _best_thresholds(y_development[val_idx], val_scores)
    _, test_scores = _fit_predict(method, x_development, y_development, x_test)
    assert test_scores is not None
    return _decode(test_scores, thresholds), {"thresholds": thresholds.tolist()}, test_scores


def _evaluate(method: str, frame: pd.DataFrame, y: np.ndarray, features: np.ndarray | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.full(len(frame), -1, dtype=int); scores = np.full(len(frame), np.nan)
    rows: list[dict[str, object]] = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        leakage = check_split_leakage(frame, {"development": development.index, "outer_test": test.index}, GROUP_COLUMN, "target_text")
        if not leakage.passed:
            raise RuntimeError("Outer-fold leakage: " + "; ".join(leakage.messages))
        if features is None:
            x_development, x_test = _sparse_features(_text(development), _text(test))
        else:
            x_development, x_test = features[development_idx], features[test_idx]
        fold_pred, selected, fold_scores = _select_and_predict(method, x_development, y[development_idx], groups[development_idx], x_test, seed + fold)
        pred[test_idx] = fold_pred
        if fold_scores is not None:
            scores[test_idx] = fold_scores
        rows.append({"fold": fold, "shared_jobrole_ids": len(set(groups[development_idx]) & set(groups[test_idx])),
                     "duplicate_target_text_warning": leakage.warnings[0] if leakage.warnings else "", **selected,
                     **_metrics(y[test_idx], fold_pred, fold_scores)})
        print(json.dumps({"method": method, **rows[-1]}), flush=True)
    if (pred < 0).any():
        raise RuntimeError("Missing OOF classification prediction")
    predictions = pd.DataFrame({"row_index": np.arange(len(frame)), "y_true": y, "y_pred": pred, "score_prediction": scores})
    aggregate = _metrics(y, pred, None if np.isnan(scores).all() else scores)
    return pd.DataFrame(rows), predictions, aggregate


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy()
    methods: list[tuple[str, np.ndarray | None, str]] = [
        ("sparse_linear_svc", None, "linear_svc"),
        ("sparse_ridge_score", None, "ridge_score"),
        ("sparse_linear_svr_score", None, "linear_svr_score"),
    ]
    if args.embeddings:
        from ai_impact_classifier.embeddings import default_device, encode_texts
        for model in ("sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"):
            cache = Path(args.output_dir) / (model.rsplit("/", 1)[-1] + ".npy")
            cache.parent.mkdir(parents=True, exist_ok=True)
            vectors = np.load(cache) if cache.exists() else encode_texts(model, _text(frame).tolist(), batch_size=args.embedding_batch_size, device=default_device())
            if not cache.exists():
                np.save(cache, vectors)
            vectors = np.asarray(vectors, dtype=np.float64)
            methods.extend([(model.rsplit("/", 1)[-1] + "_linear_svc", vectors, "linear_svc"),
                            (model.rsplit("/", 1)[-1] + "_ridge_score", vectors, "ridge_score")])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    leaderboard: list[dict[str, object]] = []
    for name, features, method in methods:
        folds, predictions, aggregate = _evaluate(method, frame, y, features, args.seed)
        folds.to_csv(out / f"{name}_fold_metrics.csv", index=False)
        predictions.to_csv(out / f"{name}_oof_predictions.csv", index=False)
        matrix = confusion_matrix(y, predictions.y_pred, labels=range(3))
        pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / f"{name}_oof_confusion_matrix.csv")
        leaderboard.append({"method": name, **aggregate})
    report = pd.DataFrame(leaderboard).sort_values("macro_f1", ascending=False)
    report.to_csv(out / "leaderboard.csv", index=False)
    payload = {"target_definition": "E0 vs E1 vs E23, where E23 merges raw E2 and E3",
               "score_mapping_for_regression": {"E0": 0.1, "E23": 0.4, "E1": 0.7},
               "allowed_input_columns": ["keytask_content", "jobrole_title"],
               "outer_split": "StratifiedGroupKFold(n_splits=5) grouped by jobrole_id",
               "inner_threshold_selection": "thresholds fitted only inside each outer development partition",
               "results": leaderboard}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
