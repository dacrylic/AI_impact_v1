"""Nested group CV that combines frozen MPNet and sparse TF-IDF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.embeddings import encode_texts
from ai_impact_classifier.experiments.run_constrained_oof_stack import _best_bias, _fit_base_features, _metrics, _oof_base_features
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


NAMES = ["E0", "E3", "E12"]
SPECS = [("task_only", 0.25), ("task_only", 0.75), ("title_task", 0.25), ("title_task", 0.75)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested group CV for frozen-MPNet plus sparse stack.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/dense_sparse_oof_cv5")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _dense_classifier() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000, n_jobs=1)),
    ])


def _oof_dense_features(embeddings: np.ndarray, frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    output = np.zeros((len(frame), 3))
    for fit_idx, holdout_idx in splitter.split(embeddings, y, groups):
        model = _dense_classifier(); model.fit(embeddings[fit_idx], y[fit_idx])
        output[holdout_idx] = model.predict_proba(embeddings[holdout_idx])
    return output


def _fit_dense_features(fit_embeddings: np.ndarray, y: np.ndarray, eval_embeddings: np.ndarray) -> np.ndarray:
    model = _dense_classifier(); model.fit(fit_embeddings, y)
    return model.predict_proba(eval_embeddings)


def _oof_features(frame: pd.DataFrame, embeddings: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return np.column_stack([_oof_base_features(frame.reset_index(drop=True), y, seed, SPECS), _oof_dense_features(embeddings, frame.reset_index(drop=True), y, seed)])


def _fit_features(fit: pd.DataFrame, fit_embeddings: np.ndarray, y: np.ndarray, evaluation: pd.DataFrame, eval_embeddings: np.ndarray) -> np.ndarray:
    return np.column_stack([_fit_base_features(fit, y, evaluation, SPECS), _fit_dense_features(fit_embeddings, y, eval_embeddings)])


def _meta(c_value: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1))])


def _select_meta(frame: pd.DataFrame, embeddings: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, np.ndarray, dict[str, float]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(frame, y, frame[GROUP_COLUMN].astype(str)))
    fit, validation = frame.iloc[fit_idx], frame.iloc[val_idx]
    oof = _oof_features(fit, embeddings[fit_idx], y[fit_idx], seed)
    val_features = _fit_features(fit, embeddings[fit_idx], y[fit_idx], validation, embeddings[val_idx])
    selected = None
    for c_value in (0.01, 0.02, 0.05, 0.1):
        model = _meta(c_value); model.fit(oof, y[fit_idx])
        scores = model.predict_proba(val_features); bias = _best_bias(y[val_idx], scores)
        metrics = _metrics(y[val_idx], np.argmax(scores + bias, axis=1))
        if selected is None or metrics["macro_f1"] > selected[2]["macro_f1"]:
            selected = (c_value, bias, metrics)
    assert selected is not None
    return selected


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    texts = _allowed_text(frame, "title_task").tolist()
    embeddings = encode_texts(args.model, texts, batch_size=args.batch_size)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions = np.full(len(frame), -1, dtype=int); rows = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        c_value, bias, inner = _select_meta(development, embeddings[development_idx], y[development_idx], args.seed + fold)
        oof = _oof_features(development, embeddings[development_idx], y[development_idx], args.seed + fold)
        test_features = _fit_features(development, embeddings[development_idx], y[development_idx], test, embeddings[test_idx])
        model = _meta(c_value); model.fit(oof, y[development_idx])
        pred = np.argmax(model.predict_proba(test_features) + bias, axis=1); predictions[test_idx] = pred
        rows.append({"fold": fold, "selected_meta_c": c_value, "bias": json.dumps(bias.tolist()), "inner_validation_macro_f1": inner["macro_f1"], **_metrics(y[test_idx], pred)})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": frame.index, "y_true": y, "y_pred": predictions}).to_csv(out / "oof_predictions.csv", index=False)
    payload = {"method": "nested group CV: frozen MPNet classifier scores + sparse TF-IDF stack", "model": args.model, "allowed_input_columns": ["keytask_content", "jobrole_title"], "aggregate_oof_metrics": _metrics(y, predictions)}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
