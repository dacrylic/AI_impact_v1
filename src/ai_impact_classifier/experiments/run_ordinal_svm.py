"""Test-blind cumulative ordinal SVM for E0 < E3 < E2 < E1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cumulative ordinal sparse SVM.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/ordinal_svm")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _text(frame: pd.DataFrame) -> pd.Series:
    return (
        "ROLE " + frame.jobrole_title.fillna("").astype(str)
        + " OCCUPATION " + frame.ssoc_title.fillna("").astype(str)
        + " SECTOR " + frame.sector_title.fillna("").astype(str)
        + " TASK " + frame.keytask_content.fillna("").astype(str)
    )


def _model(c_value: float, positive_weight: float) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
                    ],
                    transformer_weights={"word": 1.0, "char": 0.65},
                ),
            ),
            ("classifier", LinearSVC(C=c_value, class_weight={0: 1.0, 1: positive_weight}, max_iter=20_000)),
        ]
    )


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _decode(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # Boundary j answers whether a row is above ordinal rank j.
    return (scores >= thresholds[None, :]).sum(axis=1).astype(int)


def _best_thresholds(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(scores.shape[1])
    best = f1_score(y, _decode(scores, thresholds), average="macro")
    for _ in range(4):
        improved = False
        for boundary in range(scores.shape[1]):
            for value in np.linspace(-2.0, 2.0, 81):
                candidate = thresholds.copy(); candidate[boundary] = value
                metric = f1_score(y, _decode(scores, candidate), average="macro")
                if metric > best:
                    thresholds, best, improved = candidate, metric, True
        if not improved:
            break
    return thresholds


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(bundle.frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed:
        raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    texts = [_text(part) for part in (train, val, test)]
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    settings = [(0.25, 1.0), (0.5, 1.0), (0.5, 2.0), (0.75, 2.0), (1.0, 4.0)]
    selected: list[tuple[float, float]] = []
    val_scores = np.zeros((len(val), 3))
    rows: list[dict[str, object]] = []
    for boundary in range(3):
        target_train, target_val = (y_train > boundary).astype(int), (y_val > boundary).astype(int)
        best: tuple[float, float, np.ndarray, float] | None = None
        for c_value, positive_weight in settings:
            model = _model(c_value, positive_weight); model.fit(texts[0], target_train)
            score = model.decision_function(texts[1])
            metric = f1_score(target_val, score >= 0, zero_division=0)
            rows.append({"boundary": boundary, "c": c_value, "positive_weight": positive_weight, "binary_val_f1": metric})
            if best is None or metric > best[3]: best = (c_value, positive_weight, score, metric)
        assert best is not None
        selected.append((best[0], best[1])); val_scores[:, boundary] = best[2]
    thresholds = _best_thresholds(y_val, val_scores)
    val_pred = _decode(val_scores, thresholds)

    dev = pd.concat([train, val]); y_dev = dev.label_rank.to_numpy(); dev_text = _text(dev)
    test_scores = np.zeros((len(test), 3))
    for boundary, (c_value, positive_weight) in enumerate(selected):
        model = _model(c_value, positive_weight); model.fit(dev_text, (y_dev > boundary).astype(int))
        test_scores[:, boundary] = model.decision_function(texts[2])
    test_pred = _decode(test_scores, thresholds)
    pd.DataFrame(rows).to_csv(out / "boundary_validation_search.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"test_blind_selection": True, "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected_boundaries": selected, "validation_thresholds": thresholds.tolist(), "validation_metrics": _metrics(y_val, val_pred), "test_metrics": _metrics(y_test, test_pred), "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": payload["validation_metrics"], "test_metrics": payload["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
