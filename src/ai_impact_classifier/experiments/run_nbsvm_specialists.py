"""Test-blind NB-SVM one-vs-rest specialists for sparse task text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-blind NB-SVM class specialists.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/nbsvm_specialists")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _text(frame: pd.DataFrame) -> pd.Series:
    return (
        "ROLE_" + frame["jobrole_title"].fillna("").astype(str).str.replace(r"\s+", "_", regex=True)
        + " OCC_" + frame["ssoc_title"].fillna("").astype(str).str.replace(r"\s+", "_", regex=True)
        + " SECTOR_" + frame["sector_title"].fillna("").astype(str).str.replace(r"\s+", "_", regex=True)
        + " " + frame["keytask_content"].fillna("").astype(str)
    )


def _binary_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.0, -1.0
    for threshold in np.linspace(-3.0, 3.0, 121):
        value = f1_score(y, scores >= threshold, zero_division=0)
        if value > best_f1:
            best_t, best_f1 = float(threshold), float(value)
    return best_t, best_f1


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    bias = np.zeros(scores.shape[1])
    best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        changed = False
        for cls in range(scores.shape[1]):
            for value in np.linspace(-1, 1, 41):
                candidate = bias.copy(); candidate[cls] = value
                metric = f1_score(y, np.argmax(scores + candidate, axis=1), average="macro")
                if metric > best:
                    bias, best, changed = candidate, metric, True
        if not changed:
            break
    return bias


def _nb_ratio(x: csr_matrix, y: np.ndarray) -> np.ndarray:
    positive = np.asarray(x[y == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(x[y == 0].sum(axis=0)).ravel() + 1.0
    return np.log(positive / positive.sum()) - np.log(negative / negative.sum())


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(bundle.frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed:
        raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True, max_features=450_000, strip_accents="unicode")
    x_train = vectorizer.fit_transform(_text(train)); x_val = vectorizer.transform(_text(val))
    settings = [(0.1, 1.0), (0.25, 2.0), (0.5, 4.0), (1.0, 8.0), (1.0, 16.0)]
    selected: list[tuple[float, float, float]] = []
    val_scores = np.zeros((len(val), len(bundle.label_order)))
    rows: list[dict[str, object]] = []
    for cls, label in enumerate(bundle.label_order):
        binary_train, binary_val = (y_train == cls).astype(int), (y_val == cls).astype(int)
        ratio = _nb_ratio(x_train, binary_train)
        best: tuple[float, float, float, np.ndarray, float] | None = None
        for c_value, weight in settings:
            model = LinearSVC(C=c_value, class_weight={0: 1, 1: weight}, max_iter=20_000)
            model.fit(x_train.multiply(ratio), binary_train)
            scores = model.decision_function(x_val.multiply(ratio))
            threshold, value = _binary_threshold(binary_val, scores)
            rows.append({"label": label, "c": c_value, "positive_weight": weight, "threshold": threshold, "binary_val_f1": value})
            if best is None or value > best[4]: best = (c_value, weight, threshold, scores, value)
        assert best is not None
        c_value, weight, threshold, scores, _ = best
        selected.append((c_value, weight, threshold)); val_scores[:, cls] = scores - threshold
    bias = _bias(y_val, val_scores)
    val_pred = np.argmax(val_scores + bias, axis=1)

    dev = pd.concat([train, val]); y_dev = dev.label_rank.to_numpy()
    x_dev = vectorizer.fit_transform(_text(dev)); x_test = vectorizer.transform(_text(test))
    test_scores = np.zeros((len(test), len(bundle.label_order)))
    for cls, (c_value, weight, threshold) in enumerate(selected):
        binary = (y_dev == cls).astype(int); ratio = _nb_ratio(x_dev, binary)
        model = LinearSVC(C=c_value, class_weight={0: 1, 1: weight}, max_iter=20_000)
        model.fit(x_dev.multiply(ratio), binary)
        test_scores[:, cls] = model.decision_function(x_test.multiply(ratio)) - threshold
    test_pred = np.argmax(test_scores + bias, axis=1)
    pd.DataFrame(rows).to_csv(out / "binary_validation_search.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"test_blind_selection": True, "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected": dict(zip(bundle.label_order, selected)), "bias": bias.tolist(), "validation_metrics": _metrics(y_val, val_pred), "test_metrics": _metrics(y_test, test_pred), "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": payload["validation_metrics"], "test_metrics": payload["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
