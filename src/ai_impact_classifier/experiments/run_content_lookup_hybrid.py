"""Sparse specialist plus explicitly reported exact task-text lookup overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_specialist_hybrid import Candidate, _best_bias, _make_model, _text
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sparse specialist with an exact raw-task lookup overlay.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/content_lookup_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _lookup(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    counts = train.groupby(["keytask_content", "label_rank"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(4), fill_value=0)
    total = counts.sum(axis=1)
    label = counts.idxmax(axis=1)
    confidence = counts.max(axis=1) / total
    table = pd.DataFrame({"label": label.astype(int), "support": total.astype(int), "confidence": confidence})
    return target[["keytask_content"]].join(table, on="keytask_content")


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(bundle.frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    selected = [Candidate("E0", "char_word", 0.25, 2), Candidate("E3", "char_word", 1, 16), Candidate("E2", "char_word", 0.25, 2), Candidate("E1", "char_word_long", 0.5, 4)]
    thresholds = np.array([0.05, -0.3, -0.1, -0.1]); val_scores = np.zeros((len(val), 4))
    for cls, candidate in enumerate(selected):
        model = _make_model(candidate); model.fit(_text(train), (y_train == cls).astype(int))
        val_scores[:, cls] = model.decision_function(_text(val)) - thresholds[cls]
    bias = _best_bias(y_val, val_scores); base_val = np.argmax(val_scores + bias, axis=1)
    looked_val = _lookup(train, val).reset_index(drop=True)
    grid_rows, best = [], None
    for support in [1, 2, 3, 5]:
        for confidence in [0.6, 0.7, 0.8, 0.9, 0.99, 1.0]:
            for nonzero_only in [False, True]:
                pred = base_val.copy()
                mask = looked_val.label.notna() & (looked_val.support >= support) & (looked_val.confidence >= confidence)
                if nonzero_only: mask &= looked_val.label.to_numpy() != 0
                pred[mask] = looked_val.loc[mask, "label"].astype(int)
                metric = _metrics(y_val, pred)
                row = {"min_support": support, "min_confidence": confidence, "nonzero_only": nonzero_only, "overrides": int(mask.sum()), **metric}
                grid_rows.append(row)
                if best is None or metric["macro_f1"] > best[3]["macro_f1"]: best = (support, confidence, nonzero_only, metric)
    assert best is not None
    support, confidence, nonzero_only, val_metrics = best
    development = pd.concat([train, val]); y_dev = development.label_rank.to_numpy(); test_scores = np.zeros((len(test), 4))
    for cls, candidate in enumerate(selected):
        model = _make_model(candidate); model.fit(_text(development), (y_dev == cls).astype(int))
        test_scores[:, cls] = model.decision_function(_text(test)) - thresholds[cls]
    test_pred = np.argmax(test_scores + bias, axis=1)
    looked_test = _lookup(development, test).reset_index(drop=True)
    mask = looked_test.label.notna() & (looked_test.support >= support) & (looked_test.confidence >= confidence)
    if nonzero_only: mask &= looked_test.label.to_numpy() != 0
    test_pred[mask] = looked_test.loc[mask, "label"].astype(int)
    test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame(grid_rows).sort_values("macro_f1", ascending=False).to_csv(out / "lookup_validation_grid.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred, "lookup_override": mask.to_numpy()}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"test_blind_selection": True, "method_warning": "This method uses repeated exact keytask text across group-disjoint splits. It is raw-text-only and group leakage checks pass, but recurrence is reported as a soft leakage risk.", "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected_lookup": {"min_support": support, "min_confidence": confidence, "nonzero_only": nonzero_only, "test_overrides": int(mask.sum())}, "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics, "test_overrides": int(mask.sum())}, indent=2))


if __name__ == "__main__": main()
