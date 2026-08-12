"""Strict task-only evaluation with exact task duplicates removed before CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from ai_impact_classifier.data import build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _base_model


CLASS_NAMES = ("E0", "E1", "E23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict task-deduplicated task-only five-fold CV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/task_only_deduplicated_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized_task_key(tasks: pd.Series) -> pd.Series:
    return tasks.fillna("").astype(str).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def build_strict_task_dataset(path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = build_bundle(path).frame.copy()
    frame["task_key"] = normalized_task_key(frame["keytask_content"])
    label_counts = frame.groupby(["task_key", "model_label"]).size().unstack(fill_value=0)
    is_unambiguous = label_counts.gt(0).sum(axis=1).eq(1)
    unambiguous_keys = set(label_counts.index[is_unambiguous])
    # One source row per task is retained only after proving the task has one
    # canonical label. Contradictory task labels have no task-only ground truth.
    strict = frame[frame["task_key"].isin(unambiguous_keys)].drop_duplicates("task_key", keep="first").reset_index(drop=True)
    audit = {
        "source_rows": int(len(frame)),
        "unique_normalized_tasks": int(frame["task_key"].nunique()),
        "duplicate_rows_removed": int(len(frame) - frame["task_key"].nunique()),
        "conflicting_label_tasks_excluded": int((~is_unambiguous).sum()),
        "rows_in_conflicting_label_tasks_excluded": int(frame["task_key"].isin(set(label_counts.index[~is_unambiguous])).sum()),
        "strict_unique_task_rows": int(len(strict)),
    }
    return strict, audit


def main() -> None:
    args = parse_args()
    frame, audit = build_strict_task_dataset(args.input)
    y = frame["label_rank"].to_numpy()
    prediction = np.full(len(frame), -1, dtype=int)
    folds: list[dict[str, float | int]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y), start=1):
        model = _base_model(0.5)
        model.fit("TASK " + frame.iloc[train_idx]["keytask_content"], y[train_idx])
        fold_prediction = model.predict("TASK " + frame.iloc[test_idx]["keytask_content"])
        prediction[test_idx] = fold_prediction
        folds.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "accuracy": float(accuracy_score(y[test_idx], fold_prediction)),
            "macro_f1": float(f1_score(y[test_idx], fold_prediction, average="macro")),
            "weighted_f1": float(f1_score(y[test_idx], fold_prediction, average="weighted")),
        })

    if (prediction < 0).any():
        raise RuntimeError("Incomplete OOF predictions.")
    if frame["task_key"].duplicated().any():
        raise RuntimeError("Task deduplication failed.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "weighted_f1": float(f1_score(y, prediction, average="weighted")),
    }
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame({"task_key": frame["task_key"], "y_true": y, "y_pred": prediction}).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y, prediction, labels=range(3)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "oof_confusion_matrix.csv")
    report = classification_report(y, prediction, labels=range(3), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metadata = {
        "method": "five-fold StratifiedKFold after exact normalized-task deduplication",
        "input_columns": ["keytask_content"],
        "duplicate_policy": "Exclude task texts with conflicting E0/E1/E23 labels; retain one row for each remaining exact normalized task.",
        "audit": audit,
        "aggregate_oof_metrics": aggregate,
        "classification_report": report,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
