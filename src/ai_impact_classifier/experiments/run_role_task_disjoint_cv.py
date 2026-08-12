"""Title-plus-task CV with hard role and normalized-task disjointness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ai_impact_classifier.checks import check_role_task_split_leakage, normalized_text
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _base_model
from ai_impact_classifier.production import _combine_text


CLASS_NAMES = ("E0", "E1", "E23")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run role-and-task-disjoint title+task CV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/role_task_disjoint_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def connected_role_task_components(frame: pd.DataFrame) -> pd.Series:
    """Return bipartite connected components for role IDs and task text."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    tasks = normalized_text(frame["keytask_content"])
    for role, task in zip(frame[GROUP_COLUMN].astype(str), tasks):
        union(f"role:{role}", f"task:{task}")
    return pd.Series([find(f"task:{task}") for task in tasks], index=frame.index, name="role_task_component")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=range(3), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=range(3), average="weighted", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    frame["role_task_component"] = connected_role_task_components(frame)
    y = frame["label_rank"].to_numpy()
    components = frame["role_task_component"].to_numpy()
    prediction = np.full(len(frame), -1, dtype=int)
    folds: list[dict[str, float | int | bool]] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y, components), start=1):
        report = check_role_task_split_leakage(
            frame,
            {"train": frame.index[train_idx], "test": frame.index[test_idx]},
            GROUP_COLUMN,
            "keytask_content",
        )
        if not report.passed:
            raise RuntimeError("Hard role/task leakage: " + "; ".join(report.messages))
        model = _base_model(0.5)
        model.fit(_combine_text(frame.iloc[train_idx]), y[train_idx])
        fold_prediction = model.predict(_combine_text(frame.iloc[test_idx]))
        prediction[test_idx] = fold_prediction
        folds.append({
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_share": float(len(test_idx) / len(frame)),
            "test_components": int(frame.iloc[test_idx]["role_task_component"].nunique()),
            "zero_role_task_leakage": report.passed,
            **metric_row(y[test_idx], fold_prediction),
        })

    if (prediction < 0).any():
        raise RuntimeError("Incomplete OOF predictions.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame({"y_true": y, "y_pred": prediction, "role_task_component": components}).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y, prediction, labels=range(3)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "oof_confusion_matrix.csv")
    metadata = {
        "method": "five-fold StratifiedGroupKFold over role-task connected components",
        "input_columns": ["jobrole_title", "keytask_content"],
        "hard_split_contract": "No jobrole_id or normalized keytask_content occurs in both train and test.",
        "component_count": int(frame["role_task_component"].nunique()),
        "largest_component_rows": int(frame["role_task_component"].value_counts().max()),
        "largest_component_share": float(frame["role_task_component"].value_counts().max() / len(frame)),
        "aggregate_oof_metrics": metric_row(y, prediction),
        "classification_report": classification_report(y, prediction, labels=range(3), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "fold_size_warning": "The connected components are highly imbalanced; interpret aggregate OOF metrics cautiously.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "folds": folds}, indent=2))


if __name__ == "__main__":
    main()
