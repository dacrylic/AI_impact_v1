"""One strict 80:20 role-and-task-disjoint title+task holdout experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ai_impact_classifier.checks import check_role_task_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _base_model
from ai_impact_classifier.experiments.run_role_task_disjoint_cv import CLASS_NAMES, connected_role_task_components
from ai_impact_classifier.production import _combine_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict 80:20 role/task-disjoint title+task holdout.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/role_task_disjoint_holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=500)
    return parser.parse_args()


def select_test_components(frame: pd.DataFrame, test_size: float, seed: int, restarts: int) -> set[str]:
    """Find a near-stratified component subset without splitting any component."""
    labels = sorted(frame["label_rank"].unique())
    table = frame.groupby("role_task_component")["label_rank"].value_counts().unstack(fill_value=0)
    table = table.reindex(columns=labels, fill_value=0)
    sizes = table.sum(axis=1).to_numpy(dtype=float)
    counts = table.to_numpy(dtype=float)
    target = np.r_[len(frame) * test_size, frame["label_rank"].value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float) * test_size]
    # Components above the target test size cannot contribute to an 80:20 holdout.
    eligible = sizes <= target[0]
    rng = np.random.default_rng(seed)
    best_selection: np.ndarray | None = None
    best_cost = float("inf")

    def cost(total: np.ndarray) -> float:
        relative = (total - target) / np.maximum(target, 1.0)
        return float(np.dot(relative, relative))

    for _ in range(restarts):
        order = np.flatnonzero(eligible)
        rng.shuffle(order)
        order = order[np.argsort(-sizes[order], kind="stable")]
        selected = np.zeros(len(table), dtype=bool)
        total = np.zeros(len(labels) + 1)
        for index in order:
            candidate = total + np.r_[sizes[index], counts[index]]
            if cost(candidate) < cost(total) or rng.random() < 0.015:
                selected[index] = True
                total = candidate
        # Greedy local removal/addition improves the random-start partition.
        improved = True
        while improved:
            improved = False
            for index in rng.permutation(len(table)):
                candidate = total + (-1 if selected[index] else 1) * np.r_[sizes[index], counts[index]]
                if candidate[0] <= 0 or candidate[0] > target[0] * 1.10:
                    continue
                if cost(candidate) < cost(total):
                    selected[index] = not selected[index]
                    total = candidate
                    improved = True
        if cost(total) < best_cost:
            best_cost, best_selection = cost(total), selected.copy()
    if best_selection is None:
        raise RuntimeError("Unable to create a component-disjoint holdout.")
    return set(table.index[best_selection].astype(str))


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    frame["role_task_component"] = connected_role_task_components(frame)
    test_components = select_test_components(frame, test_size=0.20, seed=args.seed, restarts=args.restarts)
    test_mask = frame["role_task_component"].astype(str).isin(test_components)
    train_idx, test_idx = frame.index[~test_mask], frame.index[test_mask]
    leakage = check_role_task_split_leakage(frame, {"train": train_idx, "test": test_idx}, GROUP_COLUMN, "keytask_content")
    if not leakage.passed:
        raise RuntimeError("Hard role/task leakage: " + "; ".join(leakage.messages))
    y_train, y_test = frame.loc[train_idx, "label_rank"].to_numpy(), frame.loc[test_idx, "label_rank"].to_numpy()
    model = _base_model(0.5)
    model.fit(_combine_text(frame.loc[train_idx]), y_train)
    prediction = model.predict(_combine_text(frame.loc[test_idx]))
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, labels=range(3), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, prediction, labels=range(3), average="weighted", zero_division=0)),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": test_idx, "y_true": y_test, "y_pred": prediction}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, prediction, labels=range(3)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "test_confusion_matrix.csv")
    component_sizes = frame["role_task_component"].value_counts()
    largest_component_id = str(component_sizes.index[0])
    largest_component = int(component_sizes.iloc[0])
    metadata = {
        "method": "single component-disjoint 80:20 holdout",
        "input_columns": ["jobrole_title", "keytask_content"],
        "hard_split_contract": "No jobrole_id or normalized keytask_content occurs in both train and test.",
        "zero_role_task_leakage": leakage.passed,
        "rows": {"train": int(len(train_idx)), "test": int(len(test_idx)), "test_share": float(len(test_idx) / len(frame))},
        "component_counts": {"total": int(frame["role_task_component"].nunique()), "test": int(len(test_components))},
        "largest_component_rows": largest_component,
        "largest_component_in_train": largest_component_id not in test_components,
        "metrics": metrics,
        "classification_report": classification_report(y_test, prediction, labels=range(3), target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "interpretation_warning": "The largest connected component exceeds the 20% test target and is necessarily retained in training. This is a valid zero-leakage holdout, but not a representative test of that giant component.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
