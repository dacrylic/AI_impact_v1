from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitBundle:
    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index


def _class_columns(frame: pd.DataFrame, label_column: str) -> List[Hashable]:
    classes = sorted(frame[label_column].dropna().unique().tolist())
    return list(classes)


def _group_table(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str,
    classes: Sequence[Hashable],
) -> pd.DataFrame:
    group_rows = []
    for group_value, group_frame in frame.groupby(group_column, sort=False):
        record = {
            "group": group_value,
            "n_rows": len(group_frame),
        }
        counts = group_frame[label_column].value_counts().to_dict()
        for label in classes:
            record[f"class_{label}"] = int(counts.get(label, 0))
        group_rows.append(record)
    return pd.DataFrame(group_rows)


def _split_cost(
    current_rows: np.ndarray,
    current_class_counts: np.ndarray,
    target_rows: np.ndarray,
    target_class_counts: np.ndarray,
) -> float:
    row_cost = float(np.sum(((current_rows - target_rows) / np.maximum(target_rows, 1.0)) ** 2))
    class_cost = float(
        np.sum(((current_class_counts - target_class_counts) / np.maximum(target_class_counts, 1.0)) ** 2)
    )
    return row_cost + class_cost


def _assignment_signature(
    split_rows: np.ndarray,
    split_class_counts: np.ndarray,
    target_rows: np.ndarray,
    target_class_counts: np.ndarray,
) -> float:
    row_gap = np.sum(np.abs(split_rows - target_rows) / np.maximum(target_rows, 1.0))
    class_gap = np.sum(
        np.abs(split_class_counts - target_class_counts) / np.maximum(target_class_counts, 1.0)
    )
    return float(row_gap + class_gap)


def stratified_group_train_val_test_split(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    n_restarts: int = 25,
) -> SplitBundle:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"split sizes must sum to 1.0, got {total:.4f}")

    classes = _class_columns(frame, label_column)
    group_table = _group_table(frame, group_column, label_column, classes)
    if group_table.empty:
        raise ValueError("No groups available for splitting")

    desired_rows = np.array([train_size, val_size, test_size], dtype=float) * len(frame)
    desired_class_counts = np.vstack(
        [
            np.array([train_size, val_size, test_size], dtype=float) * frame[label_column].value_counts().get(c, 0)
            for c in classes
        ]
    ).T

    group_records = group_table.to_dict("records")
    rng = np.random.default_rng(random_state)
    best_assignment: Dict[str, int] | None = None
    best_cost = float("inf")

    for _ in range(n_restarts):
        order = sorted(
            group_records,
            key=lambda row: (
                -sum(row[f"class_{c}"] > 0 for c in classes),
                -max(row[f"class_{c}"] for c in classes),
                -row["n_rows"],
                rng.random(),
            ),
        )

        split_rows = np.zeros(3, dtype=float)
        split_class_counts = np.zeros((3, len(classes)), dtype=float)
        assignment: Dict[str, int] = {}

        for row in order:
            best_split = None
            best_split_cost = float("inf")
            group_class_counts = np.array([row[f"class_{c}"] for c in classes], dtype=float)
            for split_idx in range(3):
                trial_rows = split_rows.copy()
                trial_class_counts = split_class_counts.copy()
                trial_rows[split_idx] += row["n_rows"]
                trial_class_counts[split_idx] += group_class_counts
                cost = _split_cost(trial_rows, trial_class_counts, desired_rows, desired_class_counts)
                if cost < best_split_cost:
                    best_split_cost = cost
                    best_split = split_idx
            assignment[str(row["group"])] = int(best_split)
            split_rows[best_split] += row["n_rows"]
            split_class_counts[best_split] += group_class_counts

        total_cost = _split_cost(split_rows, split_class_counts, desired_rows, desired_class_counts)
        total_cost += 0.15 * _assignment_signature(
            split_rows, split_class_counts, desired_rows, desired_class_counts
        )
        if total_cost < best_cost:
            best_cost = total_cost
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Failed to build a stratified group split")

    groups = frame[group_column].astype(str)
    split_mask = groups.map(best_assignment)
    if split_mask.isna().any():
        missing = frame.loc[split_mask.isna(), group_column].unique().tolist()
        raise RuntimeError(f"Split assignment missing groups: {missing[:5]}")

    train_idx = frame.index[split_mask == 0]
    val_idx = frame.index[split_mask == 1]
    test_idx = frame.index[split_mask == 2]
    return SplitBundle(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
