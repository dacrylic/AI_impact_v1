"""Create the locked role-held-out split for new-target model development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ai_impact_classifier.experiments.run_new_target_sparse_cv import clean_text, load_frame, representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/new_target_locked_split")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fold", type=int, default=1, help="One-based fold to reserve as the locked test set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, audit = load_frame(args.input, strict_task_deduplicated=False, task_conflict_policy="drop")
    frame["model_input_key"] = clean_text(representation(frame, "title_structured_task")).str.lower()
    label_count = frame.groupby("model_input_key")["target"].nunique()
    conflicting = label_count[label_count > 1].index
    frame = (
        frame.loc[~frame["model_input_key"].isin(conflicting)]
        .sort_values(["model_input_key", "jobrole_id"], kind="stable")
        .drop_duplicates("model_input_key")
        .reset_index(drop=True)
    )
    groups = frame["jobrole_id"].astype(str).to_numpy()
    y = frame["target"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    outer_splits = list(outer.split(frame, y, groups))
    if not 1 <= args.test_fold <= len(outer_splits):
        raise ValueError("test-fold must be between 1 and 5")
    development_idx, test_idx = outer_splits[args.test_fold - 1]
    development = frame.iloc[development_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 1)
    train_rel, validation_rel = next(
        inner.split(development, development["target"].to_numpy(), development["jobrole_id"].astype(str).to_numpy())
    )
    train_idx, validation_idx = development_idx[train_rel], development_idx[validation_rel]
    assignments = pd.Series("", index=frame.index, dtype=object)
    assignments.iloc[train_idx] = "train"
    assignments.iloc[validation_idx] = "validation"
    assignments.iloc[test_idx] = "test"
    split_sets = {name: set(frame.loc[assignments.eq(name), "jobrole_id"].astype(str)) for name in assignments.unique()}
    input_sets = {name: set(frame.loc[assignments.eq(name), "model_input_key"]) for name in assignments.unique()}
    role_overlap = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    input_overlap = {
        f"{left}_{right}": len(input_sets[left] & input_sets[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(role_overlap.values()) or any(input_overlap.values()):
        raise RuntimeError("Locked split leakage check failed")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output = frame[["jobrole_id", "jobrole_title", "target", "task", "action", "object", "purpose"]].copy()
    output.insert(0, "row_index", frame.index)
    output["split"] = assignments
    output.to_csv(out / "split_assignments.csv", index=False)
    metadata = {
        "seed": args.seed,
        "test_fold": args.test_fold,
        "rows": len(frame),
        "split_counts": assignments.value_counts().to_dict(),
        "split_target_counts": output.groupby(["split", "target"]).size().unstack(fill_value=0).to_dict(),
        "role_overlap": role_overlap,
        "full_input_overlap": input_overlap,
        "input_representation": "title_structured_task",
        "deduplication": "drop conflicting full input labels; retain lexicographically smallest jobrole_id for same-label duplicates",
        "source_audit": audit,
    }
    (out / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
