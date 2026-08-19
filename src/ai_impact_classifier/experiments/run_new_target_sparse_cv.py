"""Leakage-audited CPU sparse benchmark for the decomposed new target dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


LABELS = ("E0", "E1", "E23")
TARGET_MAP = {"E0": "E0", "E1": "E1", "E2": "E23", "E3": "E23"}
REQUIRED_COLUMNS = {"jobrole_id", "jobrole_title", "action", "object", "purpose", "task", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to tasks_reasoning_scores.csv")
    parser.add_argument("--output-dir", default="results/new_target_sparse_cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, help="Run one 1-based outer fold; useful for resumable CPU runs.")
    parser.add_argument(
        "--strict-task-deduplicated",
        action="store_true",
        help="For task-only evaluation, exclude conflicting task labels and retain one row per normalized task.",
    )
    parser.add_argument(
        "--task-conflict-policy",
        choices=("drop", "mode"),
        default="drop",
        help="Under strict task deduplication, drop every conflict or retain only clear modal-label conflicts.",
    )
    parser.add_argument(
        "--strict-model-input-deduplicated",
        action="store_true",
        help="Retain one row per full representation string and fail on shared full inputs across folds.",
    )
    parser.add_argument("--max-word-features", type=int, default=200_000)
    parser.add_argument("--max-char-features", type=int, default=200_000)
    parser.add_argument(
        "--representations",
        default="task,structured_task,title_structured_task",
        help="Comma-separated subset of task, structured_task, title_task, title_structured_task.",
    )
    parser.add_argument(
        "--class-weights",
        default="unweighted,balanced",
        help="Comma-separated subset of unweighted, balanced.",
    )
    return parser.parse_args()


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_frame(
    path: str | Path, strict_task_deduplicated: bool, task_conflict_policy: str
) -> tuple[pd.DataFrame, dict[str, int | bool | str]]:
    frame = pd.read_csv(path, low_memory=False).copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["label", "jobrole_id"]).reset_index(drop=True)
    frame["target"] = frame["label"].map(TARGET_MAP)
    if frame["target"].isna().any():
        unknown = sorted(frame.loc[frame["target"].isna(), "label"].unique())
        raise ValueError(f"Unsupported source labels: {unknown}")
    for column in ("jobrole_title", "action", "object", "purpose", "task"):
        frame[column] = clean_text(frame[column])
    frame["task_key"] = frame["task"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    audit: dict[str, int | bool | str] = {
        "strict_task_deduplicated": strict_task_deduplicated,
        "task_conflict_policy": task_conflict_policy,
        "input_rows": len(frame),
        "input_unique_task_texts": int(frame["task_key"].nunique()),
        "excluded_conflicting_task_texts": 0,
        "excluded_rows_with_conflicting_task_text": 0,
        "excluded_tied_task_texts": 0,
        "retained_modal_conflicting_task_texts": 0,
    }
    if strict_task_deduplicated:
        label_counts = frame.groupby(["task_key", "target"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
        label_count = (label_counts > 0).sum(axis=1)
        conflicting = label_count[label_count > 1].index
        audit["excluded_conflicting_task_texts"] = int(len(conflicting))
        audit["excluded_rows_with_conflicting_task_text"] = int(frame["task_key"].isin(conflicting).sum())
        if task_conflict_policy == "drop":
            retained = frame.loc[~frame["task_key"].isin(conflicting)]
        else:
            max_counts = label_counts.max(axis=1)
            tied = label_counts.eq(max_counts, axis=0).sum(axis=1) > 1
            clear_modes = label_counts.loc[~tied].idxmax(axis=1)
            audit["excluded_tied_task_texts"] = int((tied & (label_count > 1)).sum())
            modal_conflicts = label_counts.index[(label_count > 1) & ~tied]
            audit["retained_modal_conflicting_task_texts"] = int(len(modal_conflicts))
            retained = pd.concat(
                [
                    frame.loc[~frame["task_key"].isin(conflicting)],
                    frame.loc[frame["task_key"].isin(modal_conflicts)].merge(
                        clear_modes.rename("modal_target"), left_on="task_key", right_index=True
                    ).query("target == modal_target").drop(columns="modal_target"),
                ],
                ignore_index=True,
            )
        frame = retained.sort_values(["task_key", "jobrole_id"], kind="stable").drop_duplicates("task_key").reset_index(drop=True)
    audit["model_rows"] = len(frame)
    audit["model_unique_task_texts"] = int(frame["task_key"].nunique())
    return frame, audit


def representation(frame: pd.DataFrame, name: str) -> pd.Series:
    task = "TASK: " + frame["task"]
    if name == "task":
        return task
    if name == "title_task":
        return "ROLE: " + frame["jobrole_title"] + " " + task
    structured = (
        task
        + " ACTION: " + frame["action"]
        + " OBJECT: " + frame["object"]
        + " PURPOSE: " + frame["purpose"]
    )
    if name == "structured_task":
        return structured
    if name == "title_structured_task":
        return "ROLE: " + frame["jobrole_title"] + " " + structured
    raise ValueError(f"Unknown representation: {name}")


def features(
    train_text: pd.Series,
    evaluation_text: pd.Series,
    max_word_features: int,
    max_char_features: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_word_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = sparse.hstack(
        [word.fit_transform(train_text), 0.65 * char.fit_transform(train_text)], format="csr"
    )
    x_evaluation = sparse.hstack(
        [word.transform(evaluation_text), 0.65 * char.transform(evaluation_text)], format="csr"
    )
    return x_train, x_evaluation


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rank = {label: index for index, label in enumerate(LABELS)}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score([rank[label] for label in y_true], [rank[label] for label in y_pred], weights="quadratic")
        ),
    }


def main() -> None:
    args = parse_args()
    available_representations = ("task", "structured_task", "title_task", "title_structured_task")
    requested_representations = tuple(name.strip() for name in args.representations.split(",") if name.strip())
    unknown_representations = set(requested_representations) - set(available_representations)
    if unknown_representations:
        raise ValueError(f"Unknown representations: {sorted(unknown_representations)}")
    if args.strict_task_deduplicated and set(requested_representations) != {"task"}:
        raise ValueError("strict-task-deduplicated is defined only for the task representation")
    if args.strict_task_deduplicated and args.strict_model_input_deduplicated:
        raise ValueError("Choose either strict-task-deduplicated or strict-model-input-deduplicated")
    if args.strict_model_input_deduplicated and len(requested_representations) != 1:
        raise ValueError("strict-model-input-deduplicated requires exactly one representation")
    available_class_weights: dict[str, str | None] = {"unweighted": None, "balanced": "balanced"}
    requested_weights = tuple(name.strip() for name in args.class_weights.split(",") if name.strip())
    unknown_weights = set(requested_weights) - set(available_class_weights)
    if unknown_weights:
        raise ValueError(f"Unknown class weights: {sorted(unknown_weights)}")
    class_weights = {name: available_class_weights[name] for name in requested_weights}
    frame, data_audit = load_frame(args.input, args.strict_task_deduplicated, args.task_conflict_policy)
    if args.strict_model_input_deduplicated:
        representation_name = requested_representations[0]
        frame["model_input_key"] = clean_text(representation(frame, representation_name)).str.lower()
        label_count = frame.groupby("model_input_key")["target"].nunique()
        conflicting = label_count[label_count > 1].index
        data_audit["strict_model_input_deduplicated"] = True
        data_audit["excluded_conflicting_model_inputs"] = int(len(conflicting))
        data_audit["excluded_rows_with_conflicting_model_input"] = int(frame["model_input_key"].isin(conflicting).sum())
        frame = (
            frame.loc[~frame["model_input_key"].isin(conflicting)]
            .sort_values(["model_input_key", "jobrole_id"], kind="stable")
            .drop_duplicates("model_input_key")
            .reset_index(drop=True)
        )
        data_audit["model_rows"] = len(frame)
        data_audit["model_unique_full_inputs"] = int(frame["model_input_key"].nunique())
    else:
        data_audit["strict_model_input_deduplicated"] = False
    y = frame["target"].to_numpy()
    groups = frame["jobrole_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(frame, y, groups))
    if args.fold_index is not None and not 1 <= args.fold_index <= len(splits):
        raise ValueError(f"fold-index must be between 1 and {len(splits)}")
    selected_splits = (
        [(args.fold_index, splits[args.fold_index - 1])]
        if args.fold_index is not None
        else list(enumerate(splits, start=1))
    )
    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for representation_name in requested_representations:
        text = representation(frame, representation_name)
        model_input_key = clean_text(text).str.lower()
        predictions = {name: np.empty(len(frame), dtype=object) for name in class_weights}
        fold_assignments = np.full(len(frame), -1, dtype=int)
        for fold, (train_idx, test_idx) in selected_splits:
            train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
            shared_groups = len(train_groups & test_groups)
            if shared_groups:
                raise RuntimeError(f"Fold {fold} has {shared_groups} shared jobrole_id values")
            shared_tasks = len(set(frame.iloc[train_idx]["task_key"]) & set(frame.iloc[test_idx]["task_key"]))
            shared_inputs = len(set(model_input_key.iloc[train_idx]) & set(model_input_key.iloc[test_idx]))
            if args.strict_task_deduplicated and shared_tasks:
                raise RuntimeError(f"Fold {fold} has {shared_tasks} shared normalized task texts")
            if args.strict_model_input_deduplicated and shared_inputs:
                raise RuntimeError(f"Fold {fold} has {shared_inputs} shared normalized full model inputs")
            x_train, x_test = features(
                text.iloc[train_idx], text.iloc[test_idx], args.max_word_features, args.max_char_features
            )
            fold_assignments[test_idx] = fold
            for weight_name, class_weight in class_weights.items():
                model = LinearSVC(C=1.0, class_weight=class_weight, random_state=args.seed)
                model.fit(x_train, y[train_idx])
                fold_prediction = model.predict(x_test)
                predictions[weight_name][test_idx] = fold_prediction
                fold_rows.append(
                    {
                        "representation": representation_name,
                        "class_weight": weight_name,
                        "fold": fold,
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "shared_jobrole_ids": shared_groups,
                        "shared_exact_task_texts": shared_tasks,
                        "shared_exact_model_inputs": shared_inputs,
                        **metrics(y[test_idx], fold_prediction),
                    }
                )
        for weight_name, prediction in predictions.items():
            evaluated = fold_assignments >= 0
            if not evaluated.any() or pd.isna(prediction[evaluated]).any():
                raise RuntimeError(f"Missing predictions for {representation_name}/{weight_name}")
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "row_index": np.flatnonzero(evaluated),
                        "fold": fold_assignments[evaluated],
                        "jobrole_id": groups[evaluated],
                        "source_label": frame.loc[evaluated, "label"],
                        "target": y[evaluated],
                        "prediction": prediction[evaluated],
                        "representation": representation_name,
                        "class_weight": weight_name,
                    }
                )
            )

    fold_metrics = pd.DataFrame(fold_rows)
    leaderboard = (
        fold_metrics.groupby(["representation", "class_weight"], as_index=False)[
            ["accuracy", "macro_f1", "weighted_f1", "quadratic_weighted_kappa"]
        ]
        .mean()
        .sort_values("macro_f1", ascending=False)
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
    metadata = {
        "dataset": str(args.input),
        "rows": len(frame),
        "data_audit": data_audit,
        "source_label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "canonical_target_counts": frame["target"].value_counts().reindex(LABELS).to_dict(),
        "target_mapping": TARGET_MAP,
        "allowed_input_columns": ["task", "action", "object", "purpose", "jobrole_title"],
        "forbidden_input_columns": [
            "jobrole_id", "task_no", "sector_title", "reasoning_key", "truncated_reason", "label", "band", "score"
        ],
        "splitter": f"StratifiedGroupKFold(n_splits={args.folds}, shuffle=True, random_state={args.seed})",
        "group_column": "jobrole_id",
        "all_folds_have_zero_shared_jobrole_ids": bool((fold_metrics["shared_jobrole_ids"] == 0).all()),
        "all_folds_have_zero_shared_full_model_inputs": bool((fold_metrics["shared_exact_model_inputs"] == 0).all()),
        "executed_fold_indices": [int(fold) for fold, _ in selected_splits],
        "shared_exact_task_texts_are_audit_warnings": True,
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
