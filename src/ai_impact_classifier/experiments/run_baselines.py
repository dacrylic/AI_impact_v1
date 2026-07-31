from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, LABEL_COLUMN, SCORE_COLUMN, build_bundle
from ai_impact_classifier.embeddings import encode_texts
from ai_impact_classifier.metrics import (
    classification_metrics,
    project_scores_to_ranks,
    regression_metrics,
)
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


DEFAULT_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]


SOFT_ID_COLUMNS = [
    "task_id",
    "jobrole_task_id",
    "ssoc_code",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding baselines for AI impact prediction.")
    parser.add_argument("--input", required=True, help="Path to the source .xlsx workbook.")
    parser.add_argument("--output-dir", default="results/baselines", help="Directory to write run artifacts.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Embedding model names to compare.",
    )
    parser.add_argument("--train-size", type=float, default=0.7, help="Train fraction.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--heads",
        nargs="*",
        default=["classification"],
        choices=["classification", "regression"],
        help="Which prediction heads to run.",
    )
    return parser.parse_args()


def _save_artifact(obj: object, path: Path) -> None:
    joblib.dump(obj, path)


def _prepare_xy(frame: pd.DataFrame, idx: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    subset = frame.loc[idx]
    texts = subset["target_text"].tolist()
    y = subset["label_rank"].to_numpy(dtype=np.int64)
    scores = subset[SCORE_COLUMN].to_numpy(dtype=np.float32)
    return texts, y, scores


def _fit_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_order: List[str],
) -> Dict[str, float]:
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(class_weight="balanced"),
    )
    classifier.fit(x_train, y_train)
    val_pred = classifier.predict(x_val)
    test_pred = classifier.predict(x_test)
    metrics = classification_metrics(y_test, test_pred)
    metrics["val_accuracy"] = float(classification_metrics(y_val, val_pred)["accuracy"])
    metrics["val_macro_f1"] = float(classification_metrics(y_val, val_pred)["macro_f1"])
    metrics["val_qwk"] = float(classification_metrics(y_val, val_pred)["qwk"])
    metrics["model_head"] = "classification"
    metrics["label_order"] = "|".join(label_order)
    return metrics


def _fit_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_order: List[str],
    score_values: np.ndarray,
) -> Dict[str, float]:
    regressor = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, random_state=42),
    )
    regressor.fit(x_train, y_train)
    val_pred = regressor.predict(x_val)
    test_pred = regressor.predict(x_test)
    metrics = regression_metrics(y_test, test_pred, score_values=score_values)
    val_projected = project_scores_to_ranks(val_pred, score_values=score_values)
    metrics["val_ordinal_mae"] = float(np.mean(np.abs(y_val - val_projected)))
    metrics["val_accuracy"] = float(np.mean(val_projected == y_val))
    metrics["val_macro_f1"] = float(
        classification_metrics(y_val, val_projected)["macro_f1"]
    )
    metrics["val_qwk"] = float(classification_metrics(y_val, val_projected)["qwk"])
    metrics["model_head"] = "regression"
    metrics["label_order"] = "|".join(label_order)
    return metrics


def main() -> None:
    args = parse_args()
    if "regression" in args.heads:
        raise ValueError(
            "Regression is not defined for the canonical E0/E3/E12 target. "
            "ai_impact_score is retained for audit only."
        )
    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(
        bundle.frame,
        group_column=GROUP_COLUMN,
        label_column="label_rank",
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    leak_report = check_split_leakage(
        bundle.frame,
        split_assignments={
            "train": split.train_idx,
            "val": split.val_idx,
            "test": split.test_idx,
        },
        group_column=GROUP_COLUMN,
        hard_id_columns=(),
        soft_id_columns=SOFT_ID_COLUMNS,
        text_column="target_text",
    )
    if not leak_report.passed:
        raise RuntimeError("Leakage check failed:\n" + "\n".join(leak_report.messages))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input": str(Path(args.input).resolve()),
        "label_order": bundle.label_order,
        "label_to_rank": bundle.label_to_rank,
        "score_by_label": bundle.score_by_label,
        "train_rows": int(len(split.train_idx)),
        "val_rows": int(len(split.val_idx)),
        "test_rows": int(len(split.test_idx)),
        "models": args.models,
        "heads": args.heads,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "allowed_input_columns": ["keytask_content", "jobrole_title"],
        "forbidden_input_columns": [
            "ai_impact_score", "ai_impact_category", "openai_label", "label_rank",
            "task_id", "jobrole_id", "ssoc_code", "ssoc_title", "sector_title",
        ],
        "split_leakage_passed": leak_report.passed,
        "split_leakage_messages": leak_report.messages,
        "split_leakage_warnings": leak_report.warnings,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    split_assignments = {
        "train": split.train_idx,
        "val": split.val_idx,
        "test": split.test_idx,
    }
    split_assignments_path = output_dir / "split_assignments.csv"
    split_frame = pd.concat(
        [
            pd.DataFrame({"row_index": idx, "split": name})
            for name, idx in split_assignments.items()
        ],
        ignore_index=True,
    )
    split_frame.to_csv(split_assignments_path, index=False)

    split_summary = []
    for split_name, idx in split_assignments.items():
        subset = bundle.frame.loc[idx]
        counts = subset["label_rank"].value_counts().sort_index()
        row = {"split": split_name, "rows": int(len(subset))}
        for label_rank, count in counts.items():
            row[f"class_{int(label_rank)}"] = int(count)
        split_summary.append(row)
    pd.DataFrame(split_summary).fillna(0).to_csv(output_dir / "split_summary.csv", index=False)

    results = []
    for model_name in args.models:
        train_texts, y_train, _ = _prepare_xy(bundle.frame, np.asarray(split.train_idx))
        val_texts, y_val, _ = _prepare_xy(bundle.frame, np.asarray(split.val_idx))
        test_texts, y_test, _ = _prepare_xy(bundle.frame, np.asarray(split.test_idx))

        x_train = encode_texts(model_name, train_texts, batch_size=args.batch_size)
        x_val = encode_texts(model_name, val_texts, batch_size=args.batch_size)
        x_test = encode_texts(model_name, test_texts, batch_size=args.batch_size)

        model_dir = output_dir / Path(model_name).name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        _save_artifact({"x_train_shape": x_train.shape, "x_val_shape": x_val.shape, "x_test_shape": x_test.shape}, model_dir / "embedding_shapes.joblib")

        if "classification" in args.heads:
            metrics = _fit_classification(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                label_order=bundle.label_order,
            )
            metrics.update({"model": model_name, "target": LABEL_COLUMN})
            results.append(metrics)

    results_frame = pd.DataFrame(results)
    results_frame = results_frame.sort_values(
        by=["model_head", "qwk", "accuracy"],
        ascending=[True, False, False],
        na_position="last",
    )
    results_frame.to_csv(output_dir / "baseline_results.csv", index=False)
    results_frame.to_json(output_dir / "baseline_results.json", orient="records", indent=2)

    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
