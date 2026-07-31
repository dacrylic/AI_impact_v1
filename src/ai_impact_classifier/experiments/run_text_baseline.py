from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.metrics import classification_metrics
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


DEFAULT_MODEL = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=250_000),
    LinearSVC(class_weight="balanced", C=1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strongest sparse text baseline.")
    parser.add_argument("--input", required=True, help="Path to the source .xlsx workbook.")
    parser.add_argument("--output-dir", default="results/text_baseline", help="Directory to write artifacts.")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        text_column="target_text",
        soft_id_columns=["task_id", "jobrole_task_id", "ssoc_code"],
    )
    if not leak_report.passed:
        raise RuntimeError("Leakage check failed:\n" + "\n".join(leak_report.messages))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = bundle.frame.loc[split.train_idx, "target_text"]
    x_val = bundle.frame.loc[split.val_idx, "target_text"]
    x_test = bundle.frame.loc[split.test_idx, "target_text"]
    y_train = bundle.frame.loc[split.train_idx, "label_rank"]
    y_val = bundle.frame.loc[split.val_idx, "label_rank"]
    y_test = bundle.frame.loc[split.test_idx, "label_rank"]

    model = DEFAULT_MODEL
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    train_metrics = classification_metrics(y_train.to_numpy(), train_pred)
    val_metrics = classification_metrics(y_val.to_numpy(), val_pred)
    test_metrics = classification_metrics(y_test.to_numpy(), test_pred)

    report = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "val", **val_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    report.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "label_order": bundle.label_order,
                "label_to_rank": bundle.label_to_rank,
                "score_by_label": bundle.score_by_label,
                "train_rows": int(len(split.train_idx)),
                "val_rows": int(len(split.val_idx)),
                "test_rows": int(len(split.test_idx)),
                "split_leakage_passed": leak_report.passed,
                "split_leakage_messages": leak_report.messages,
                "split_leakage_warnings": leak_report.warnings,
                "model": "TfidfVectorizer(1,2)+LinearSVC(class_weight=balanced,C=1.0)",
                "text_source": "jobrole_title + keytask_content",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(model, output_dir / "model.joblib")

    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
