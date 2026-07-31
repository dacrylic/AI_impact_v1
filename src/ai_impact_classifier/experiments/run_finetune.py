from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, SCORE_COLUMN, build_bundle
from ai_impact_classifier.embeddings import default_device, encode_texts
from ai_impact_classifier.metrics import classification_metrics, regression_metrics, project_scores_to_ranks
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
    parser = argparse.ArgumentParser(description="Fine-tune a sentence transformer on the AI impact dataset.")
    parser.add_argument("--input", required=True, help="Path to the source .xlsx workbook.")
    parser.add_argument("--output-dir", default="results/finetune", help="Directory to write run artifacts.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS[0],
        help="Sentence-transformer checkpoint or local path to fine-tune.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["title_task"],
        default="title_task",
        help="Use the permitted job-role-title and key-task-content text pair.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--heads",
        nargs="*",
        default=["classification"],
        choices=["classification", "regression"],
        help="Which probes to evaluate on the fine-tuned embeddings.",
    )
    return parser.parse_args()


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_pairs(frame: pd.DataFrame, pair_mode: str) -> Tuple[List[Tuple[str, str]], pd.Series]:
    if pair_mode == "title_task":
        left = frame["jobrole_title"].fillna("").astype(str).tolist()
        right = frame["keytask_content"].fillna("").astype(str).tolist()
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")
    return list(zip(left, right)), frame["target_text"]


def _prepare_examples(
    frame: pd.DataFrame,
    idx: pd.Index,
    pair_mode: str,
) -> Tuple[List[InputExample], List[str], np.ndarray, np.ndarray]:
    subset = frame.loc[idx]
    pairs, texts = _build_pairs(subset, pair_mode)
    examples = [InputExample(texts=[left, right], label=int(label)) for (left, right), label in zip(pairs, subset["label_rank"].tolist())]
    return examples, texts.tolist(), subset["label_rank"].to_numpy(dtype=np.int64), subset[SCORE_COLUMN].to_numpy(dtype=np.float32)


def _fit_classification_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[str, Dict[str, float], np.ndarray]:
    candidates = {
        "ridge": make_pipeline(StandardScaler(), RidgeClassifier(class_weight="balanced")),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=1)),
        "linear_svc": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", max_iter=10000)),
    }
    best_name = None
    best_val = -1.0
    best_metrics: Dict[str, float] = {}
    best_test_pred = None
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        val_metrics = classification_metrics(y_val, val_pred)
        test_metrics = classification_metrics(y_test, test_pred)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_name = name
            best_metrics = {
                **test_metrics,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_qwk": float(val_metrics["qwk"]),
            }
            best_test_pred = test_pred
    assert best_name is not None and best_test_pred is not None
    return best_name, best_metrics, best_test_pred


def _fit_regression_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    score_values: np.ndarray,
) -> Tuple[str, Dict[str, float], np.ndarray]:
    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42)),
        "huber": make_pipeline(StandardScaler(), HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=2000)),
    }
    best_name = None
    best_val = -1.0
    best_metrics: Dict[str, float] = {}
    best_test_pred = None
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        val_projected = project_scores_to_ranks(val_pred, score_values=score_values)
        test_projected = project_scores_to_ranks(test_pred, score_values=score_values)
        val_macro_f1 = float(f1_score(y_val, val_projected, average="macro"))
        if val_macro_f1 > best_val:
            best_val = val_macro_f1
            best_name = name
            best_metrics = regression_metrics(y_test, test_pred, score_values=score_values)
            best_metrics["val_ordinal_mae"] = float(np.mean(np.abs(y_val - val_projected)))
            best_metrics["val_accuracy"] = float(accuracy_score(y_val, val_projected))
            best_metrics["val_macro_f1"] = val_macro_f1
            best_metrics["val_qwk"] = float(cohen_kappa_score(y_val, val_projected, weights="quadratic"))
            best_test_pred = test_projected
    assert best_name is not None and best_test_pred is not None
    return best_name, best_metrics, best_test_pred


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

    train = bundle.frame.loc[split.train_idx].copy()
    val = bundle.frame.loc[split.val_idx].copy()
    test = bundle.frame.loc[split.test_idx].copy()

    train_examples, train_texts, y_train, train_scores = _prepare_examples(train, train.index, args.pair_mode)
    val_examples, val_texts, y_val, val_scores = _prepare_examples(val, val.index, args.pair_mode)
    test_examples, test_texts, y_test, test_scores = _prepare_examples(test, test.index, args.pair_mode)

    metadata = {
        "input": str(Path(args.input).resolve()),
        "model": args.model,
        "pair_mode": args.pair_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_rows": int(len(train_examples)),
        "val_rows": int(len(val_examples)),
        "test_rows": int(len(test_examples)),
        "label_order": bundle.label_order,
        "label_to_rank": bundle.label_to_rank,
        "score_by_label": bundle.score_by_label,
        "split_leakage_passed": leak_report.passed,
        "split_leakage_messages": leak_report.messages,
        "split_leakage_warnings": leak_report.warnings,
        "device": default_device(),
        "allowed_input_columns": ["keytask_content", "jobrole_title"],
        "forbidden_input_columns": [
            "ai_impact_score", "ai_impact_category", "openai_label", "label_rank",
            "task_id", "jobrole_id", "ssoc_code", "ssoc_title", "sector_title",
        ],
    }
    _save_json(output_dir / "run_metadata.json", metadata)

    model_path = output_dir / "finetuned_model"
    model = SentenceTransformer(args.model, device=default_device())
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_loss = losses.SoftmaxLoss(
        model,
        model.get_sentence_embedding_dimension(),
        num_labels=len(bundle.label_order),
    )
    warmup_steps = max(10, int(len(train_loader) * args.epochs * 0.1))
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(model_path),
        show_progress_bar=True,
        save_best_model=False,
    )

    fine_tuned_path = str(model_path)
    x_train = encode_texts(fine_tuned_path, train_texts, batch_size=args.batch_size)
    x_val = encode_texts(fine_tuned_path, val_texts, batch_size=args.batch_size)
    x_test = encode_texts(fine_tuned_path, test_texts, batch_size=args.batch_size)

    results = []
    if "classification" in args.heads:
        best_name, best_metrics, best_test_pred = _fit_classification_probe(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )
        results.append({
            "head": "classification",
            "probe": best_name,
            "model": args.model,
            **best_metrics,
        })
        joblib.dump(best_test_pred, output_dir / "classification_test_pred.joblib")

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_dir / "results.csv", index=False)
    results_frame.to_json(output_dir / "results.json", orient="records", indent=2)
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
