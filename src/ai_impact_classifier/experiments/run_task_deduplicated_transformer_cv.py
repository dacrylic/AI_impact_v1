"""GPU-ready strict task-only Transformer CV for the canonical E0/E1/E23 target."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from ai_impact_classifier.experiments.run_task_deduplicated_cv import build_strict_task_dataset


CLASS_NAMES = ("E0", "E1", "E23")


class TaskDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: list[str], labels: np.ndarray, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels.astype(int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {**{key: value[index] for key, value in self.encodings.items()}, "labels": int(self.labels[index])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict task-deduplicated Transformer five-fold CV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/task_only_transformer_cv5")
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: DataLoader, target_device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for batch in loader:
        batch.pop("labels")
        logits = model(**{key: value.to(target_device) for key, value in batch.items()}).logits
        predictions.append(logits.float().cpu().numpy().argmax(axis=1))
    return np.concatenate(predictions)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    target_device = device()
    frame, audit = build_strict_task_dataset(args.input)
    texts = ("TASK " + frame["keytask_content"].fillna("").astype(str)).tolist()
    y = frame["label_rank"].to_numpy()
    oof = np.full(len(frame), -1, dtype=int)
    fold_rows: list[dict[str, float | int]] = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed).split(texts, y), start=1):
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
        train_loader = DataLoader(TaskDataset(tokenizer, [texts[index] for index in train_idx], y[train_idx], args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        test_loader = DataLoader(TaskDataset(tokenizer, [texts[index] for index in test_idx], y[test_idx], args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(target_device)
        class_counts = np.bincount(y[train_idx], minlength=3)
        class_weights = torch.tensor(np.sqrt(len(train_idx) / (3 * class_counts)), dtype=torch.float32, device=target_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        schedule = get_linear_schedule_with_warmup(optimizer, max(1, int(len(train_loader) * args.epochs * 0.1)), len(train_loader) * args.epochs)
        for _ in range(args.epochs):
            model.train()
            for batch in train_loader:
                labels = batch.pop("labels").to(target_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(**{key: value.to(target_device) for key, value in batch.items()}).logits
                loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
        oof[test_idx] = predict(model, test_loader, target_device)
        fold_rows.append({"fold": fold, "accuracy": float(accuracy_score(y[test_idx], oof[test_idx])), "macro_f1": float(f1_score(y[test_idx], oof[test_idx], average="macro"))})
        print(json.dumps(fold_rows[-1]), flush=True)
        del model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "method": "strict normalized-task-deduplicated five-fold Transformer CV",
        "input_columns": ["keytask_content"],
        "audit": audit,
        "model": args.model,
        "device": str(target_device),
        "fold_metrics": fold_rows,
        "aggregate": {"accuracy": float(accuracy_score(y, oof)), "macro_f1": float(f1_score(y, oof, average="macro")), "weighted_f1": float(f1_score(y, oof, average="weighted"))},
        "classification_report": classification_report(y, oof, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }
    pd.DataFrame({"task_key": frame["task_key"], "y_true": y, "y_pred": oof}).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y, oof, labels=range(3)), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "oof_confusion_matrix.csv")
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
