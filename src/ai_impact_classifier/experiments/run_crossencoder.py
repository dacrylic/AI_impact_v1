"""Leak-safe cross-encoder experiment for text-only AI impact prediction.

The encoder sees role, occupation, sector, and task text jointly.  It never
receives the score, label, IDs, or any target-derived columns as inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cross-encoder text classifier with validation checkpoint selection.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/crossencoder")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _texts(frame: pd.DataFrame) -> list[str]:
    return (
        "Role: "
        + frame["jobrole_title"].fillna("").astype(str)
        + "\nOccupation: "
        + frame["ssoc_title"].fillna("").astype(str)
        + "\nSector: "
        + frame["sector_title"].fillna("").astype(str)
        + "\nTask: "
        + frame["keytask_content"].fillna("").astype(str)
    ).tolist()


class TextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: list[str], labels: np.ndarray, max_length: int) -> None:
        # Dynamic padding avoids running every short task through a 256-token model pass.
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


@torch.inference_mode()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        label = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(**batch).logits
        scores.append(logits.detach().float().cpu().numpy())
        labels.append(label.numpy())
    return np.vstack(scores), np.concatenate(labels)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leakage = check_split_leakage(
        bundle.frame,
        split_assignments={"train": split.train_idx, "val": split.val_idx, "test": split.test_idx},
        group_column=GROUP_COLUMN,
        text_column="target_text",
        soft_id_columns=SOFT_ID_COLUMNS,
    )
    if not leakage.passed:
        raise RuntimeError("Leakage check failed: " + "; ".join(leakage.messages))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part["label_rank"].to_numpy() for part in (train, val, test))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    train_loader = DataLoader(TextDataset(tokenizer, _texts(train), y_train, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(TextDataset(tokenizer, _texts(val), y_val, args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)
    test_loader = DataLoader(TextDataset(tokenizer, _texts(test), y_test, args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)

    device = _device()
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(bundle.label_order)).to(device)
    # Sqrt inverse frequency retains a minority signal without making the 201-row E3 class dominate updates.
    counts = np.bincount(y_train, minlength=len(bundle.label_order)).astype(np.float32)
    class_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    class_weights /= class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(steps * 0.1)), num_training_steps=steps)

    best_state, best_epoch, best_val = None, 0, -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        val_scores, val_labels = _predict(model, val_loader, device)
        val_metric = _metrics(val_labels, np.argmax(val_scores, axis=1))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), **val_metric})
        print(json.dumps(history[-1]), flush=True)
        if val_metric["macro_f1"] > best_val:
            best_val, best_epoch = val_metric["macro_f1"], epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    assert best_state is not None
    model.load_state_dict(best_state)
    test_scores, test_labels = _predict(model, test_loader, device)
    test_pred = np.argmax(test_scores, axis=1)
    test_metrics = _metrics(test_labels, test_pred)
    torch.save(best_state, output_dir / "best_model_state.pt")
    pd.DataFrame(history).to_csv(output_dir / "validation_history.csv", index=False)
    pd.DataFrame(confusion_matrix(test_labels, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(output_dir / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": test_labels, "y_pred": test_pred}).to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "test_blind_selection": True,
                "model": args.model,
                "device": str(device),
                "best_epoch_by_val_macro_f1": best_epoch,
                "best_val_macro_f1": best_val,
                "test_metrics": test_metrics,
                "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id"],
                "leakage_passed": leakage.passed,
                "leakage_warnings": leakage.warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_epoch": best_epoch, "best_val_macro_f1": best_val, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
