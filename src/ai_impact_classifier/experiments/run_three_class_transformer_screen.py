"""Strict title/task three-class Transformer screening experiment on one held-out role fold."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _metrics
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


NAMES = ["E0", "E3", "E12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen a strict three-class title/task Transformer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/three_class_transformer_screen")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: list[str], y: np.ndarray, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.labels = y.astype(int)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {**{name: values[index] for name, values in self.encodings.items()}, "labels": int(self.labels[index])}


@torch.inference_mode()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); logits, labels = [], []
    for batch in loader:
        y = batch.pop("labels")
        inputs = {name: values.to(device) for name, values in batch.items()}
        logits.append(model(**inputs).logits.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, validation_idx = next(outer.split(frame, y, groups))
    train, validation = frame.iloc[train_idx], frame.iloc[validation_idx]
    y_train, y_validation = y[train_idx], y[validation_idx]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    train_loader = DataLoader(TextDataset(tokenizer, _allowed_text(train, "title_task").tolist(), y_train, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(TextDataset(tokenizer, _allowed_text(validation, "title_task").tolist(), y_validation, args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)
    device = _device(); model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3).to(device)
    counts = np.bincount(y_train, minlength=3)
    class_weights = np.sqrt(len(y_train) / (3 * counts)); class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * .1)), steps)
    best = None; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: values.to(device) for name, values in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            per_row = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction="none")
            focal_weight = (1.0 - torch.exp(-per_row)).pow(1.5)
            loss = (focal_weight * per_row).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); schedule.step(); losses.append(float(loss.detach().cpu()))
        logits, labels = _predict(model, validation_loader, device); predictions = logits.argmax(axis=1)
        metrics = _metrics(labels, predictions)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(row); print(json.dumps(row), flush=True)
        if best is None or metrics["macro_f1"] > best[0]:
            best = (metrics["macro_f1"], copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}), predictions, logits)
    assert best is not None
    _, state, predictions, logits = best; model.load_state_dict(state)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "validation_history.csv", index=False)
    pd.DataFrame(confusion_matrix(y_validation, predictions, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "validation_confusion_matrix.csv")
    pd.DataFrame({"row_index": validation_idx, "y_true": y_validation, "y_pred": predictions, "e0_logit": logits[:, 0], "e3_logit": logits[:, 1], "e12_logit": logits[:, 2]}).to_csv(out / "validation_predictions.csv", index=False)
    torch.save(state, out / "best_model_state.pt")
    payload = {"method": "three-class title/task Transformer with square-root class weighting and focal loss", "screening_only": True, "validation_is_one group-held-out outer fold": True, "device": str(device), "allowed_input_columns": ["keytask_content", "jobrole_title"], "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code", "ssoc_title", "sector_title"], "best_validation_metrics": _metrics(y_validation, predictions), "class_counts_train": counts.tolist()}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
