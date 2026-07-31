"""Balanced text-only transformer detector for rare E3 tasks."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a balanced text-only E3 detector.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/e3_detector")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _texts(frame: pd.DataFrame) -> list[str]:
    return (
        "Role: " + frame.jobrole_title.fillna("").astype(str)
        + "\nTask: " + frame.keytask_content.fillna("").astype(str)
    ).tolist()


class TextSet(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: list[str], y: np.ndarray, max_length: int) -> None:
        self.enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False); self.y = y.astype(int)
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, index: int) -> dict[str, object]:
        return {**{key: value[index] for key, value in self.enc.items()}, "labels": int(self.y[index])}


@torch.inference_mode()
def _scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); output, labels = [], []
    for batch in loader:
        y = batch.pop("labels"); batch = {key: value.to(device) for key, value in batch.items()}
        output.append(model(**batch).logits[:, 1].float().cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(output), np.concatenate(labels)


def _best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_t, best = 0.0, -1.0
    for value in np.linspace(-5, 5, 201):
        metric = f1_score(y, scores >= value, zero_division=0)
        if metric > best: best_t, best = float(value), float(metric)
    return best_t, best


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    bundle = build_bundle(args.input); split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(bundle.frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = ((part.openai_label == "E3").to_numpy(dtype=int) for part in (train, val, test))
    tokenizer = AutoTokenizer.from_pretrained(args.model); collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    train_set = TextSet(tokenizer, _texts(train), y_train, args.max_length)
    weights = np.where(y_train == 1, len(y_train) / (2 * max(y_train.sum(), 1)), len(y_train) / (2 * max((y_train == 0).sum(), 1)))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(y_train), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collator)
    val_loader = DataLoader(TextSet(tokenizer, _texts(val), y_val, args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)
    test_loader = DataLoader(TextSet(tokenizer, _texts(test), y_test, args.max_length), batch_size=args.batch_size * 2, collate_fn=collator)
    device = _device(); model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01); steps = len(train_loader) * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * .1)), steps)
    best_state, best_epoch, best_threshold, best_val = None, 0, 0.0, -1.0; history=[]
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_total = 0.0
        for batch in train_loader:
            y = batch.pop("labels").to(device); batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(**batch).logits, y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); schedule.step(); loss_total += float(loss.detach().cpu())
        score, labels = _scores(model, val_loader, device); threshold, metric = _best_threshold(labels, score)
        history.append({"epoch": epoch, "train_loss": loss_total / len(train_loader), "val_e3_f1": metric, "threshold": threshold}); print(json.dumps(history[-1]), flush=True)
        if metric > best_val: best_state, best_epoch, best_threshold, best_val = copy.deepcopy({key:value.detach().cpu() for key,value in model.state_dict().items()}), epoch, threshold, metric
    assert best_state is not None; model.load_state_dict(best_state); score, labels = _scores(model, test_loader, device); pred = (score >= best_threshold).astype(int)
    test_f1 = f1_score(labels, pred, zero_division=0)
    torch.save(best_state, out / "best_model_state.pt"); pd.DataFrame(history).to_csv(out / "validation_history.csv", index=False); pd.DataFrame({"y_true": labels, "e3_logit": score, "y_pred": pred}).to_csv(out / "test_predictions.csv", index=False); pd.DataFrame(confusion_matrix(labels,pred,labels=[0,1]),index=["not_E3","E3"],columns=["not_E3","E3"]).to_csv(out / "test_confusion_matrix.csv")
    payload={"target":"E3 vs not_E3", "test_blind_selection":True,"device":str(device),"allowed_input_columns":["keytask_content","jobrole_title"],"best_epoch":best_epoch,"threshold":best_threshold,"best_val_e3_f1":best_val,"test_e3_f1":float(test_f1),"forbidden_input_columns":["ai_impact_score","ai_impact_category","openai_label","label_rank","task_id","jobrole_id","ssoc_code","ssoc_title","sector_title"],"leakage_passed":leak.passed,"leakage_warnings":leak.warnings}; (out / "run_metadata.json").write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))


if __name__ == "__main__": main()
