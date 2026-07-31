"""Validation-selected semantic E3 gate using only title and task text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text, _bias, _metrics, _threshold
from ai_impact_classifier.experiments.run_specialist_hybrid import Candidate, _make_model
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


NAMES = ["E0", "E3", "E12"]
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
E3_HYPOTHESIS = "This is a creative, visual-design, artistic, media-production, or marketing-content creation task."
SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a semantic E3 gate with only job role title and task content.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/e3_nli_gate")
    parser.add_argument("--model", default=NLI_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _premises(frame: pd.DataFrame) -> list[str]:
    return (
        "Role: " + frame.jobrole_title.fillna("").astype(str)
        + "\nTask: " + frame.keytask_content.fillna("").astype(str)
    ).tolist()


@torch.inference_mode()
def _entailment_scores(model_name: str, premises: list[str], batch_size: int) -> np.ndarray:
    device = _device(); tokenizer = AutoTokenizer.from_pretrained(model_name); model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device); model.eval()
    entailment_index = next((int(index) for index, label in model.config.id2label.items() if "entail" in str(label).lower()), None)
    if entailment_index is None: raise ValueError(f"Could not identify entailment label in {model.config.id2label}")
    result = []
    for start in range(0, len(premises), batch_size):
        batch = premises[start : start + batch_size]
        encoded = tokenizer(batch, [E3_HYPOTHESIS] * len(batch), truncation=True, max_length=256, padding=True, return_tensors="pt")
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        result.append(torch.softmax(logits, dim=1)[:, entailment_index].float().cpu().numpy())
    return np.concatenate(result)


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input); frame = bundle.frame.copy()
    split = stratified_group_train_val_test_split(frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    # Fixed text-only baseline selected in the preceding validation-only constrained search.
    selected = [(Candidate("E0", "char_word", .75, 8), "title_task"), (Candidate("E3", "word", .5, 4), "task_only"), (Candidate("E12", "char_word_long", .5, 4), "task_only")]
    thresholds = np.array([.05, -.35, .10]); val_scores = np.zeros((len(val), 3))
    for cls, ((candidate, view), threshold) in enumerate(zip(selected, thresholds)):
        model = _make_model(candidate); model.fit(_allowed_text(train, view), (y_train == cls).astype(int)); val_scores[:, cls] = model.decision_function(_allowed_text(val, view)) - threshold
    bias = _bias(y_val, val_scores); base_val = np.argmax(val_scores + bias, axis=1)
    nli_val = _entailment_scores(args.model, _premises(val), args.batch_size)
    best, grid = None, []
    for threshold in np.linspace(.05, .95, 37):
        pred = base_val.copy(); mask = nli_val >= threshold; pred[mask] = 1; metric = _metrics(y_val, pred)
        row = {"nli_threshold": float(threshold), "overrides": int(mask.sum()), **metric}; grid.append(row)
        if best is None or metric["macro_f1"] > best[1]["macro_f1"]: best = (float(threshold), metric)
    assert best is not None
    threshold, val_metrics = best
    development = pd.concat([train, val]); y_dev = development.label_rank.to_numpy(); test_scores = np.zeros((len(test), 3))
    for cls, ((candidate, view), margin_threshold) in enumerate(zip(selected, thresholds)):
        model = _make_model(candidate); model.fit(_allowed_text(development, view), (y_dev == cls).astype(int)); test_scores[:, cls] = model.decision_function(_allowed_text(test, view)) - margin_threshold
    test_pred = np.argmax(test_scores + bias, axis=1)
    nli_test = _entailment_scores(args.model, _premises(test), args.batch_size); mask = nli_test >= threshold; test_pred[mask] = 1; test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame(grid).sort_values("macro_f1", ascending=False).to_csv(out / "nli_gate_validation_grid.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred, "e3_nli_entailment": nli_test, "e3_nli_override": mask}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"target_definition": "E0 vs E3 vs E12", "test_blind_selection": True, "model": args.model, "device": _device(), "allowed_input_columns": ["keytask_content", "jobrole_title"], "hypothesis": E3_HYPOTHESIS, "selected_nli_threshold": threshold, "test_overrides": int(mask.sum()), "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics, "test_overrides": int(mask.sum())}, indent=2))


if __name__ == "__main__": main()
