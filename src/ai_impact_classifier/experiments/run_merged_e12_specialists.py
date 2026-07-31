"""Test-blind E0/E3/E12 one-vs-rest sparse specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_specialist_hybrid import Candidate, _make_model
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


NAMES = ["E0", "E3", "E12"]
SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]
ALLOWED_TEXT_COLUMNS = ("keytask_content", "jobrole_title")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-only E0/E3/E12 sparse specialists.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/merged_e12_specialists")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.0, -1.0
    for value in np.linspace(-2.5, 2.5, 101):
        metric = f1_score(y, scores >= value, zero_division=0)
        if metric > best_f1: best_t, best_f1 = float(value), float(metric)
    return best_t, best_f1


def _allowed_text(frame: pd.DataFrame, view: str) -> pd.Series:
    """Build the model input solely from the permitted textual columns."""
    task = frame["keytask_content"].fillna("").astype(str)
    if view == "task_only":
        return "TASK " + task
    if view == "title_task":
        title = frame["jobrole_title"].fillna("").astype(str)
        return "ROLE " + title + " TASK " + task
    raise ValueError(f"Unsupported text view: {view}")


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    bias = np.zeros(scores.shape[1]); best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        changed = False
        for cls in range(scores.shape[1]):
            for value in np.linspace(-1.5, 1.5, 41):
                trial = bias.copy(); trial[cls] = value
                metric = f1_score(y, np.argmax(scores + trial, axis=1), average="macro")
                if metric > best: bias, best, changed = trial, metric, True
        if not changed: break
    return bias


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input); frame = bundle.frame.copy()
    split = stratified_group_train_val_test_split(frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    candidates = [Candidate("word_c025_w2", "word", .25, 2), Candidate("word_c05_w4", "word", .5, 4), Candidate("word_c1_w8", "word", 1, 8), Candidate("char_c025_w2", "char_word", .25, 2), Candidate("char_c05_w4", "char_word", .5, 4), Candidate("char_c075_w8", "char_word", .75, 8), Candidate("char_c1_w16", "char_word", 1, 16), Candidate("charlong_c05_w4", "char_word_long", .5, 4), Candidate("charlong_c1_w12", "char_word_long", 1, 12)]
    views = ("task_only", "title_task")
    selected: list[tuple[Candidate, str]] = []; thresholds: list[float] = []; val_scores = np.zeros((len(val), 3)); rows = []
    for cls, label in enumerate(NAMES):
        best: tuple[Candidate, str, np.ndarray, float, float] | None = None
        for view in views:
            train_text = _allowed_text(train, view)
            val_text = _allowed_text(val, view)
            for candidate in candidates:
                model = _make_model(candidate); model.fit(train_text, (y_train == cls).astype(int)); scores = model.decision_function(val_text)
                threshold, metric = _threshold((y_val == cls).astype(int), scores)
                rows.append({"label": label, "view": view, "candidate": candidate.name, "threshold": threshold, "binary_val_f1": metric})
                if best is None or metric > best[4]: best = (candidate, view, scores, threshold, metric)
        assert best is not None
        selected.append((best[0], best[1])); thresholds.append(best[3]); val_scores[:, cls] = best[2] - best[3]
    bias = _bias(y_val, val_scores); val_pred = np.argmax(val_scores + bias, axis=1); val_metrics = _metrics(y_val, val_pred)
    development = pd.concat([train, val]); y_dev = development.label_rank.to_numpy(); test_scores = np.zeros((len(test), 3))
    for cls, ((candidate, view), threshold) in enumerate(zip(selected, thresholds)):
        model = _make_model(candidate); model.fit(_allowed_text(development, view), (y_dev == cls).astype(int)); test_scores[:, cls] = model.decision_function(_allowed_text(test, view)) - threshold
    test_pred = np.argmax(test_scores + bias, axis=1); test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame(rows).to_csv(out / "binary_validation_search.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"original_label": test.openai_label.to_numpy(), "y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"target_definition": "E0 vs E3 vs E12, where E12 merges original E1 and E2", "test_blind_selection": True, "allowed_input_columns": list(ALLOWED_TEXT_COLUMNS), "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code", "ssoc_title", "sector_title"], "selected_by_label": {name: {"candidate": candidate.name, "text_view": view, "threshold": threshold} for name, (candidate, view), threshold in zip(NAMES, selected, thresholds)}, "bias": bias.tolist(), "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__": main()
