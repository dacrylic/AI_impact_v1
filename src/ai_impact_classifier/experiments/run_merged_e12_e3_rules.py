"""E0/E3/E12 specialists plus validation-selected E3 phrase gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_merged_e12_specialists import NAMES, _bias, _metrics, _text, _threshold
from ai_impact_classifier.experiments.run_specialist_hybrid import Candidate, _make_model
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 phrase-rule gates on top of merged-class specialists.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/merged_e12_e3_rules")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _rule_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.jobrole_title.fillna("").astype(str) + " "
        + frame.ssoc_title.fillna("").astype(str) + " "
        + frame.sector_title.fillna("").astype(str) + " "
        + frame.keytask_content.fillna("").astype(str)
    )


def _rule_scores(train_text: pd.Series, train_y: np.ndarray, eval_text: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=2, binary=True, max_features=500_000, strip_accents="unicode")
    x_train = vectorizer.fit_transform(train_text); x_eval = vectorizer.transform(eval_text)
    pos = np.asarray(x_train[train_y == 1].sum(axis=0)).ravel()
    total = np.asarray(x_train.sum(axis=0)).ravel()
    confidence = pos / np.maximum(total, 1)
    # For each row, expose the most E3-specific matching phrase and its positive support.
    weighted_conf = x_eval.multiply(confidence).max(axis=1).toarray().ravel()
    weighted_pos = x_eval.multiply(pos).max(axis=1).toarray().ravel()
    return weighted_conf, weighted_pos


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input); frame = bundle.frame.copy()
    frame["merged_rank"] = frame.openai_label.map({"E0": 0, "E3": 1, "E2": 2, "E1": 2}).astype(int)
    split = stratified_group_train_val_test_split(frame, GROUP_COLUMN, "merged_rank", random_state=args.seed)
    leak = check_split_leakage(frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.merged_rank.to_numpy() for part in (train, val, test))
    selected = [Candidate("E0", "char_word", .5, 4), Candidate("E3", "char_word", .5, 4), Candidate("E12", "char_word_long", .5, 4)]
    thresholds = []; val_scores = np.zeros((len(val), 3))
    for cls, candidate in enumerate(selected):
        model = _make_model(candidate); model.fit(_text(train), (y_train == cls).astype(int)); scores = model.decision_function(_text(val))
        threshold, _ = _threshold((y_val == cls).astype(int), scores); thresholds.append(threshold); val_scores[:, cls] = scores - threshold
    bias = _bias(y_val, val_scores); base_val = np.argmax(val_scores + bias, axis=1)
    val_conf, val_pos = _rule_scores(_rule_text(train), y_train, _rule_text(val))
    best, grid = None, []
    for confidence in np.linspace(0.25, 0.95, 15):
        for support in [1, 2, 3, 4, 5]:
            pred = base_val.copy(); mask = (val_conf >= confidence) & (val_pos >= support)
            pred[mask] = 1; metric = _metrics(y_val, pred)
            row = {"min_confidence": float(confidence), "min_positive_support": support, "overrides": int(mask.sum()), **metric}; grid.append(row)
            if best is None or metric["macro_f1"] > best[2]["macro_f1"]: best = (float(confidence), support, metric)
    assert best is not None
    confidence, support, val_metrics = best
    development = pd.concat([train, val]); y_dev = development.merged_rank.to_numpy(); test_scores = np.zeros((len(test), 3))
    for cls, candidate in enumerate(selected):
        model = _make_model(candidate); model.fit(_text(development), (y_dev == cls).astype(int)); test_scores[:, cls] = model.decision_function(_text(test)) - thresholds[cls]
    test_pred = np.argmax(test_scores + bias, axis=1)
    test_conf, test_pos = _rule_scores(_rule_text(development), y_dev, _rule_text(test)); mask = (test_conf >= confidence) & (test_pos >= support); test_pred[mask] = 1
    test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame(grid).sort_values("macro_f1", ascending=False).to_csv(out / "e3_rule_validation_grid.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"original_label": test.openai_label.to_numpy(), "y_true": y_test, "y_pred": test_pred, "e3_rule_override": mask}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"target_definition": "E0 vs E3 vs E12", "test_blind_selection": True, "method_warning": "Exact phrase recurrence can occur across group-disjoint roles; this is reported in leakage warnings.", "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected_e3_gate": {"min_confidence": confidence, "min_positive_support": support, "test_overrides": int(mask.sum())}, "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics, "test_overrides": int(mask.sum())}, indent=2))


if __name__ == "__main__": main()
