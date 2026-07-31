"""Leak-safe text-only classifier with E1 and E2 merged into E12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


MERGED_NAMES = ["E0", "E3", "E12"]
SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-only E0/E3/E12 classifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/merged_e12")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _text(frame: pd.DataFrame) -> pd.Series:
    return (
        "ROLE " + frame.jobrole_title.fillna("").astype(str)
        + " OCCUPATION " + frame.ssoc_title.fillna("").astype(str)
        + " SECTOR " + frame.sector_title.fillna("").astype(str)
        + " TASK " + frame.keytask_content.fillna("").astype(str)
    )


def _model(c_value: float, class_weight: dict[int, float] | str) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=400_000, sublinear_tf=True, strip_accents="unicode")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=350_000, sublinear_tf=True)),
                    ],
                    transformer_weights={"word": 1.0, "char": 0.65},
                ),
            ),
            ("classifier", LinearSVC(C=c_value, class_weight=class_weight, max_iter=20_000)),
        ]
    )


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    bias = np.zeros(scores.shape[1]); best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        changed = False
        for cls in range(1, scores.shape[1]):
            for value in np.linspace(-1.5, 1.5, 31):
                trial = bias.copy(); trial[cls] = value
                metric = f1_score(y, np.argmax(scores + trial, axis=1), average="macro")
                if metric > best: bias, best, changed = trial, metric, True
        if not changed: break
    return bias


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input); frame = bundle.frame.copy()
    frame["merged_rank"] = frame.openai_label.map({"E0": 0, "E3": 1, "E2": 2, "E1": 2}).astype(int)
    split = stratified_group_train_val_test_split(frame, GROUP_COLUMN, "merged_rank", random_state=args.seed)
    leak = check_split_leakage(frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.merged_rank.to_numpy() for part in (train, val, test))
    settings = [(0.25, "balanced"), (0.5, "balanced"), (0.75, "balanced"), (0.5, {0: 1, 1: 3, 2: 1}), (0.75, {0: 1, 1: 5, 2: 1}), (1.0, {0: 1, 1: 8, 2: 1})]
    candidates = []
    for c_value, weight in settings:
        model = _model(c_value, weight); model.fit(_text(train), y_train)
        scores = model.decision_function(_text(val)); bias = _bias(y_val, scores); pred = np.argmax(scores + bias, axis=1)
        candidates.append((c_value, weight, bias, _metrics(y_val, pred)))
    c_value, weight, bias, val_metrics = max(candidates, key=lambda item: item[3]["macro_f1"])
    development = pd.concat([train, val]); model = _model(c_value, weight); model.fit(_text(development), development.merged_rank.to_numpy())
    test_pred = np.argmax(model.decision_function(_text(test)) + bias, axis=1); test_metrics = _metrics(y_test, test_pred)
    split_summary = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        row = {"split": name, "rows": len(part), **{f"merged_{label}": int((part.merged_rank == rank).sum()) for rank, label in enumerate(MERGED_NAMES)}}
        row.update({f"original_{label}": int((part.openai_label == label).sum()) for label in ["E0", "E3", "E2", "E1"]})
        split_summary.append(row)
    pd.DataFrame(split_summary).to_csv(out / "split_summary.csv", index=False)
    pd.DataFrame([{"c": c, "class_weight": str(w), "bias": json.dumps(b.tolist()), **m} for c, w, b, m in candidates]).to_csv(out / "validation_leaderboard.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(3)), index=MERGED_NAMES, columns=MERGED_NAMES).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"original_label": test.openai_label.to_numpy(), "y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"target_definition": "E0 vs E3 vs E12, where E12 merges original E1 and E2", "test_blind_selection": True, "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected": {"c": c_value, "class_weight": weight, "bias": bias.tolist()}, "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__": main()
