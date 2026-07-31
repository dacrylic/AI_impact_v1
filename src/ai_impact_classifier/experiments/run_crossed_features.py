"""Test-blind role/occupation x task lexical interaction classifier."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


TOKEN_RE = re.compile(r"[a-z][a-z0-9+/-]{1,}")
STOP = set(ENGLISH_STOP_WORDS) | {"task", "role", "occupation", "sector", "using", "use"}
SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-blind crossed sparse features.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/crossed_features")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tokens(text: str, limit: int) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(text).lower()) if token not in STOP][:limit]


def _cross_doc(row: str) -> list[str]:
    # Row format: role\x1foccupation\x1fsector\x1ftask.
    role, occupation, sector, task = row.split("\x1f", 3)
    role_tokens = _tokens(role, 8) + _tokens(occupation, 8) + _tokens(sector, 5)
    task_tokens = _tokens(task, 45)
    features = [f"task={token}" for token in task_tokens]
    features += [f"context={token}" for token in role_tokens]
    # Cross only content-bearing tokens.  This yields compact context-sensitive lexical cues.
    for context in role_tokens:
        features.extend(f"cross={context}::{task_token}" for task_token in task_tokens)
    return features


def _packed(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.jobrole_title.fillna("").astype(str) + "\x1f"
        + frame.ssoc_title.fillna("").astype(str) + "\x1f"
        + frame.sector_title.fillna("").astype(str) + "\x1f"
        + frame.keytask_content.fillna("").astype(str)
    )


def _model(c_value: float, class_weight: dict[int, float] | str) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("cross", TfidfVectorizer(analyzer=_cross_doc, lowercase=False, min_df=2, max_features=700_000, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=250_000, sublinear_tf=True)),
                    ],
                    transformer_weights={"cross": 1.0, "char": 0.35},
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
                candidate = bias.copy(); candidate[cls] = value
                metric = f1_score(y, np.argmax(scores + candidate, axis=1), average="macro")
                if metric > best: bias, best, changed = candidate, metric, True
        if not changed: break
    return bias


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(bundle.frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    x_train, x_val, x_test = (_packed(part) for part in (train, val, test)); y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    settings = [(0.25, "balanced"), (0.5, "balanced"), (0.75, "balanced"), (0.5, {0: 1, 1: 3, 2: 2, 3: 2}), (0.75, {0: 1, 1: 4, 2: 2, 3: 2}), (1.0, {0: 1, 1: 5, 2: 2, 3: 2})]
    candidates = []
    for c_value, weight in settings:
        model = _model(c_value, weight); model.fit(x_train, y_train)
        scores = model.decision_function(x_val); bias = _bias(y_val, scores); pred = np.argmax(scores + bias, axis=1)
        candidates.append((c_value, weight, bias, _metrics(y_val, pred)))
    c_value, weight, bias, val_metrics = max(candidates, key=lambda item: item[3]["macro_f1"])
    development = pd.concat([train, val]); model = _model(c_value, weight); model.fit(_packed(development), development.label_rank.to_numpy())
    test_pred = np.argmax(model.decision_function(x_test) + bias, axis=1); test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame([{"c": c, "weight": str(w), "bias": json.dumps(b.tolist()), **m} for c, w, b, m in candidates]).to_csv(out / "validation_leaderboard.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"test_blind_selection": True, "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected": {"c": c_value, "class_weight": weight, "bias": bias.tolist()}, "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__": main()
