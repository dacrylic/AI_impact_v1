"""Test-blind task classifier with unlabeled within-role textual context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a text-only task model augmented with sibling-task context.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/role_context")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    task = prepared.keytask_content.fillna("").astype(str).str.strip()
    # The profile is constructed independently inside each split and contains only raw task text.
    contexts = task.groupby(prepared[GROUP_COLUMN]).agg(lambda values: " [ROLE_TASK] ".join(pd.unique(values)[:40]))
    prepared["role_context"] = prepared[GROUP_COLUMN].map(contexts).fillna("")
    return prepared


def _column_vectorizer(*, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, max_features: int = 250_000) -> Pipeline:
    return Pipeline(
        [
            ("flatten", FunctionTransformer(lambda x: np.asarray(x).ravel().astype(str), validate=False)),
            ("tfidf", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_features=max_features, sublinear_tf=True, strip_accents="unicode")),
        ]
    )


def _model(c_value: float, class_weight: dict[int, float] | str, context_weight: float) -> Pipeline:
    features = ColumnTransformer(
        [
            ("task_word", _column_vectorizer(ngram_range=(1, 3), max_features=350_000), "keytask_content"),
            ("task_char", _column_vectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000), "keytask_content"),
            ("role", _column_vectorizer(max_features=100_000), "jobrole_title"),
            ("occupation", _column_vectorizer(max_features=80_000), "ssoc_title"),
            ("sector", _column_vectorizer(max_features=50_000), "sector_title"),
            ("profile", _column_vectorizer(ngram_range=(1, 2), max_features=450_000), "role_context"),
        ],
        transformer_weights={"task_word": 1.0, "task_char": 0.65, "role": 1.3, "occupation": 1.1, "sector": 0.8, "profile": context_weight},
        sparse_threshold=0.3,
    )
    return Pipeline([("features", features), ("classifier", LinearSVC(C=c_value, class_weight=class_weight, max_iter=20_000))])


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
    train, val, test = (_prepare(bundle.frame.loc[idx]) for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    settings = [(0.5, "balanced", 0.25), (0.5, "balanced", 0.5), (0.75, "balanced", 1.0), (0.75, "balanced", 1.5), (0.75, {0: 1, 1: 3, 2: 2, 3: 2}, 0.5), (1.0, {0: 1, 1: 4, 2: 2, 3: 2}, 1.0)]
    candidates = []
    for c_value, weight, context_weight in settings:
        model = _model(c_value, weight, context_weight); model.fit(train, y_train)
        scores = model.decision_function(val); bias = _bias(y_val, scores); pred = np.argmax(scores + bias, axis=1)
        candidates.append((c_value, weight, context_weight, bias, _metrics(y_val, pred)))
    c_value, weight, context_weight, bias, val_metrics = max(candidates, key=lambda item: item[4]["macro_f1"])
    development = _prepare(pd.concat([bundle.frame.loc[split.train_idx], bundle.frame.loc[split.val_idx]]))
    model = _model(c_value, weight, context_weight); model.fit(development, development.label_rank.to_numpy())
    test_pred = np.argmax(model.decision_function(test) + bias, axis=1); test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame([{"c": c, "class_weight": str(w), "context_weight": cw, "bias": json.dumps(b.tolist()), **m} for c, w, cw, b, m in candidates]).to_csv(out / "validation_leaderboard.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"test_blind_selection": True, "feature_contract": "current task text plus unlabeled sibling task text within the same role; group IDs are used solely to aggregate raw text within each isolated split", "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"], "selected": {"c": c_value, "class_weight": weight, "context_weight": context_weight, "bias": bias.tolist()}, "validation_metrics": val_metrics, "test_metrics": test_metrics, "leakage_passed": leak.passed, "leakage_warnings": leak.warnings}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__": main()
