"""Validation-selected one-vs-rest text specialists.

Each label gets its own binary sparse classifier and threshold.  This lets the
rare E3 class use a different decision boundary from the much larger E0 class.
The held-out test set is not accessed until the selected specialists are
refitted on train plus validation rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


@dataclass(frozen=True)
class Candidate:
    name: str
    mode: str
    c_value: float
    positive_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-blind one-vs-rest sparse specialists.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/specialist_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-gates", action="store_true", help="Enable validation-selected specialist overrides (off by default).")
    return parser.parse_args()


def _text(frame: pd.DataFrame) -> pd.Series:
    # Prefixes preserve field identity while exposing every input as text only.
    return (
        "ROLE " + frame["jobrole_title"].fillna("").astype(str)
        + " OCCUPATION " + frame["ssoc_title"].fillna("").astype(str)
        + " SECTOR " + frame["sector_title"].fillna("").astype(str)
        + " TASK " + frame["keytask_content"].fillna("").astype(str)
    )


def _make_model(candidate: Candidate) -> Pipeline:
    long = candidate.mode in {"word_long", "char_word_long"}
    word = TfidfVectorizer(
        ngram_range=(1, 4) if long else (1, 3),
        min_df=2,
        max_features=450_000 if long else 350_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if candidate.mode in {"word", "word_long"}:
        features = word
    elif candidate.mode in {"char_word", "char_word_long"}:
        features = FeatureUnion(
            [
                ("word", word),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 6) if long else (3, 5),
                        min_df=3,
                        max_features=400_000 if long else 300_000,
                        sublinear_tf=True,
                    ),
                ),
            ],
            transformer_weights={"word": 1.0, "char": 0.65},
        )
    else:
        raise ValueError(candidate.mode)
    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(C=candidate.c_value, class_weight={0: 1.0, 1: candidate.positive_weight}, max_iter=20_000)),
        ]
    )


def _best_binary_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold, best_f1 = 0.0, -1.0
    for threshold in np.linspace(-2.5, 2.5, 101):
        value = f1_score(y, scores >= threshold, zero_division=0)
        if value > best_f1:
            best_threshold, best_f1 = float(threshold), float(value)
    return best_threshold, best_f1


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
    }


def _best_bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    bias = np.zeros(scores.shape[1], dtype=float)
    best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        improved = False
        for cls in range(scores.shape[1]):
            for value in np.linspace(-1.0, 1.0, 41):
                trial = bias.copy()
                trial[cls] = value
                score = f1_score(y, np.argmax(scores + trial, axis=1), average="macro")
                if score > best:
                    bias, best, improved = trial, score, True
        if not improved:
            break
    return bias


def _select_gates(y: np.ndarray, scores: np.ndarray, base_pred: np.ndarray) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Add only validation-improving high-confidence specialist overrides."""
    pred = base_pred.copy()
    selected: list[tuple[int, float]] = []
    best = f1_score(y, pred, average="macro")
    # E3 first because it is rare and has a specialist with a substantially different boundary.
    for cls in [1, 2, 3, 0]:
        candidate_threshold: float | None = None
        candidate_pred = pred
        for threshold in np.linspace(-0.5, 1.5, 81):
            trial = pred.copy()
            trial[scores[:, cls] >= threshold] = cls
            metric = f1_score(y, trial, average="macro")
            if metric > best:
                candidate_threshold, candidate_pred, best = float(threshold), trial, metric
        if candidate_threshold is not None:
            pred = candidate_pred
            selected.append((cls, candidate_threshold))
    return pred, selected


def main() -> None:
    args = parse_args()
    bundle = build_bundle(args.input)
    split = stratified_group_train_val_test_split(bundle.frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(
        bundle.frame,
        split_assignments={"train": split.train_idx, "val": split.val_idx, "test": split.test_idx},
        group_column=GROUP_COLUMN,
        text_column="target_text",
        soft_id_columns=SOFT_ID_COLUMNS,
    )
    if not leak.passed:
        raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = (bundle.frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    train_text, val_text, test_text = (_text(part) for part in (train, val, test))
    y_train, y_val, y_test = (part["label_rank"].to_numpy() for part in (train, val, test))

    candidates = [
        Candidate("word_c025_w2", "word", 0.25, 2.0), Candidate("word_c05_w4", "word", 0.5, 4.0),
        Candidate("word_c1_w8", "word", 1.0, 8.0), Candidate("word_c1_w16", "word", 1.0, 16.0),
        Candidate("char_c025_w2", "char_word", 0.25, 2.0), Candidate("char_c05_w4", "char_word", 0.5, 4.0),
        Candidate("char_c075_w8", "char_word", 0.75, 8.0), Candidate("char_c1_w16", "char_word", 1.0, 16.0),
        Candidate("wordlong_c025_w2", "word_long", 0.25, 2.0), Candidate("wordlong_c05_w4", "word_long", 0.5, 4.0),
        Candidate("wordlong_c1_w8", "word_long", 1.0, 8.0),
        Candidate("charlong_c025_w2", "char_word_long", 0.25, 2.0), Candidate("charlong_c05_w4", "char_word_long", 0.5, 4.0),
        Candidate("charlong_c075_w8", "char_word_long", 0.75, 8.0), Candidate("charlong_c1_w16", "char_word_long", 1.0, 16.0),
    ]
    selected: list[Candidate] = []
    selected_thresholds: list[float] = []
    selection_rows: list[dict[str, object]] = []
    val_scores = np.zeros((len(val), len(bundle.label_order)), dtype=float)

    for cls, label in enumerate(bundle.label_order):
        binary_train = (y_train == cls).astype(int)
        binary_val = (y_val == cls).astype(int)
        best: tuple[Candidate, np.ndarray, float, float] | None = None
        for candidate in candidates:
            model = _make_model(candidate)
            model.fit(train_text, binary_train)
            scores = model.decision_function(val_text)
            threshold, class_f1 = _best_binary_threshold(binary_val, scores)
            selection_rows.append({"label": label, "candidate": candidate.name, "threshold": threshold, "binary_val_f1": class_f1})
            if best is None or class_f1 > best[3]:
                best = (candidate, scores, threshold, class_f1)
        assert best is not None
        candidate, scores, threshold, class_f1 = best
        # Center at the validation-selected binary boundary; this makes specialist scores comparable.
        val_scores[:, cls] = scores - threshold
        selected.append(candidate)
        selected_thresholds.append(threshold)

    bias = _best_bias(y_val, val_scores)
    val_pred = np.argmax(val_scores + bias, axis=1)
    gates: list[tuple[int, float]] = []
    if args.enable_gates:
        val_pred, gates = _select_gates(y_val, val_scores, val_pred)
    val_metrics = _metrics(y_val, val_pred)

    # Test is used for the first time here. Refit selected specialists on train+validation and preserve all selected settings.
    development = pd.concat([train, val])
    development_text = _text(development)
    y_development = development["label_rank"].to_numpy()
    test_scores = np.zeros((len(test), len(bundle.label_order)), dtype=float)
    for cls, (candidate, threshold) in enumerate(zip(selected, selected_thresholds)):
        model = _make_model(candidate)
        model.fit(development_text, (y_development == cls).astype(int))
        test_scores[:, cls] = model.decision_function(test_text) - threshold
    test_pred = np.argmax(test_scores + bias, axis=1)
    if args.enable_gates:
        for cls, threshold in gates:
            test_pred[test_scores[:, cls] >= threshold] = cls
    test_metrics = _metrics(y_test, test_pred)

    pd.DataFrame(selection_rows).to_csv(output_dir / "binary_validation_search.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(output_dir / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "test_blind_selection": True,
                "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank", "task_id", "jobrole_id", "ssoc_code"],
                "selected_by_label": {label: {"candidate": candidate.name, "threshold": threshold} for label, candidate, threshold in zip(bundle.label_order, selected, selected_thresholds)},
                "multiclass_bias": bias.tolist(),
                "specialist_gates": {bundle.label_order[cls]: threshold for cls, threshold in gates},
                "specialist_gates_enabled": args.enable_gates,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "leakage_passed": leak.passed,
                "leakage_warnings": leak.warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"validation_metrics": val_metrics, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
