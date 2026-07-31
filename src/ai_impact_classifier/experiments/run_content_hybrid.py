"""Test-blind sparse-text hybrid search for AI impact classification.

This runner uses validation data only for model and ensemble selection.  The
held-out test split is evaluated once, after the chosen configuration is
refitted on train plus validation rows.  It intentionally never includes
``ai_impact_score`` or other target-derived fields in the feature matrix.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

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
    parser = argparse.ArgumentParser(description="Run a test-blind sparse NLP hybrid search.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/content_hybrid")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _column_vectorizer(
    *, analyzer: str = "word", ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, max_features: int = 250_000
) -> Pipeline:
    return Pipeline(
        [
            ("flatten", FunctionTransformer(lambda x: np.asarray(x).ravel().astype(str), validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class SparseConfig:
    name: str
    c_value: float
    class_weight: Mapping[int, float] | str
    weights: Mapping[str, float]
    include_char: bool


def _make_model(config: SparseConfig) -> Pipeline:
    parts: list[tuple[str, Pipeline, str]] = [
        ("task_word", _column_vectorizer(max_features=280_000), "keytask_content"),
        ("role_word", _column_vectorizer(max_features=90_000), "jobrole_title"),
        ("ssoc_word", _column_vectorizer(max_features=70_000), "ssoc_title"),
        ("sector_word", _column_vectorizer(max_features=45_000), "sector_title"),
    ]
    if config.include_char:
        parts.append(
            ("task_char", _column_vectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=250_000), "keytask_content")
        )
    features = ColumnTransformer(parts, transformer_weights=dict(config.weights), sparse_threshold=0.3)
    return Pipeline([("features", features), ("classifier", LinearSVC(C=config.c_value, class_weight=config.class_weight, max_iter=20_000))])


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z -= z.max(axis=1, keepdims=True)
    out = np.exp(z)
    return out / out.sum(axis=1, keepdims=True)


def _best_bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # Small coordinate search: E0 is the reference and all selected offsets are frozen before test use.
    bias = np.zeros(scores.shape[1], dtype=float)
    best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        changed = False
        for cls in range(1, scores.shape[1]):
            old = bias[cls]
            for candidate in np.linspace(-1.5, 1.5, 25):
                trial = bias.copy()
                trial[cls] = candidate
                metric = f1_score(y, np.argmax(scores + trial, axis=1), average="macro")
                if metric > best:
                    best, bias[cls], changed = metric, candidate, True
            if not changed:
                bias[cls] = old
        if not changed:
            break
    return bias


def _best_blend(y: np.ndarray, score_mats: Sequence[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_weights = np.full(len(score_mats), 1.0 / len(score_mats))
    best_scores = sum(score_mats) / len(score_mats)
    best_metric = f1_score(y, np.argmax(best_scores, axis=1), average="macro")
    for _ in range(12_000):
        weights = rng.dirichlet(np.ones(len(score_mats)))
        scores = sum(weight * matrix for weight, matrix in zip(weights, score_mats))
        metric = f1_score(y, np.argmax(scores, axis=1), average="macro")
        if metric > best_metric:
            best_weights, best_scores, best_metric = weights, scores, metric
    return best_weights, best_scores, float(best_metric)


def _rule_e3_mask(frame: pd.DataFrame) -> np.ndarray:
    """Conservative creative-content gate, intentionally independent of labels at prediction time."""
    text = (
        frame["jobrole_title"].fillna("").astype(str)
        + " "
        + frame["ssoc_title"].fillna("").astype(str)
        + " "
        + frame["keytask_content"].fillna("").astype(str)
    ).str.lower()
    pattern = r"\b(animator|animation|3d artist|2d artist|video editor|photographer|visual merchand|illustrator|storyboard|graphic design|ux/ui|ui/ux|multimedia artist)\b"
    return text.str.contains(pattern, regex=True).to_numpy()


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
    y_train, y_val, y_test = (part["label_rank"].to_numpy() for part in (train, val, test))

    configs = [
        SparseConfig("balanced_words", 0.5, "balanced", {"task_word": 1, "role_word": 1, "ssoc_word": 1, "sector_word": 1}, False),
        SparseConfig("balanced_role2", 0.5, "balanced", {"task_word": 1, "role_word": 2, "ssoc_word": 1.5, "sector_word": 1}, False),
        SparseConfig("balanced_role4", 0.75, "balanced", {"task_word": 1, "role_word": 4, "ssoc_word": 2, "sector_word": 1}, False),
        SparseConfig("moderate_role2", 1.0, {0: 1, 1: 1.5, 2: 1.5, 3: 5}, {"task_word": 1, "role_word": 2, "ssoc_word": 1.5, "sector_word": 1}, False),
        SparseConfig("moderate_role4", 0.75, {0: 1, 1: 1.8, 2: 1.8, 3: 6}, {"task_word": 1, "role_word": 4, "ssoc_word": 2, "sector_word": 1}, False),
        SparseConfig("char_role2", 0.5, "balanced", {"task_word": 1, "role_word": 2, "ssoc_word": 1.5, "sector_word": 1, "task_char": 0.7}, True),
        SparseConfig("char_role4", 0.75, {0: 1, 1: 1.8, 2: 1.8, 3: 6}, {"task_word": 1, "role_word": 4, "ssoc_word": 2, "sector_word": 1, "task_char": 0.7}, True),
    ]

    fitted: dict[str, Pipeline] = {}
    val_margins: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for config in configs:
        model = _make_model(config)
        model.fit(train, y_train)
        margins = model.decision_function(val)
        bias = _best_bias(y_val, margins)
        pred = np.argmax(margins + bias, axis=1)
        fitted[config.name] = model
        val_margins[config.name] = margins
        rows.append({"candidate": config.name, "kind": "single", **_metrics(y_val, pred), "bias": json.dumps(bias.tolist())})

    # Pick the three independent sparse views before blending, then select only on validation.
    selected_names = [row["candidate"] for row in sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)[:3]]
    blend_inputs = [_softmax(val_margins[name], temperature=1.0) for name in selected_names]
    blend_weights, blend_val_probs, _ = _best_blend(y_val, blend_inputs, args.seed)
    blend_bias = _best_bias(y_val, blend_val_probs)
    blend_pred = np.argmax(blend_val_probs + blend_bias, axis=1)
    rows.append({"candidate": "sparse_probability_blend", "kind": "blend", **_metrics(y_val, blend_pred), "bias": json.dumps(blend_bias.tolist()), "weights": json.dumps(dict(zip(selected_names, blend_weights.tolist())))})

    # Rule gating is tested only as a validation-time override.  It must improve validation F1 to be retained.
    e3_rank = bundle.label_to_rank["E3"]
    gated_pred = blend_pred.copy()
    gated_pred[_rule_e3_mask(val)] = e3_rank
    gated_metrics = _metrics(y_val, gated_pred)
    use_gate = gated_metrics["macro_f1"] > _metrics(y_val, blend_pred)["macro_f1"]
    rows.append({"candidate": "sparse_blend_e3_gate", "kind": "rule_hybrid", **gated_metrics, "selected": use_gate})

    # Refit precisely the validation-selected base models on train+validation; test remains unused until here.
    development = pd.concat([train, val])
    y_development = development["label_rank"].to_numpy()
    test_probs = []
    for name in selected_names:
        config = next(item for item in configs if item.name == name)
        model = _make_model(config)
        model.fit(development, y_development)
        test_probs.append(_softmax(model.decision_function(test), temperature=1.0))
    final_probs = sum(weight * scores for weight, scores in zip(blend_weights, test_probs))
    final_pred = np.argmax(final_probs + blend_bias, axis=1)
    if use_gate:
        final_pred[_rule_e3_mask(test)] = e3_rank
    final_metrics = _metrics(y_test, final_pred)

    pd.DataFrame(rows).sort_values("macro_f1", ascending=False).to_csv(output_dir / "validation_leaderboard.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, final_pred, labels=range(len(bundle.label_order))), index=bundle.label_order, columns=bundle.label_order).to_csv(output_dir / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": final_pred}).to_csv(output_dir / "test_predictions.csv", index=False)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "test_blind_selection": True,
                "forbidden_input_columns": ["ai_impact_score", "ai_impact_category", "openai_label", "label_rank"],
                "selected_base_models": selected_names,
                "blend_weights": dict(zip(selected_names, blend_weights.tolist())),
                "blend_bias": blend_bias.tolist(),
                "e3_rule_gate_selected": use_gate,
                "test_metrics": final_metrics,
                "leakage_passed": leak.passed,
                "leakage_messages": leak.messages,
                "leakage_warnings": leak.warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"validation_candidates": rows, "test_metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
