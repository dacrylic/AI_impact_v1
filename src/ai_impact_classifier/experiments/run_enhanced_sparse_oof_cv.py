"""Nested group CV for sparse stack enriched with rule scores and text interactions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _best_bias, _metrics
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


NAMES = ["E0", "E3", "E12"]
WORD_RE = re.compile(r"[a-z0-9]{2,}")
KEYWORDS = frozenset({
    "creative", "design", "designing", "illustration", "illustrative", "graphics", "graphic", "visual",
    "visualisation", "drawing", "drawings", "video", "animation", "wireframe", "mockup", "collateral",
    "collaterals", "storyboard", "photography", "image", "images", "layout", "editing",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested group CV for enhanced sparse E3-aware stack.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/enhanced_sparse_oof_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tokens(value: str, limit: int) -> list[str]:
    return [token for token in WORD_RE.findall(value.lower()) if token not in ENGLISH_STOP_WORDS][:limit]


def _interaction_text(frame: pd.DataFrame) -> pd.Series:
    """Encode lexical title-task interactions without introducing non-text columns."""
    output = []
    for title, task in zip(frame.jobrole_title.fillna(""), frame.keytask_content.fillna("")):
        role_words = _tokens(str(title), 6)
        task_words = _tokens(str(task), 18)
        output.append(" ".join(f"R_{role}__T_{task_word}" for role in role_words for task_word in task_words))
    return pd.Series(output, index=frame.index)


def _standard_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
        ], transformer_weights={"word": 1.0, "char": 0.65})),
        ("classifier", LinearSVC(C=c_value, class_weight="balanced", max_iter=20_000)),
    ])


def _interaction_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_features=400_000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("classifier", LinearSVC(C=c_value, class_weight="balanced", max_iter=20_000)),
    ])


SPECS = [("task_only", 0.25), ("task_only", 0.75), ("title_task", 0.25), ("title_task", 0.75), ("interaction", 0.25), ("interaction", 0.75)]


def _text(frame: pd.DataFrame, view: str) -> pd.Series:
    return _interaction_text(frame) if view == "interaction" else _allowed_text(frame, view)


def _model(view: str, c_value: float) -> Pipeline:
    return _interaction_model(c_value) if view == "interaction" else _standard_model(c_value)


def _fit_base_features(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    output = np.zeros((len(target), len(SPECS) * len(NAMES)))
    for index, (view, c_value) in enumerate(SPECS):
        model = _model(view, c_value)
        model.fit(_text(train, view), y)
        output[:, index * 3:(index + 1) * 3] = model.decision_function(_text(target, view))
    return output


def _oof_base_features(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = np.zeros((len(frame), len(SPECS) * len(NAMES)))
    for train_idx, holdout_idx in splitter.split(frame, y, groups):
        output[holdout_idx] = _fit_base_features(frame.iloc[train_idx], y[train_idx], frame.iloc[holdout_idx])
    return output


def _rule_features(fit: pd.DataFrame, y: np.ndarray, evaluation: pd.DataFrame) -> np.ndarray:
    vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=2, binary=True, max_features=500_000, strip_accents="unicode")
    x_fit = vectorizer.fit_transform(_allowed_text(fit, "title_task"))
    x_eval = vectorizer.transform(_allowed_text(evaluation, "title_task"))
    positive = np.asarray(x_fit[y == 1].sum(axis=0)).ravel()
    total = np.asarray(x_fit.sum(axis=0)).ravel()
    confidence = positive / np.maximum(total, 1)
    association_confidence = x_eval.multiply(confidence).max(axis=1).toarray().ravel()
    association_support = x_eval.multiply(positive).max(axis=1).toarray().ravel()
    keyword_count = np.array([
        sum(token in KEYWORDS for token in _tokens(text, 100))
        for text in _allowed_text(evaluation, "title_task")
    ])
    return np.column_stack([association_confidence, np.log1p(association_support), keyword_count])


def _oof_rule_features(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = np.zeros((len(frame), 3))
    for train_idx, holdout_idx in splitter.split(frame, y, groups):
        output[holdout_idx] = _rule_features(frame.iloc[train_idx], y[train_idx], frame.iloc[holdout_idx])
    return output


def _features_fit(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_fit_base_features(train, y, target), _rule_features(train, y, target)])


def _features_oof(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    return np.column_stack([_oof_base_features(frame, y, seed), _oof_rule_features(frame, y, seed)])


def _select_meta(development: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[float, np.ndarray, dict[str, float]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(development, y, development[GROUP_COLUMN].astype(str)))
    fit, validation = development.iloc[fit_idx], development.iloc[val_idx]
    oof = _features_oof(fit.reset_index(drop=True), y[fit_idx], seed)
    val_features = _features_fit(fit, y[fit_idx], validation)
    choice = None
    for c_value in (0.01, 0.02, 0.05, 0.1):
        meta = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1))])
        meta.fit(oof, y[fit_idx])
        scores = meta.predict_proba(val_features)
        bias = _best_bias(y[val_idx], scores)
        metrics = _metrics(y[val_idx], np.argmax(scores + bias, axis=1))
        if choice is None or metrics["macro_f1"] > choice[2]["macro_f1"]:
            choice = (c_value, bias, metrics)
    assert choice is not None
    return choice


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions = np.full(len(frame), -1, dtype=int); rows = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        c_value, bias, inner = _select_meta(development, y[development_idx], args.seed + fold)
        oof = _features_oof(development.reset_index(drop=True), y[development_idx], args.seed + fold)
        test_features = _features_fit(development, y[development_idx], test)
        meta = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1))])
        meta.fit(oof, y[development_idx])
        pred = np.argmax(meta.predict_proba(test_features) + bias, axis=1)
        predictions[test_idx] = pred
        rows.append({"fold": fold, "selected_meta_c": c_value, "bias": json.dumps(bias.tolist()), "inner_validation_macro_f1": inner["macro_f1"], **_metrics(y[test_idx], pred)})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": frame.index, "y_true": y, "y_pred": predictions}).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(__import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y, predictions, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "oof_confusion_matrix.csv")
    payload = {"method": "nested group CV: sparse base stack + title-task interaction features + cross-fitted E3 rule scores", "allowed_input_columns": ["keytask_content", "jobrole_title"], "aggregate_oof_metrics": _metrics(y, predictions), "base_specs": SPECS, "meta_rule_features": ["association_confidence", "log_association_support", "keyword_count"]}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
