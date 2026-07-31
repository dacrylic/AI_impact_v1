"""CPU-only nested group CV of sparse SVM, NB-SVM, and ComplementNB signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _best_bias, _metrics
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


NAMES = ["E0", "E3", "E12"]
SVM_SPECS = [("task_only", 0.25), ("task_only", 0.75), ("title_task", 0.25), ("title_task", 0.75)]
NB_VIEWS = ("task_only", "title_task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested CPU-only CV for classical sparse ensemble.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/cpu_classical_oof_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _svm() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
        ], transformer_weights={"word": 1.0, "char": 0.65})),
        ("classifier", LinearSVC(C=0.5, class_weight="balanced", max_iter=20_000)),
    ])


def _svm_scores(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    output = np.zeros((len(target), len(SVM_SPECS) * 3))
    for index, (view, c_value) in enumerate(SVM_SPECS):
        model = _svm(); model.set_params(classifier__C=c_value)
        model.fit(_allowed_text(train, view), y)
        output[:, index * 3:(index + 1) * 3] = model.decision_function(_allowed_text(target, view))
    return output


def _nb_ratio(matrix: csr_matrix, labels: np.ndarray) -> np.ndarray:
    positive = np.asarray(matrix[labels == 1].sum(axis=0)).ravel() + 1.0
    negative = np.asarray(matrix[labels == 0].sum(axis=0)).ravel() + 1.0
    return np.log(positive / positive.sum()) - np.log(negative / negative.sum())


def _nbsvm_scores(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame, view: str) -> np.ndarray:
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True, max_features=450_000, strip_accents="unicode")
    x_train = vectorizer.fit_transform(_allowed_text(train, view)); x_target = vectorizer.transform(_allowed_text(target, view))
    output = np.zeros((len(target), 3))
    for cls in range(3):
        binary = (y == cls).astype(int); ratio = _nb_ratio(x_train, binary)
        model = LinearSVC(C=0.25, class_weight="balanced", max_iter=20_000)
        model.fit(x_train.multiply(ratio), binary)
        output[:, cls] = model.decision_function(x_target.multiply(ratio))
    return output


def _complement_scores(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=400_000, sublinear_tf=True, strip_accents="unicode")
    x_train = vectorizer.fit_transform(_allowed_text(train, "title_task")); x_target = vectorizer.transform(_allowed_text(target, "title_task"))
    model = ComplementNB(alpha=0.25, norm=True); model.fit(x_train, y)
    return model.predict_log_proba(x_target)


def _fit_features(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        _svm_scores(train, y, target),
        _nbsvm_scores(train, y, target, "task_only"),
        _nbsvm_scores(train, y, target, "title_task"),
        _complement_scores(train, y, target),
    ])


def _oof_features(frame: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = None
    for fit_idx, holdout_idx in splitter.split(frame, y, groups):
        values = _fit_features(frame.iloc[fit_idx], y[fit_idx], frame.iloc[holdout_idx])
        if output is None:
            output = np.zeros((len(frame), values.shape[1]))
        output[holdout_idx] = values
    assert output is not None
    return output


def _meta(c_value: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1))])


def _select(frame: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[float, np.ndarray, dict[str, float]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(frame, y, frame[GROUP_COLUMN].astype(str)))
    fit, validation = frame.iloc[fit_idx], frame.iloc[val_idx]
    oof = _oof_features(fit.reset_index(drop=True), y[fit_idx], seed)
    validation_features = _fit_features(fit, y[fit_idx], validation)
    selected = None
    for c_value in (0.01, 0.02, 0.05, 0.1):
        model = _meta(c_value); model.fit(oof, y[fit_idx])
        scores = model.predict_proba(validation_features); bias = _best_bias(y[val_idx], scores)
        metrics = _metrics(y[val_idx], np.argmax(scores + bias, axis=1))
        if selected is None or metrics["macro_f1"] > selected[2]["macro_f1"]:
            selected = (c_value, bias, metrics)
    assert selected is not None
    return selected


def main() -> None:
    args = parse_args(); frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions = np.full(len(frame), -1, dtype=int); rows = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        c_value, bias, inner = _select(development, y[development_idx], args.seed + fold)
        oof = _oof_features(development.reset_index(drop=True), y[development_idx], args.seed + fold)
        test_features = _fit_features(development, y[development_idx], test)
        model = _meta(c_value); model.fit(oof, y[development_idx])
        pred = np.argmax(model.predict_proba(test_features) + bias, axis=1); predictions[test_idx] = pred
        rows.append({"fold": fold, "selected_meta_c": c_value, "bias": json.dumps(bias.tolist()), "inner_validation_macro_f1": inner["macro_f1"], **_metrics(y[test_idx], pred)})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": frame.index, "y_true": y, "y_pred": predictions}).to_csv(out / "oof_predictions.csv", index=False)
    payload = {"method": "CPU-only nested group CV: word/character LinearSVC + two NB-SVM views + ComplementNB", "allowed_input_columns": ["keytask_content", "jobrole_title"], "aggregate_oof_metrics": _metrics(y, predictions)}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
