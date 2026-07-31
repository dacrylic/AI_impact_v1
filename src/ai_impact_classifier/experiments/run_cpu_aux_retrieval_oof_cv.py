"""CPU-only nested CV with auxiliary raw labels, E3 retrieval, and soft specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _best_bias, _fit_base_features, _metrics, _oof_base_features
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


CANONICAL_SPECS = [("task_only", 0.25), ("task_only", 0.75), ("title_task", 0.25), ("title_task", 0.75)]
RAW_ORDER = ["E0", "E3", "E2", "E1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested CPU CV for auxiliary-label and E3-retrieval ensemble.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/cpu_aux_retrieval_oof_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _raw_svm(c_value: float) -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
        ], transformer_weights={"word": 1.0, "char": 0.65})),
        ("classifier", LinearSVC(C=c_value, class_weight="balanced", max_iter=20_000)),
    ])


def _raw_scores(fit: pd.DataFrame, raw_y: np.ndarray, evaluation: pd.DataFrame) -> np.ndarray:
    output = np.zeros((len(evaluation), 8))
    for index, c_value in enumerate((0.25, 0.75)):
        model = _raw_svm(c_value)
        model.fit(_allowed_text(fit, "title_task"), raw_y)
        output[:, index * 4:(index + 1) * 4] = model.decision_function(_allowed_text(evaluation, "title_task"))
    return output


def _e3_retrieval_and_specialist(fit: pd.DataFrame, y: np.ndarray, evaluation: pd.DataFrame) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=450_000, sublinear_tf=True, strip_accents="unicode")
    x_fit = vectorizer.fit_transform(_allowed_text(fit, "title_task")); x_eval = vectorizer.transform(_allowed_text(evaluation, "title_task"))
    e3 = x_fit[y == 1]
    # Sparse nearest-neighbour evidence from the accepted E3 teacher examples.
    similarities = x_eval @ e3.T
    max_e3 = similarities.max(axis=1).toarray().ravel()
    total_e3 = np.asarray(similarities.sum(axis=1)).ravel() / max(1, e3.shape[0])
    centroids = []
    for cls in range(3):
        centroid = np.asarray(x_fit[y == cls].mean(axis=0)).ravel()
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(np.asarray(x_eval @ centroid).ravel())
    specialist_scores = []
    binary = (y == 1).astype(int)
    for c_value, positive_weight in ((0.25, 2.0), (0.75, 4.0)):
        specialist = LinearSVC(C=c_value, class_weight={0: 1.0, 1: positive_weight}, max_iter=20_000)
        specialist.fit(x_fit, binary)
        specialist_scores.append(specialist.decision_function(x_eval))
    return np.column_stack([max_e3, total_e3, *centroids, *specialist_scores])


def _fit_features(fit: pd.DataFrame, y: np.ndarray, raw_y: np.ndarray, evaluation: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        _fit_base_features(fit, y, evaluation, CANONICAL_SPECS),
        _raw_scores(fit, raw_y, evaluation),
        _e3_retrieval_and_specialist(fit, y, evaluation),
    ])


def _oof_features(frame: pd.DataFrame, y: np.ndarray, raw_y: np.ndarray, seed: int) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy(); splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = None
    for fit_idx, holdout_idx in splitter.split(frame, y, groups):
        values = _fit_features(frame.iloc[fit_idx], y[fit_idx], raw_y[fit_idx], frame.iloc[holdout_idx])
        if output is None: output = np.zeros((len(frame), values.shape[1]))
        output[holdout_idx] = values
    assert output is not None
    return output


def _meta(c_value: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1))])


def _select(frame: pd.DataFrame, y: np.ndarray, raw_y: np.ndarray, seed: int) -> tuple[float, np.ndarray, dict[str, float]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(splitter.split(frame, y, frame[GROUP_COLUMN].astype(str)))
    fit, validation = frame.iloc[fit_idx], frame.iloc[val_idx]
    oof = _oof_features(fit.reset_index(drop=True), y[fit_idx], raw_y[fit_idx], seed)
    val_features = _fit_features(fit, y[fit_idx], raw_y[fit_idx], validation)
    selected = None
    for c_value in (0.005, 0.01, 0.02, 0.05):
        model = _meta(c_value); model.fit(oof, y[fit_idx])
        scores = model.predict_proba(val_features); bias = _best_bias(y[val_idx], scores)
        metrics = _metrics(y[val_idx], np.argmax(scores + bias, axis=1))
        if selected is None or metrics["macro_f1"] > selected[2]["macro_f1"]: selected = (c_value, bias, metrics)
    assert selected is not None
    return selected


def main() -> None:
    args = parse_args(); frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); raw_y = frame.openai_label.map({label: index for index, label in enumerate(RAW_ORDER)}).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy(); outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions = np.full(len(frame), -1, dtype=int); rows = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development, test = frame.iloc[development_idx], frame.iloc[test_idx]
        c_value, bias, inner = _select(development, y[development_idx], raw_y[development_idx], args.seed + fold)
        oof = _oof_features(development.reset_index(drop=True), y[development_idx], raw_y[development_idx], args.seed + fold)
        test_features = _fit_features(development, y[development_idx], raw_y[development_idx], test)
        model = _meta(c_value); model.fit(oof, y[development_idx])
        pred = np.argmax(model.predict_proba(test_features) + bias, axis=1); predictions[test_idx] = pred
        rows.append({"fold": fold, "selected_meta_c": c_value, "bias": json.dumps(bias.tolist()), "inner_validation_macro_f1": inner["macro_f1"], **_metrics(y[test_idx], pred)})
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": frame.index, "y_true": y, "y_pred": predictions}).to_csv(out / "oof_predictions.csv", index=False)
    payload = {"method": "CPU-only nested group CV: canonical sparse stack + raw E1/E2 auxiliary scores + E3 nearest-example retrieval + soft E3 specialists", "allowed_input_columns": ["keytask_content", "jobrole_title"], "raw_labels_used_as_training_only_auxiliary_targets": True, "aggregate_oof_metrics": _metrics(y, predictions)}
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
