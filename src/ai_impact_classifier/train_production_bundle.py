"""Train the final sparse classifier and OOF-calibrate production review flags."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import _base_model
from ai_impact_classifier.production import ReviewCalibration, ReviewThresholds, _combine_text, _known_ngram_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train production sparse model bundle with OOF review calibration.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="models/e0_e1_e23_sparse_svc.joblib")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _signals(model, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    text = _combine_text(frame)
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    word = features.transformer_list[0][1]
    char = features.transformer_list[1][1]
    scores = classifier.decision_function(features.transform(text))
    order = np.argsort(scores, axis=1)
    margin = scores[np.arange(len(frame)), order[:, -1]] - scores[np.arange(len(frame)), order[:, -2]]
    word_coverage = _known_ngram_coverage(word, text)
    char_coverage = _known_ngram_coverage(char, text)
    return np.argmax(scores, axis=1), margin, word_coverage, char_coverage, (word_coverage + char_coverage) / 2.0


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy(); groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    margin = np.full(len(frame), np.nan); word = np.full(len(frame), np.nan)
    char = np.full(len(frame), np.nan); total = np.full(len(frame), np.nan)
    oof_pred = np.full(len(frame), -1, dtype=int)
    for train_idx, holdout_idx in splitter.split(frame, y, groups):
        model = _base_model(0.5)
        model.fit(_combine_text(frame.iloc[train_idx]), y[train_idx])
        pred, fold_margin, fold_word, fold_char, fold_total = _signals(model, frame.iloc[holdout_idx])
        oof_pred[holdout_idx] = pred; margin[holdout_idx] = fold_margin; word[holdout_idx] = fold_word
        char[holdout_idx] = fold_char; total[holdout_idx] = fold_total
    if (oof_pred < 0).any() or not np.isfinite(margin).all():
        raise RuntimeError("Failed to obtain a complete OOF calibration set.")
    # Coverage cutoffs deliberately reserve a small tail for unfamiliar writing;
    # margin cutoff captures the riskiest tenth of familiar in-distribution rows.
    thresholds = ReviewThresholds(
        minimum_decision_margin=float(np.quantile(margin, 0.10)),
        minimum_word_coverage=float(np.quantile(word, 0.03)),
        minimum_char_coverage=float(np.quantile(char, 0.03)),
        minimum_total_coverage=float(np.quantile(total, 0.03)),
    )
    calibration = ReviewCalibration(
        thresholds=thresholds,
        margin_reference=tuple(np.sort(margin).tolist()),
        word_coverage_reference=tuple(np.sort(word).tolist()),
        char_coverage_reference=tuple(np.sort(char).tolist()),
        total_coverage_reference=tuple(np.sort(total).tolist()),
    )
    oof_review_score = np.maximum.reduce((
        1.0 - np.searchsorted(np.sort(margin), margin, side="right") / len(margin),
        1.0 - np.searchsorted(np.sort(word), word, side="right") / len(word),
        1.0 - np.searchsorted(np.sort(char), char, side="right") / len(char),
        1.0 - np.searchsorted(np.sort(total), total, side="right") / len(total),
    ))
    oof_high_review = (
        (margin < thresholds.minimum_decision_margin)
        | (word < thresholds.minimum_word_coverage)
        | (char < thresholds.minimum_char_coverage)
        | (total < thresholds.minimum_total_coverage)
        | (oof_review_score >= calibration.high_review_score)
    )
    final_model = _base_model(0.5)
    final_model.fit(_combine_text(frame), y)
    artifact = {
        "artifact_version": "1.0",
        "target_labels": ["E0", "E1", "E23"],
        "model": final_model,
        "review_calibration": calibration,
        "training_rows": len(frame),
        "oof_accuracy": float(np.mean(oof_pred == y)),
        "oof_review_rate": float(np.mean(oof_high_review)),
        "oof_review_error_rate": float(np.mean(oof_pred[oof_high_review] != y[oof_high_review])),
        "oof_nonreview_error_rate": float(np.mean(oof_pred[~oof_high_review] != y[~oof_high_review])),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print({
        "output": str(output),
        "oof_accuracy": artifact["oof_accuracy"],
        "oof_review_rate": artifact["oof_review_rate"],
        "oof_review_error_rate": artifact["oof_review_error_rate"],
        "oof_nonreview_error_rate": artifact["oof_nonreview_error_rate"],
        "thresholds": thresholds,
    })


if __name__ == "__main__":
    main()
