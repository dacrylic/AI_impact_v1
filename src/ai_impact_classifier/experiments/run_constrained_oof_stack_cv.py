"""Nested five-fold, group-held-out validation for the constrained sparse stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_constrained_oof_stack import (
    NAMES,
    _best_bias,
    _fit_base_features,
    _metrics,
    _oof_base_features,
)


SPECS = [("task_only", 0.25), ("task_only", 0.75), ("title_task", 0.25), ("title_task", 0.75)]
META_C_VALUES = (0.02, 0.05, 0.1, 0.25, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested 5-fold group CV for the constrained OOF sparse stack.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/constrained_oof_stack_cv5")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _meta(c_value: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=c_value, max_iter=5000, n_jobs=1)),
    ])


def _select_meta(train: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[float, np.ndarray, dict[str, float]]:
    """Select combiner regularization and class offsets without seeing an outer fold."""
    groups = train[GROUP_COLUMN].astype(str).to_numpy()
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(inner.split(train, y, groups))
    fit, val = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
    y_fit, y_val = y[fit_idx], y[val_idx]
    oof_fit = _oof_base_features(fit.reset_index(drop=True), y_fit, seed, SPECS)
    val_features = _fit_base_features(fit, y_fit, val, SPECS)
    selected: tuple[float, np.ndarray, dict[str, float]] | None = None
    for c_value in META_C_VALUES:
        model = _meta(c_value)
        model.fit(oof_fit, y_fit)
        scores = model.predict_proba(val_features)
        bias = _best_bias(y_val, scores)
        metrics = _metrics(y_val, np.argmax(scores + bias, axis=1))
        if selected is None or metrics["macro_f1"] > selected[2]["macro_f1"]:
            selected = (c_value, bias, metrics)
    assert selected is not None
    return selected


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    y = frame.label_rank.to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    predictions = np.full(len(frame), -1, dtype=int)
    rows: list[dict[str, object]] = []

    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development = frame.iloc[development_idx].copy()
        test = frame.iloc[test_idx].copy()
        y_development, y_test = y[development_idx], y[test_idx]
        c_value, bias, inner_metrics = _select_meta(development, y_development, args.seed + fold)
        oof_development = _oof_base_features(development.reset_index(drop=True), y_development, args.seed + fold, SPECS)
        test_features = _fit_base_features(development, y_development, test, SPECS)
        model = _meta(c_value)
        model.fit(oof_development, y_development)
        test_pred = np.argmax(model.predict_proba(test_features) + bias, axis=1)
        predictions[test_idx] = test_pred
        rows.append({
            "fold": fold,
            "train_rows": len(development),
            "test_rows": len(test),
            "train_groups": development[GROUP_COLUMN].nunique(),
            "test_groups": test[GROUP_COLUMN].nunique(),
            "shared_groups": len(set(development[GROUP_COLUMN]) & set(test[GROUP_COLUMN])),
            "selected_meta_c": c_value,
            "bias": json.dumps(bias.tolist()),
            "inner_validation_macro_f1": inner_metrics["macro_f1"],
            **_metrics(y_test, test_pred),
        })
        print(json.dumps(rows[-1]), flush=True)

    if (predictions < 0).any():
        raise RuntimeError("At least one row did not receive an outer-fold prediction.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({"row_index": np.arange(len(frame)), "jobrole_id": groups, "y_true": y, "y_pred": predictions}).to_csv(out / "oof_predictions.csv", index=False)
    aggregate = _metrics(y, predictions)
    matrix = confusion_matrix(y, predictions, labels=range(len(NAMES)))
    pd.DataFrame(matrix, index=NAMES, columns=NAMES).to_csv(out / "oof_confusion_matrix.csv")
    payload = {
        "method": "nested group-stratified 5-fold cross-validation",
        "target_definition": "E0 vs E1 vs E23 (E23 merges raw E2 and E3)",
        "allowed_input_columns": ["keytask_content", "jobrole_title"],
        "outer_split": "StratifiedGroupKFold(n_splits=5) grouped by jobrole_id",
        "inner_selection": "group-stratified fold within each outer-development partition",
        "base_models": [{"text_view": view, "c": c_value} for view, c_value in SPECS],
        "aggregate_oof_metrics": aggregate,
        "all_outer_folds_have_zero_shared_jobrole_id": bool((pd.DataFrame(rows)["shared_groups"] == 0).all()),
    }
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
