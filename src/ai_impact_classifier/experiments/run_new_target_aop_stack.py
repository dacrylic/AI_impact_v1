"""Leakage-audited action/object/purpose holdout stack for the new target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ai_impact_classifier.experiments.run_new_target_sparse_cv import LABELS, TARGET_MAP, clean_text


VIEWS = ("action", "object", "purpose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/new_target_aop_stack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--max-word-features", type=int, default=100_000)
    parser.add_argument("--max-char-features", type=int, default=100_000)
    return parser.parse_args()


def load_frame(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path, low_memory=False).copy()
    required = {"jobrole_id", "action", "object", "purpose", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    frame = frame.dropna(subset=["jobrole_id", "label"]).reset_index(drop=True)
    frame["target"] = frame["label"].map(TARGET_MAP)
    if frame["target"].isna().any():
        raise ValueError("Input contains unsupported source labels")
    for view in VIEWS:
        frame[view] = clean_text(frame[view])
    frame["aop_key"] = (
        "ACTION: " + frame["action"].str.lower() + " OBJECT: " + frame["object"].str.lower() + " PURPOSE: " + frame["purpose"].str.lower()
    )
    target_counts = frame.groupby("aop_key")["target"].nunique()
    conflicting = target_counts[target_counts > 1].index
    audit = {
        "input_rows": len(frame),
        "input_unique_aop_tuples": int(frame["aop_key"].nunique()),
        "excluded_conflicting_aop_tuples": int(len(conflicting)),
        "excluded_rows_with_conflicting_aop_tuple": int(frame["aop_key"].isin(conflicting).sum()),
    }
    frame = (
        frame.loc[~frame["aop_key"].isin(conflicting)]
        .sort_values(["aop_key", "jobrole_id"], kind="stable")
        .drop_duplicates("aop_key")
        .reset_index(drop=True)
    )
    audit["model_rows"] = len(frame)
    audit["model_unique_aop_tuples"] = int(frame["aop_key"].nunique())
    return frame, audit


def feature_matrices(
    train_text: pd.Series, test_text: pd.Series, max_word_features: int, max_char_features: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=max_word_features, sublinear_tf=True, strip_accents="unicode", dtype=np.float32
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_char_features, sublinear_tf=True, dtype=np.float32
    )
    x_train = sparse.hstack([word.fit_transform(train_text), 0.65 * char.fit_transform(train_text)], format="csr")
    x_test = sparse.hstack([word.transform(test_text), 0.65 * char.transform(test_text)], format="csr")
    return x_train, x_test


def aligned_scores(model: LinearSVC, x: sparse.csr_matrix) -> np.ndarray:
    scores = np.full((x.shape[0], len(LABELS)), 0.0, dtype=np.float32)
    decision = np.clip(np.nan_to_num(model.decision_function(x), nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    for source_index, label in enumerate(model.classes_):
        scores[:, LABELS.index(label)] = decision[:, source_index]
    return scores


def fit_specialists(
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    score_idx: np.ndarray,
    max_word_features: int,
    max_char_features: int,
    seed: int,
) -> np.ndarray:
    y_fit = frame.iloc[fit_idx]["target"].to_numpy()
    view_scores: list[np.ndarray] = []
    for view in VIEWS:
        x_fit, x_score = feature_matrices(
            frame.iloc[fit_idx][view], frame.iloc[score_idx][view], max_word_features, max_char_features
        )
        model = LinearSVC(C=1.0, class_weight="balanced", dual=True, random_state=seed)
        model.fit(x_fit, y_fit)
        view_scores.append(aligned_scores(model, x_score))
    purpose_present = (frame.iloc[score_idx]["purpose"] != "").to_numpy(dtype=np.float32)[:, None]
    return np.hstack([*view_scores, purpose_present])


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rank = {label: index for index, label in enumerate(LABELS)}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score([rank[label] for label in y_true], [rank[label] for label in y_pred], weights="quadratic")
        ),
    }


def main() -> None:
    args = parse_args()
    frame, data_audit = load_frame(args.input)
    y = frame["target"].to_numpy()
    groups = frame["jobrole_id"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    outer_splits = list(outer.split(frame, y, groups))
    if args.fold_index is not None and not 1 <= args.fold_index <= args.folds:
        raise ValueError(f"fold-index must be between 1 and {args.folds}")
    selected_splits = (
        [(args.fold_index, outer_splits[args.fold_index - 1])]
        if args.fold_index is not None
        else list(enumerate(outer_splits, start=1))
    )
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold, (development_idx, test_idx) in selected_splits:
        development = frame.iloc[development_idx]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed + fold)
        base_idx_relative, meta_idx_relative = next(
            inner.split(development, development["target"].to_numpy(), development["jobrole_id"].astype(str).to_numpy())
        )
        base_idx, meta_idx = development_idx[base_idx_relative], development_idx[meta_idx_relative]
        shared_outer_roles = len(set(groups[development_idx]) & set(groups[test_idx]))
        shared_outer_inputs = len(set(frame.iloc[development_idx]["aop_key"]) & set(frame.iloc[test_idx]["aop_key"]))
        shared_inner_roles = len(set(groups[base_idx]) & set(groups[meta_idx]))
        if shared_outer_roles or shared_outer_inputs or shared_inner_roles:
            raise RuntimeError("A/O/P stack split leakage check failed")
        x_meta = fit_specialists(frame, base_idx, meta_idx, args.max_word_features, args.max_char_features, args.seed + fold)
        meta_model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=10.0, class_weight="balanced", solver="lsqr"),
        )
        meta_model.fit(x_meta.astype(np.float64), y[meta_idx])
        x_test = fit_specialists(
            frame, development_idx, test_idx, args.max_word_features, args.max_char_features, args.seed + 10 + fold
        )
        # The small dense meta matrix and fitted coefficients are finite; suppress
        # spurious BLAS floating-point flags emitted by the local macOS runtime.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            prediction = meta_model.predict(x_test.astype(np.float64))
        fold_rows.append(
            {
                "fold": fold,
                "development_rows": len(development_idx),
                "meta_fit_rows": len(meta_idx),
                "test_rows": len(test_idx),
                "shared_outer_jobrole_ids": shared_outer_roles,
                "shared_outer_aop_tuples": shared_outer_inputs,
                "shared_inner_jobrole_ids": shared_inner_roles,
                **metric_row(y[test_idx], prediction),
            }
        )
        prediction_rows.append(
            pd.DataFrame({"row_index": test_idx, "fold": fold, "jobrole_id": groups[test_idx], "target": y[test_idx], "prediction": prediction})
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "oof_predictions.csv", index=False)
    metadata = {
        "dataset": str(args.input),
        "data_audit": data_audit,
        "target_mapping": TARGET_MAP,
        "base_views": list(VIEWS),
        "meta_features": [f"{view}_{label}_margin" for view in VIEWS for label in LABELS] + ["purpose_present"],
        "splitter": "outer 5-fold StratifiedGroupKFold(jobrole_id); one inner group-disjoint meta-fit holdout",
        "executed_fold_indices": [fold for fold, _ in selected_splits],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(fold_rows).to_string(index=False))


if __name__ == "__main__":
    main()
