"""Train and package the approved GPT-5.2/current-guidance CPU classifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

from .contracts import CLASS_ORDER, MODEL_VERSION, SOURCE_TO_TARGET, compose_model_text, normalize_text
from .features import FEATURE_PROFILES, FieldAwareTfidfFeatures, TEXT_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Approved GPT-5.2 current-guidance labels CSV.")
    parser.add_argument("--output-dir", required=True, help="Release artifact directory.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fold", type=int, default=2, help="One-based role-held-out outer fold reserved as sealed test.")
    parser.add_argument("--c", type=float, default=0.2, help="Fixed, validation-selected LinearSVC regularization.")
    parser.add_argument("--feature-profile", choices=sorted(FEATURE_PROFILES), default="champion", help="Serving feature footprint to train.")
    parser.add_argument("--model-version", default=MODEL_VERSION, help="Model version stored in the release artifact.")
    parser.add_argument("--refit-full", action="store_true", help="Refit selected model on every deduplicated row after sealed evaluation.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path)
    required = {"jobrole_id", "jobrole_title", "task", "action", "object", "purpose", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    source_rows = len(frame)
    frame = frame.dropna(subset=["jobrole_id", "task", "label"]).copy()
    frame["target_label"] = frame["label"].map(SOURCE_TO_TARGET)
    if frame["target_label"].isna().any():
        unknown = sorted(frame.loc[frame["target_label"].isna(), "label"].astype(str).unique())
        raise ValueError(f"Unsupported source labels: {unknown}")
    for column in TEXT_FIELDS:
        frame[column] = frame[column].map(normalize_text)
    frame = frame.loc[frame["task"].ne("")].copy()
    frame["model_input_key"] = [
        compose_model_text(task, title, action, object_, purpose).lower()
        for task, title, action, object_, purpose in zip(
            frame["task"], frame["jobrole_title"], frame["action"], frame["object"], frame["purpose"]
        )
    ]
    label_counts = frame.groupby("model_input_key", sort=False)["target_label"].nunique()
    conflicts = set(label_counts[label_counts.gt(1)].index)
    without_conflicts = frame.loc[~frame["model_input_key"].isin(conflicts)].copy()
    deduplicated = (
        without_conflicts.sort_values(["model_input_key", "jobrole_id"], kind="stable")
        .drop_duplicates("model_input_key", keep="first")
        .reset_index(drop=True)
    )
    return deduplicated, {
        "source_rows": int(source_rows),
        "valid_rows": int(len(frame)),
        "conflicting_full_inputs_excluded": int(frame["model_input_key"].isin(conflicts).sum()),
        "duplicate_full_inputs_collapsed": int(len(without_conflicts) - len(deduplicated)),
        "model_rows": int(len(deduplicated)),
    }


def _make_split(frame: pd.DataFrame, seed: int, test_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame["target_label"].to_numpy()
    groups = frame["jobrole_id"].astype(str).to_numpy()
    outer_splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(frame, y, groups))
    if not 1 <= test_fold <= len(outer_splits):
        raise ValueError("test-fold must be between 1 and 5")
    development_idx, test_idx = outer_splits[test_fold - 1]
    development = frame.iloc[development_idx]
    train_relative, validation_relative = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1).split(
            development, development["target_label"], development["jobrole_id"].astype(str)
        )
    )
    return development_idx[train_relative], development_idx[validation_relative], test_idx


def _verify_split(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> dict[str, int]:
    checks: dict[str, int] = {}
    names = tuple(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            checks[f"shared_roles_{left}_{right}"] = len(set(frame.iloc[splits[left]]["jobrole_id"].astype(str)) & set(frame.iloc[splits[right]]["jobrole_id"].astype(str)))
            checks[f"shared_full_inputs_{left}_{right}"] = len(set(frame.iloc[splits[left]]["model_input_key"]) & set(frame.iloc[splits[right]]["model_input_key"]))
    if any(checks.values()):
        raise RuntimeError(f"Strict release leakage check failed: {checks}")
    return checks


def _fit(train: pd.DataFrame, c_value: float, feature_profile: str) -> tuple[FieldAwareTfidfFeatures, LinearSVC]:
    features = FieldAwareTfidfFeatures(profile=feature_profile)
    classifier = LinearSVC(C=c_value, class_weight=None, random_state=42)
    classifier.fit(features.fit_transform(train), train["target_label"])
    return features, classifier


def _evaluate(features: FieldAwareTfidfFeatures, classifier: LinearSVC, frame: pd.DataFrame) -> dict[str, Any]:
    predicted = classifier.predict(features.transform(frame))
    labels = frame["target_label"]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", labels=list(CLASS_ORDER))),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", labels=list(CLASS_ORDER))),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(CLASS_ORDER)).tolist(),
        "classification_report": classification_report(labels, predicted, labels=list(CLASS_ORDER), output_dict=True, zero_division=0),
    }


def main() -> None:
    args = parse_args()
    source_path, output_dir = Path(args.input), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, preparation = _load_frame(source_path)
    if set(frame["target_label"].unique()) != set(CLASS_ORDER):
        raise ValueError("All production classes must be present after target mapping")
    train_idx, validation_idx, test_idx = _make_split(frame, args.seed, args.test_fold)
    splits = {"train": train_idx, "validation": validation_idx, "test": test_idx}
    leakage = _verify_split(frame, splits)
    train, validation, test = (frame.iloc[index].reset_index(drop=True) for index in (train_idx, validation_idx, test_idx))
    features, classifier = _fit(train, args.c, args.feature_profile)
    validation_metrics, sealed_test_metrics = _evaluate(features, classifier, validation), _evaluate(features, classifier, test)
    sealed_predictions = classifier.predict(features.transform(test))
    if args.refit_full:
        features, classifier = _fit(frame, args.c, args.feature_profile)
        fit_scope, fitted_rows = "full_deduplicated_approved_dataset", len(frame)
    else:
        fit_scope, fitted_rows = "training_partition_only", len(train)
    artifact = {
        "artifact_schema_version": 2,
        "model_version": args.model_version,
        "label_policy": "GPT-5.2 with approved current POC guidance; source E2 and E3 merged to E23",
        "classes": list(CLASS_ORDER), "features": features, "classifier": classifier,
        "fit_scope": fit_scope, "fitted_rows": fitted_rows,
    }
    artifact_path = output_dir / "model.joblib"
    joblib.dump(artifact, artifact_path, compress=3)
    report = {
        "model_version": args.model_version,
        "target_definition": artifact["label_policy"],
        "input_columns": list(TEXT_FIELDS),
        "primary_api_input": ["action", "object", "purpose (optional)"],
        "compatibility_api_input": ["keytaskContent"],
        "optional_api_inputs": ["jobroleTitle"],
        "excluded_columns": ["reasoning_key", "truncated_reason", "score", "band", "sector_title", "ssoc_code", "jobrole_id", "cwfkt_id", "label"],
        "preparation": preparation,
        "split": {"seed": args.seed, "test_fold": args.test_fold, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "strict_leakage_passed": True, "leakage_checks": leakage},
        "fixed_hyperparameters": {"classifier": "LinearSVC", "c": args.c, "class_weight": None, "feature_profile": args.feature_profile, "feature_recipe": features.weights},
        "validation_metrics": validation_metrics, "sealed_test_metrics": sealed_test_metrics,
        "artifact": {"path": "model.joblib", "sha256": _sha256(artifact_path), "fit_scope": fit_scope, "fitted_rows": fitted_rows},
        "source_data": {"path": str(source_path), "sha256": _sha256(source_path)},
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    test.assign(predicted_label=sealed_predictions).to_csv(output_dir / "sealed_test_rows.csv", index=False)
    print(json.dumps({"artifact": str(artifact_path), "sealed_test_macro_f1": sealed_test_metrics["macro_f1"], "fit_scope": fit_scope}, indent=2))


if __name__ == "__main__":
    main()
