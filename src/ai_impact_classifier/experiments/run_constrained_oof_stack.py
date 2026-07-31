"""Group-OOF stack using only key task content and optional job role title."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ai_impact_classifier.checks import check_split_leakage
from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text
from ai_impact_classifier.splitters import stratified_group_train_val_test_split


NAMES = ["E0", "E3", "E12"]
SOFT_ID_COLUMNS = ["task_id", "jobrole_task_id", "ssoc_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a group-OOF stack with task-only and title+task text.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results/constrained_oof_stack")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _base_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=350_000, sublinear_tf=True, strip_accents="unicode")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300_000, sublinear_tf=True)),
                    ],
                    transformer_weights={"word": 1.0, "char": 0.65},
                ),
            ),
            ("classifier", LinearSVC(C=c_value, class_weight="balanced", max_iter=20_000)),
        ]
    )


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro")), "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def _best_bias(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    bias = np.zeros(scores.shape[1]); best = f1_score(y, np.argmax(scores, axis=1), average="macro")
    for _ in range(3):
        changed = False
        for cls in range(1, scores.shape[1]):
            for value in np.linspace(-1.5, 1.5, 41):
                candidate = bias.copy(); candidate[cls] = value
                metric = f1_score(y, np.argmax(scores + candidate, axis=1), average="macro")
                if metric > best: bias, best, changed = candidate, metric, True
        if not changed: break
    return bias


def _oof_base_features(frame: pd.DataFrame, y: np.ndarray, seed: int, specs: list[tuple[str, float]]) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy(); splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = np.zeros((len(frame), len(specs) * len(NAMES)))
    for train_idx, holdout_idx in splitter.split(frame, y, groups):
        fit_frame, holdout_frame = frame.iloc[train_idx], frame.iloc[holdout_idx]
        for spec_idx, (view, c_value) in enumerate(specs):
            model = _base_model(c_value); model.fit(_allowed_text(fit_frame, view), y[train_idx])
            output[holdout_idx, spec_idx * len(NAMES):(spec_idx + 1) * len(NAMES)] = model.decision_function(_allowed_text(holdout_frame, view))
    return output


def _fit_base_features(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame, specs: list[tuple[str, float]]) -> np.ndarray:
    output = np.zeros((len(target), len(specs) * len(NAMES)))
    for spec_idx, (view, c_value) in enumerate(specs):
        model = _base_model(c_value); model.fit(_allowed_text(train, view), y)
        output[:, spec_idx * len(NAMES):(spec_idx + 1) * len(NAMES)] = model.decision_function(_allowed_text(target, view))
    return output


def _oof_e3_features(frame: pd.DataFrame, y: np.ndarray, seed: int, specs: list[tuple[str, float, float]]) -> np.ndarray:
    groups = frame[GROUP_COLUMN].astype(str).to_numpy(); splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    output = np.zeros((len(frame), len(specs)))
    for train_idx, holdout_idx in splitter.split(frame, y, groups):
        fit_frame, holdout_frame = frame.iloc[train_idx], frame.iloc[holdout_idx]
        for spec_idx, (view, c_value, positive_weight) in enumerate(specs):
            model = _base_model(c_value)
            model.set_params(classifier__class_weight={0: 1.0, 1: positive_weight})
            model.fit(_allowed_text(fit_frame, view), (y[train_idx] == 1).astype(int))
            output[holdout_idx, spec_idx] = model.decision_function(_allowed_text(holdout_frame, view))
    return output


def _fit_e3_features(train: pd.DataFrame, y: np.ndarray, target: pd.DataFrame, specs: list[tuple[str, float, float]]) -> np.ndarray:
    output = np.zeros((len(target), len(specs)))
    for spec_idx, (view, c_value, positive_weight) in enumerate(specs):
        model = _base_model(c_value)
        model.set_params(classifier__class_weight={0: 1.0, 1: positive_weight})
        model.fit(_allowed_text(train, view), (y == 1).astype(int))
        output[:, spec_idx] = model.decision_function(_allowed_text(target, view))
    return output


def main() -> None:
    args = parse_args(); bundle = build_bundle(args.input); frame = bundle.frame.copy()
    split = stratified_group_train_val_test_split(frame, GROUP_COLUMN, "label_rank", random_state=args.seed)
    leak = check_split_leakage(frame, {"train": split.train_idx, "val": split.val_idx, "test": split.test_idx}, GROUP_COLUMN, text_column="target_text", soft_id_columns=SOFT_ID_COLUMNS)
    if not leak.passed: raise RuntimeError("Leakage check failed: " + "; ".join(leak.messages))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train, val, test = (frame.loc[idx].copy() for idx in (split.train_idx, split.val_idx, split.test_idx))
    y_train, y_val, y_test = (part.label_rank.to_numpy() for part in (train, val, test))
    specs = [("task_only", .25), ("task_only", .75), ("title_task", .25), ("title_task", .75)]
    e3_specs = [("task_only", .25, 2.0), ("task_only", .5, 4.0), ("task_only", .75, 8.0), ("title_task", .5, 4.0)]
    oof_train = np.column_stack([
        _oof_base_features(train.reset_index(drop=True), y_train, args.seed, specs),
        _oof_e3_features(train.reset_index(drop=True), y_train, args.seed, e3_specs),
    ])
    val_base = np.column_stack([_fit_base_features(train, y_train, val, specs), _fit_e3_features(train, y_train, val, e3_specs)])
    meta_rows, selected = [], None
    for c_value in [.02, .05, .1, .25, .5]:
        for e3_weight in [1.0, 1.5, 2.0, 3.0]:
            class_weight = {0: 1.0, 1: e3_weight, 2: 1.0}
            meta = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=c_value, class_weight=class_weight, max_iter=5000, n_jobs=1))])
            meta.fit(oof_train, y_train); scores = meta.predict_proba(val_base); bias = _best_bias(y_val, scores); pred = np.argmax(scores + bias, axis=1); metrics = _metrics(y_val, pred)
            meta_rows.append({"meta_c": c_value, "e3_weight": e3_weight, "bias": json.dumps(bias.tolist()), **metrics})
            if selected is None or metrics["macro_f1"] > selected[3]["macro_f1"]: selected = (c_value, e3_weight, bias, metrics)
    assert selected is not None
    meta_c, e3_weight, bias, val_metrics = selected
    development = pd.concat([train, val], ignore_index=True); y_dev = development.label_rank.to_numpy()
    oof_dev = np.column_stack([_oof_base_features(development, y_dev, args.seed, specs), _oof_e3_features(development, y_dev, args.seed, e3_specs)])
    test_base = np.column_stack([_fit_base_features(development, y_dev, test, specs), _fit_e3_features(development, y_dev, test, e3_specs)])
    meta = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(C=meta_c, class_weight={0: 1.0, 1: e3_weight, 2: 1.0}, max_iter=5000, n_jobs=1))])
    meta.fit(oof_dev, y_dev); test_pred = np.argmax(meta.predict_proba(test_base) + bias, axis=1); test_metrics = _metrics(y_test, test_pred)
    pd.DataFrame(meta_rows).sort_values("macro_f1", ascending=False).to_csv(out / "meta_validation_leaderboard.csv", index=False)
    pd.DataFrame(confusion_matrix(y_test, test_pred, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "test_confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    payload = {"target_definition":"E0 vs E3 vs E12","test_blind_selection":True,"allowed_input_columns":["keytask_content","jobrole_title"],"base_models":[{"text_view":view,"c":c} for view,c in specs],"e3_specialists":[{"text_view":view,"c":c,"positive_weight":weight} for view,c,weight in e3_specs],"selected_meta_c":meta_c,"selected_meta_e3_weight":e3_weight,"bias":bias.tolist(),"validation_metrics":val_metrics,"test_metrics":test_metrics,"leakage_passed":leak.passed,"leakage_warnings":leak.warnings}; (out / "run_metadata.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"validation_metrics":val_metrics,"test_metrics":test_metrics},indent=2))


if __name__ == "__main__": main()
