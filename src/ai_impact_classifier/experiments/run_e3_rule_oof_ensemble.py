"""OOF evaluation of E3-only association and keyword rule gates over the sparse stack."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from ai_impact_classifier.data import GROUP_COLUMN, build_bundle
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text, _metrics


NAMES = ["E0", "E3", "E12"]
# Domain-word heuristic; thresholds are selected only on an inner held-out role fold.
KEYWORD_PATTERN = re.compile(
    r"\b(?:creative|design(?:ing|s)?|illustrat(?:e|ion|ive)|graphic(?:s)?|visual(?:s|isation)?|"
    r"draw(?:ing|ings)?|video|animation|wireframe|mock-?up|collateral(?:s)?|storyboard|"
    r"photograph(?:y|ic)?|image(?:s)?|layout|3d\s+(?:model|art)|editing)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RuleConfig:
    rule: str
    min_confidence: float
    min_support: int
    min_matches: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OOF E3 rule gates over stack predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--stack-predictions", default="results/constrained_oof_stack_cv5_v1/oof_predictions.csv")
    parser.add_argument("--output-dir", default="results/e3_rule_oof_ensemble_v1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _phrase_scores(fit_text: pd.Series, fit_y: np.ndarray, eval_text: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(ngram_range=(1, 4), min_df=2, binary=True, max_features=500_000, strip_accents="unicode")
    x_fit = vectorizer.fit_transform(fit_text)
    x_eval = vectorizer.transform(eval_text)
    positive = np.asarray(x_fit[fit_y == 1].sum(axis=0)).ravel()
    total = np.asarray(x_fit.sum(axis=0)).ravel()
    confidence = positive / np.maximum(total, 1)
    return (
        x_eval.multiply(confidence).max(axis=1).toarray().ravel(),
        x_eval.multiply(positive).max(axis=1).toarray().ravel(),
        np.asarray(x_eval.sum(axis=1)).ravel(),
    )


def _keyword_scores(text: pd.Series) -> np.ndarray:
    return text.map(lambda value: len(KEYWORD_PATTERN.findall(value))).to_numpy(dtype=int)


def _rule_mask(config: RuleConfig, confidence: np.ndarray, support: np.ndarray, keyword_matches: np.ndarray) -> np.ndarray:
    if config.rule == "association":
        return (confidence >= config.min_confidence) & (support >= config.min_support)
    if config.rule == "keyword":
        return keyword_matches >= config.min_matches
    raise ValueError(f"Unknown rule: {config.rule}")


def _select_rule(fit: pd.DataFrame, fit_y: np.ndarray, validation: pd.DataFrame, val_y: np.ndarray) -> tuple[RuleConfig, dict[str, float]]:
    fit_text = _allowed_text(fit, "title_task")
    val_text = _allowed_text(validation, "title_task")
    confidence, support, _ = _phrase_scores(fit_text, fit_y, val_text)
    keyword_matches = _keyword_scores(val_text)
    candidates = [
        RuleConfig("association", confidence, minimum_support, 0)
        for confidence in np.linspace(0.35, 0.95, 13)
        for minimum_support in (1, 2, 3, 4, 5)
    ] + [
        RuleConfig("keyword", 0.0, 0, minimum_matches)
        for minimum_matches in (1, 2, 3, 4)
    ]
    best: tuple[RuleConfig, dict[str, float]] | None = None
    target = (val_y == 1).astype(int)
    for config in candidates:
        mask = _rule_mask(config, confidence, support, keyword_matches)
        # Prioritize E3 F1, then precision so a gate does not become a broad E3 override.
        f1 = float(f1_score(target, mask, zero_division=0))
        precision = float((target[mask].mean()) if mask.any() else 0.0)
        record = {"e3_f1": f1, "e3_precision": precision, "rule_fires": int(mask.sum())}
        if best is None or (f1, precision) > (best[1]["e3_f1"], best[1]["e3_precision"]):
            best = (config, record)
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    stack = pd.read_csv(args.stack_predictions).set_index("row_index")
    if set(stack.index) != set(frame.index):
        raise ValueError("Stack prediction rows do not match the source data rows.")
    y = frame.label_rank.to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    baseline = stack.loc[frame.index, "y_pred"].to_numpy(dtype=int)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    ensemble = baseline.copy()
    fold_rows: list[dict[str, object]] = []

    for fold, (development_idx, test_idx) in enumerate(outer.split(frame, y, groups), start=1):
        development = frame.iloc[development_idx].copy()
        y_development = y[development_idx]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + fold)
        fit_idx, validation_idx = next(inner.split(development, y_development, development[GROUP_COLUMN].astype(str)))
        config, validation = _select_rule(
            development.iloc[fit_idx], y_development[fit_idx], development.iloc[validation_idx], y_development[validation_idx]
        )
        confidence, support, _ = _phrase_scores(_allowed_text(development, "title_task"), y_development, _allowed_text(frame.iloc[test_idx], "title_task"))
        mask = _rule_mask(config, confidence, support, _keyword_scores(_allowed_text(frame.iloc[test_idx], "title_task")))
        before = baseline[test_idx].copy()
        after = before.copy(); after[mask] = 1
        ensemble[test_idx] = after
        fold_rows.append({
            "fold": fold,
            "rule": config.rule,
            "min_confidence": config.min_confidence,
            "min_support": config.min_support,
            "min_matches": config.min_matches,
            "inner_rule_e3_f1": validation["e3_f1"],
            "inner_rule_e3_precision": validation["e3_precision"],
            "outer_overrides": int(mask.sum()),
            "baseline_macro_f1": _metrics(y[test_idx], before)["macro_f1"],
            "ensemble_macro_f1": _metrics(y[test_idx], after)["macro_f1"],
        })
        print(json.dumps(fold_rows[-1]), flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out / "fold_rule_metrics.csv", index=False)
    pd.DataFrame({"row_index": frame.index, "y_true": y, "stack_prediction": baseline, "ensemble_prediction": ensemble, "e3_override": baseline != ensemble}).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y, ensemble, labels=range(3)), index=NAMES, columns=NAMES).to_csv(out / "ensemble_oof_confusion_matrix.csv")
    payload = {
        "method": "E3-only association / keyword rule gate over nested-stack OOF predictions",
        "allowed_input_columns": ["keytask_content", "jobrole_title"],
        "base_stack_predictions_are_outer_fold_out_of_sample": True,
        "rule_selection_is_inner_group_stratified": True,
        "baseline_oof_metrics": _metrics(y, baseline),
        "ensemble_oof_metrics": _metrics(y, ensemble),
        "total_e3_overrides": int((baseline != ensemble).sum()),
    }
    (out / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
