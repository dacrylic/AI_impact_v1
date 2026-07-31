"""Create a review pack for the remaining E3 errors in OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ai_impact_classifier.data import build_bundle
from ai_impact_classifier.experiments.run_merged_e12_specialists import _allowed_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a focused E3 error and duplicate-text audit pack.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictions", default="results/dense_sparse_oof_cv5_v1/oof_predictions.csv")
    parser.add_argument("--output-dir", default="results/e3_audit_v1")
    return parser.parse_args()


def _top_terms(text: pd.Series, limit: int = 50) -> pd.DataFrame:
    if text.empty:
        return pd.DataFrame(columns=["term", "document_count"])
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, binary=True, max_features=100_000, strip_accents="unicode")
    matrix = vectorizer.fit_transform(text)
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    return pd.DataFrame({"term": terms, "document_count": counts}).sort_values("document_count", ascending=False).head(limit)


def main() -> None:
    args = parse_args()
    frame = build_bundle(args.input).frame.reset_index(drop=True)
    pred = pd.read_csv(args.predictions).set_index("row_index")
    audit = frame.loc[pred.index, ["jobrole_id", "jobrole_title", "keytask_content", "openai_label", "model_label", "label_rank"]].copy()
    audit["prediction_rank"] = pred["y_pred"].to_numpy()
    names = {0: "E0", 1: "E3", 2: "E12"}; audit["prediction"] = audit.prediction_rank.map(names)
    audit["error_type"] = "correct"
    audit.loc[(audit.label_rank == 1) & (audit.prediction_rank != 1), "error_type"] = "missed_E3"
    audit.loc[(audit.label_rank != 1) & (audit.prediction_rank == 1), "error_type"] = "false_E3"
    audit["model_text"] = _allowed_text(audit, "title_task")
    errors = audit.loc[audit.error_type.ne("correct")].sort_values(["error_type", "jobrole_title", "keytask_content"])
    duplicate_summary = (
        audit.assign(task_normalized=audit.keytask_content.str.lower().str.strip())
        .groupby("task_normalized")
        .agg(rows=("jobrole_id", "size"), roles=("jobrole_id", "nunique"), labels=("model_label", "nunique"), raw_labels=("openai_label", "nunique"))
        .query("rows > 1 and labels > 1")
        .sort_values(["rows", "roles"], ascending=False)
        .reset_index()
    )
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    errors.drop(columns="model_text").to_csv(out / "e3_error_review.csv", index=False)
    _top_terms(audit.loc[audit.error_type.eq("missed_E3"), "model_text"]).to_csv(out / "missed_e3_top_terms.csv", index=False)
    _top_terms(audit.loc[audit.error_type.eq("false_E3"), "model_text"]).to_csv(out / "false_e3_top_terms.csv", index=False)
    duplicate_summary.to_csv(out / "cross_role_conflicting_task_text.csv", index=False)
    payload = {
        "source_predictions": str(args.predictions),
        "e3_true_rows": int((audit.label_rank == 1).sum()),
        "missed_e3": int((audit.error_type == "missed_E3").sum()),
        "false_e3": int((audit.error_type == "false_E3").sum()),
        "cross_role_conflicting_canonical_task_text_groups": int(len(duplicate_summary)),
        "review_instruction": "Prioritize rows where title/task semantics make the assigned E3 versus E0/E12 label ambiguous; use corrections only after human adjudication.",
    }
    (out / "audit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
