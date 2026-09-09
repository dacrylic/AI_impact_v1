"""Complete matched analysis of legacy, production, and anchored AI-impact prompts.

The study contains a 2 x 2 core (legacy vs updated production prompt, GPT-4o
vs GPT-5.2) and a third prompt arm that adds the recovered legacy few-shot
examples to the updated production prompt.  Raw parent tasks and AOP atomic
children are deliberately analysed as linked, different units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


LABELS = ("E0", "E1", "E2", "E3")
OUTCOMES = {
    "E0": lambda s: s.eq("E0"),
    "E1": lambda s: s.eq("E1"),
    "E23": lambda s: s.isin(["E2", "E3"]),
    "impacted_non_E0": lambda s: s.ne("E0"),
}
PROMPT_ORDER = ("legacy", "production", "production_legacy_anchors")
MODEL_ORDER = ("gpt_4o", "gpt_5_2")
GRAIN_ORDER = ("raw_parent", "atomic_child")


def display_prompt(value: str) -> str:
    return {
        "legacy": "Legacy prompt (recovered colleague script)",
        "production": "Updated production prompt",
        "production_legacy_anchors": "Updated production + legacy few-shot anchors",
    }[value]


def display_model(value: str) -> str:
    return {"gpt_4o": "GPT-4o", "gpt_5_2": "GPT-5.2"}[value]


def ci_mean(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for a panel mean."""
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    estimates = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def ci_paired_delta(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float, float]:
    """Paired bootstrap for left minus right, in percentage points."""
    delta = left.astype(float) - right.astype(float)
    mean, lo, hi = ci_mean(delta, rng, reps)
    return mean * 100, lo * 100, hi * 100


def read_condition(manifest_path: Path, responses_path: Path, prompt: str, prompt_id: str | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str)
    responses = pd.read_csv(responses_path, dtype=str)
    if prompt_id is not None:
        manifest = manifest.loc[manifest.prompt_id.eq(prompt_id)].copy()
    responses = responses.loc[responses.status.eq("success") & responses.label.isin(LABELS)].copy()
    joined = manifest.merge(responses[["call_id", "label", "resolved_model"]], on="call_id", how="inner", validate="one_to_one")
    joined["prompt"] = prompt
    joined["source_run"] = responses_path.parent.name
    return joined


def cell_table(calls: pd.DataFrame, grain: str) -> pd.DataFrame:
    """Wide matched table: one row per raw parent or atomic child."""
    keys = ["panel_parent_id"] if grain == "raw_parent" else ["panel_parent_id", "panel_child_id"]
    part = calls.loc[calls.task_grain.eq(grain)].copy()
    wide = part.pivot(index=keys, columns=["prompt", "model_family"], values="label")
    expected = pd.MultiIndex.from_product([PROMPT_ORDER, MODEL_ORDER], names=["prompt", "model_family"])
    wide = wide.reindex(columns=expected).dropna(how="any")
    return wide


def distribution_rows(table: pd.DataFrame, grain: str, unit: str, reps: int, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for prompt in PROMPT_ORDER:
        for model in MODEL_ORDER:
            labels = table[(prompt, model)].astype(str)
            for name, fn in {"E23": OUTCOMES["E23"], "impacted_non_E0": OUTCOMES["impacted_non_E0"]}.items():
                mean, lo, hi = ci_mean(fn(labels).to_numpy(dtype=float), rng, reps)
                rows.append({
                    "grain": grain, "unit": unit, "prompt": prompt, "prompt_display": display_prompt(prompt),
                    "model_family": model, "model": display_model(model), "outcome": name,
                    "n": len(labels), "count": int(fn(labels).sum()), "share": mean,
                    "share_pct": mean * 100, "ci_low_pct": lo * 100, "ci_high_pct": hi * 100,
                })
            for label in LABELS:
                mean, lo, hi = ci_mean(labels.eq(label).to_numpy(dtype=float), rng, reps)
                rows.append({
                    "grain": grain, "unit": unit, "prompt": prompt, "prompt_display": display_prompt(prompt),
                    "model_family": model, "model": display_model(model), "outcome": label,
                    "n": len(labels), "count": int(labels.eq(label).sum()), "share": mean,
                    "share_pct": mean * 100, "ci_low_pct": lo * 100, "ci_high_pct": hi * 100,
                })
    return rows


def paired_rows(table: pd.DataFrame, grain: str, comparison_type: str, left: tuple[str, str], right: tuple[str, str], reps: int, rng: np.random.Generator) -> list[dict]:
    rows: list[dict] = []
    for outcome, fn in OUTCOMES.items():
        l = fn(table[left]).to_numpy(dtype=float)
        r = fn(table[right]).to_numpy(dtype=float)
        delta, lo, hi = ci_paired_delta(l, r, rng, reps)
        rows.append({
            "grain": grain, "comparison_type": comparison_type,
            "left_prompt": left[0], "left_model": left[1], "right_prompt": right[0], "right_model": right[1],
            "left_condition": f"{display_prompt(left[0])} / {display_model(left[1])}",
            "right_condition": f"{display_prompt(right[0])} / {display_model(right[1])}",
            "outcome": outcome, "n": len(table), "delta_pp_left_minus_right": delta,
            "ci_low_pp": lo, "ci_high_pp": hi,
            "agreement_exact_4class": float(table[left].eq(table[right]).mean()),
        })
    return rows


def representation_rows(raw: pd.DataFrame, atomic: pd.DataFrame, reps: int, rng: np.random.Generator) -> list[dict]:
    """Compare raw labels with each parent's mean child outcome rate.

    This respects the linked data structure: every parent has one raw label but
    may have several child tasks. A parent receives equal weight regardless of
    how many children it was decomposed into.
    """
    rows: list[dict] = []
    parent_index = raw.index
    for prompt in PROMPT_ORDER:
        for model in MODEL_ORDER:
            child_labels = atomic[(prompt, model)]
            for outcome, fn in OUTCOMES.items():
                child_parent_mean = fn(child_labels).groupby(level="panel_parent_id").mean().reindex(parent_index)
                raw_indicator = fn(raw[(prompt, model)]).reindex(parent_index)
                delta, lo, hi = ci_paired_delta(child_parent_mean.to_numpy(), raw_indicator.to_numpy(), rng, reps)
                rows.append({
                    "prompt": prompt, "prompt_display": display_prompt(prompt), "model_family": model,
                    "model": display_model(model), "outcome": outcome, "n_parents": len(parent_index),
                    "raw_parent_pct": raw_indicator.mean() * 100,
                    "mean_child_pct_parent_weighted": child_parent_mean.mean() * 100,
                    "child_minus_raw_pp": delta, "ci_low_pp": lo, "ci_high_pp": hi,
                })
    return rows


def historical_rows(raw: pd.DataFrame, parent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = parent.set_index("panel_parent_id").loc[raw.index, "openai_label"].astype(str)
    rows: list[dict] = []
    matrices: list[dict] = []
    for prompt in PROMPT_ORDER:
        for model in MODEL_ORDER:
            pred = raw[(prompt, model)].astype(str)
            rows.append({
                "prompt": prompt, "prompt_display": display_prompt(prompt), "model_family": model,
                "model": display_model(model), "n": len(raw),
                "exact_4class_agreement": float(pred.eq(historical).mean()),
                "merged_3class_agreement": float(pred.replace({"E2": "E23", "E3": "E23"}).eq(historical.replace({"E2": "E23", "E3": "E23"})).mean()),
                "binary_E0_vs_impacted_agreement": float(pred.eq("E0").eq(historical.eq("E0")).mean()),
                "macro_f1_4class": float(f1_score(historical, pred, labels=list(LABELS), average="macro", zero_division=0)),
                "historical_E0_pct": float(historical.eq("E0").mean() * 100),
                "predicted_E0_pct": float(pred.eq("E0").mean() * 100),
            })
            matrix = pd.crosstab(historical, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
            for actual in LABELS:
                for predicted in LABELS:
                    matrices.append({"prompt": prompt, "model_family": model, "actual_historical": actual, "predicted": predicted, "count": int(matrix.loc[actual, predicted])})
    return pd.DataFrame(rows), pd.DataFrame(matrices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--legacy-responses", required=True)
    parser.add_argument("--production-manifest", required=True)
    parser.add_argument("--production-responses", required=True)
    parser.add_argument("--anchors-manifest", required=True)
    parser.add_argument("--anchors-responses", required=True)
    parser.add_argument("--parent-panel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    calls = pd.concat([
        read_condition(Path(args.legacy_manifest), Path(args.legacy_responses), "legacy"),
        read_condition(Path(args.production_manifest), Path(args.production_responses), "production", "new_production"),
        read_condition(Path(args.anchors_manifest), Path(args.anchors_responses), "production_legacy_anchors"),
    ], ignore_index=True)
    parent = pd.read_csv(args.parent_panel, dtype=str)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = cell_table(calls, "raw_parent")
    atomic = cell_table(calls, "atomic_child")
    raw.index.name = "panel_parent_id"
    atomic.index.names = ["panel_parent_id", "panel_child_id"]

    distributions = pd.DataFrame(
        distribution_rows(raw, "raw_parent", "parent", args.bootstrap_reps, rng)
        + distribution_rows(atomic, "atomic_child", "child_micro", args.bootstrap_reps, rng)
    )
    paired: list[dict] = []
    # Prompt contrasts within each model: legacy vs production; anchors vs production; anchors vs legacy.
    for table, grain in ((raw, "raw_parent"), (atomic, "atomic_child")):
        for model in MODEL_ORDER:
            paired += paired_rows(table, grain, "prompt: production minus legacy", ("production", model), ("legacy", model), args.bootstrap_reps, rng)
            paired += paired_rows(table, grain, "prompt: anchored minus production", ("production_legacy_anchors", model), ("production", model), args.bootstrap_reps, rng)
            paired += paired_rows(table, grain, "prompt: anchored minus legacy", ("production_legacy_anchors", model), ("legacy", model), args.bootstrap_reps, rng)
        # Model contrasts within each prompt.
        for prompt in PROMPT_ORDER:
            paired += paired_rows(table, grain, "model: GPT-5.2 minus GPT-4o", (prompt, "gpt_5_2"), (prompt, "gpt_4o"), args.bootstrap_reps, rng)
    paired_df = pd.DataFrame(paired)
    representation = pd.DataFrame(representation_rows(raw, atomic, args.bootstrap_reps, rng))
    historic, confusion = historical_rows(raw, parent)

    raw_export = raw.copy()
    raw_export.columns = [f"{prompt}__{model}" for prompt, model in raw_export.columns]
    atomic_export = atomic.copy()
    atomic_export.columns = [f"{prompt}__{model}" for prompt, model in atomic_export.columns]
    raw_export.reset_index().to_csv(out / "raw_complete_cell_labels.csv", index=False)
    atomic_export.reset_index().to_csv(out / "atomic_complete_cell_labels.csv", index=False)
    distributions.to_csv(out / "distributions.csv", index=False)
    paired_df.to_csv(out / "paired_prompt_and_model_effects.csv", index=False)
    representation.to_csv(out / "parent_clustered_representation_effects.csv", index=False)
    historic.to_csv(out / "historical_repeatability.csv", index=False)
    confusion.to_csv(out / "historical_confusion_matrices.csv", index=False)
    calls.to_csv(out / "all_successful_calls.csv", index=False)

    summary = {
        "design": "matched 3 prompt x 2 model x 2 task-grain study; legacy-vs-production is the 2x2 core and few-shot anchors are a third prompt arm",
        "successful_calls": int(len(calls)),
        "duplicate_call_ids_within_source_run": int(calls.duplicated(["source_run", "call_id"]).sum()),
        "call_ids_reused_across_separate_runs": int(calls.call_id.duplicated().sum()),
        "raw_parent_complete_cases": int(len(raw)),
        "atomic_child_complete_cases": int(len(atomic)),
        "bootstrap_replicates": args.bootstrap_reps,
        "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
