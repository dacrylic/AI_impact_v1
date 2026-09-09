"""Analyze the representative factorial teacher-label run.

The script is deliberately separate from collection: it only uses successful
responses, keeps raw-parent and atomic-child grains distinct, and reports the
simple-random panel as a prevalence estimate. The old labels are used only for
the historical-repeatability comparison, never for the outbound sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CELLS = (
    ("legacy", "gpt_4o", "legacy_4o"),
    ("legacy", "gpt_5_2", "legacy_5_2"),
    ("new_production", "gpt_4o", "production_4o"),
    ("new_production", "gpt_5_2", "production_5_2"),
)
LABELS = ("E0", "E1", "E2", "E3")


def merged(label: str) -> str:
    return "E23" if label in {"E2", "E3"} else label


def bootstrap_prevalence(values: np.ndarray, seed: int, reps: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(x) for x in (values.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975)))


def pivot_success(manifest: pd.DataFrame, responses: pd.DataFrame, grain: str) -> pd.DataFrame:
    requested = manifest.loc[manifest.task_grain.eq(grain), ["call_id", "panel_parent_id", "panel_child_id", "prompt_id", "model_family"]]
    ok = responses.loc[responses.status.eq("success"), ["call_id", "label"]]
    joined = requested.merge(ok, on="call_id", how="inner")
    cell_names = {(p, m): cell for p, m, cell in CELLS}
    joined["cell"] = [cell_names[(p, m)] for p, m in zip(joined.prompt_id, joined.model_family)]
    keys = ["panel_parent_id"] if grain == "raw_parent" else ["panel_parent_id", "panel_child_id"]
    table = joined.pivot_table(index=keys, columns="cell", values="label", aggfunc="first")
    table.columns.name = None
    for _, _, cell in CELLS:
        if cell not in table.columns:
            table[cell] = pd.Series(dtype="object")
    return table.dropna(subset=[x[2] for x in CELLS], how="any")


def distribution(table: pd.DataFrame, prefix: str, reps: int) -> pd.DataFrame:
    rows = []
    for _, _, cell in CELLS:
        values = table[cell].astype(str)
        for label in LABELS:
            rows.append({"grain": prefix, "cell": cell, "label": label, "count": int(values.eq(label).sum()), "share": float(values.eq(label).mean())})
        merged_values = values.map(merged)
        for label in ("E0", "E1", "E23"):
            rows.append({"grain": prefix, "cell": cell, "label": f"merged_{label}", "count": int(merged_values.eq(label).sum()), "share": float(merged_values.eq(label).mean())})
        non_e0 = values.ne("E0").to_numpy(dtype=float)
        mean, lo, hi = bootstrap_prevalence(non_e0, 20260829 + len(rows), reps)
        rows.append({"grain": prefix, "cell": cell, "label": "non_E0", "count": int(non_e0.sum()), "share": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def transitions(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    baseline = table["legacy_4o"].map(merged)
    rows = []
    for _, _, cell in CELLS:
        values = table[cell].map(merged)
        rows.append({"grain": prefix, "comparison": f"{cell}_vs_legacy_4o", "cell": cell, "n": len(table), "agreement_3class": float(values.eq(baseline).mean()), "non_e0_delta_pp": float((values.ne("E0").mean() - baseline.ne("E0").mean()) * 100), "e1_delta_pp": float((values.eq("E1").mean() - baseline.eq("E1").mean()) * 100)})
    return pd.DataFrame(rows)


def factorial_effects(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Estimate 2x2 prompt/model effects for prevalence outcomes.

    Effects are percentage-point contrasts on the same paired panel. The
    interaction is the difference-in-differences: whether the prompt change
    behaves differently under GPT-5.2 than under GPT-4o.
    """
    cells = {cell: table[cell].map(merged) for _, _, cell in CELLS}
    outcomes = {
        "non_E0": {cell: values.ne("E0").astype(float) for cell, values in cells.items()},
        "E1": {cell: values.eq("E1").astype(float) for cell, values in cells.items()},
        "E23": {cell: values.eq("E23").astype(float) for cell, values in cells.items()},
    }
    rows = []
    for outcome, values in outcomes.items():
        p4, p5 = values["legacy_4o"].mean(), values["legacy_5_2"].mean()
        n4, n5 = values["production_4o"].mean(), values["production_5_2"].mean()
        paper_model = (p5 - p4) * 100
        production_model = (n5 - n4) * 100
        paper_prompt = (n4 - p4) * 100
        five_prompt = (n5 - p5) * 100
        rows.append({
            "grain": prefix, "outcome": outcome, "n": len(table),
            "paper_prompt_effect_pp_at_4o": paper_prompt,
            "paper_prompt_effect_pp_at_5_2": five_prompt,
            "model_effect_pp_under_paper": paper_model,
            "model_effect_pp_under_production": production_model,
            "prompt_by_model_interaction_pp": (five_prompt - paper_prompt),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--parent-panel", required=True)
    parser.add_argument("--child-panel", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["prompt_id"] = manifest["prompt_id"].replace({"paper_2023": "legacy"})
    responses = pd.read_csv(args.responses)
    parent = pd.read_csv(args.parent_panel)
    child = pd.read_csv(args.child_panel)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw = pivot_success(manifest, responses, "raw_parent")
    atomic = pivot_success(manifest, responses, "atomic_child")
    raw_dist = distribution(raw, "raw_parent", args.bootstrap_reps)
    atomic_dist = distribution(atomic, "atomic_child", args.bootstrap_reps)
    trans = pd.concat([transitions(raw, "raw_parent"), transitions(atomic, "atomic_child")], ignore_index=True)
    effects = pd.concat([factorial_effects(raw, "raw_parent"), factorial_effects(atomic, "atomic_child")], ignore_index=True)

    raw_ids = raw.reset_index()[["panel_parent_id"]].merge(parent[["panel_parent_id", "openai_label", "ai_impact_score"]], on="panel_parent_id", how="left")
    historic = []
    for _, _, cell in CELLS:
        pred = raw[cell].map(merged).to_numpy()
        truth = raw_ids.openai_label.astype(str).map(merged).to_numpy()
        historic.append({"cell": cell, "n": len(raw), "agreement_with_old_3class": float((pred == truth).mean()), "old_non_e0": float((truth != "E0").mean()), "pred_non_e0": float((pred != "E0").mean())})
    historic_df = pd.DataFrame(historic)

    raw.to_csv(out / "raw_complete_cell_labels.csv")
    atomic.to_csv(out / "atomic_complete_cell_labels.csv")
    raw_dist.to_csv(out / "raw_distributions.csv", index=False)
    atomic_dist.to_csv(out / "atomic_distributions.csv", index=False)
    trans.to_csv(out / "paired_effects.csv", index=False)
    effects.to_csv(out / "factorial_effects.csv", index=False)
    historic_df.to_csv(out / "historical_agreement.csv", index=False)

    summary = {
        "successful_responses": int(responses.status.eq("success").sum()),
        "response_rows": int(len(responses)),
        "duplicate_call_ids": int(responses.call_id.duplicated().sum()),
        "complete_raw_parents": int(len(raw)),
        "complete_atomic_children": int(len(atomic)),
        "expected_raw_parents": int(parent.shape[0]),
        "expected_atomic_children": int(child.shape[0]),
        "bootstrap_replicates": args.bootstrap_reps,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
