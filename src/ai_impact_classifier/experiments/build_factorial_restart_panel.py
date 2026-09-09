"""Build a persistent matched factorial request panel without sending API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


LABELS = ("E0", "E1", "E2", "E3")
QUOTAS = {"E0": 400, "E1": 250, "E2": 250, "E3": 100}


def norm(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def quota(frame: pd.DataFrame, n: int) -> dict[str, int]:
    counts = frame.sector_title.fillna("Unknown").astype(str).value_counts()
    desired = counts / counts.sum() * n
    result = desired.astype(int).to_dict()
    for sector in sorted(counts.index, key=lambda s: (-(desired[s] % 1), s))[: n - sum(result.values())]:
        result[sector] += 1
    return result


def sample(frame: pd.DataFrame, n: int, cap: int, role_counts: dict[str, int], seed: int) -> pd.DataFrame:
    shuffled = frame.sample(frac=1, random_state=seed).copy()
    shuffled["_sector"] = shuffled.sector_title.fillna("Unknown").astype(str)
    queues = {s: deque(g.index.tolist()) for s, g in shuffled.groupby("_sector", sort=False)}
    chosen: list[int] = []
    for sector, take in quota(shuffled, min(n, len(shuffled))).items():
        while take and queues[sector]:
            idx = queues[sector].popleft()
            role = str(shuffled.at[idx, "jobrole_id"])
            if role_counts[role] < cap:
                chosen.append(idx)
                role_counts[role] += 1
                take -= 1
    for idx, row in shuffled.iterrows():
        if len(chosen) >= n:
            break
        role = str(row.jobrole_id)
        if idx not in chosen and role_counts[role] < cap:
            chosen.append(idx)
            role_counts[role] += 1
    return shuffled.loc[chosen].drop(columns="_sector")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-workbook", required=True)
    parser.add_argument("--new-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--role-cap", type=int, default=3)
    parser.add_argument("--sampling-method", choices=("label_stratified", "simple_random"), default="label_stratified")
    parser.add_argument("--sample-size", type=int, default=978)
    args = parser.parse_args()

    old = pd.read_excel(args.old_workbook, sheet_name="task ai impact")
    new = pd.read_csv(args.new_csv)
    old = old.dropna(subset=["jobrole_id", "jobrole_title", "keytask_content", "openai_label"]).copy()
    new = new.dropna(subset=["jobrole_id", "jobrole_title", "kt_content", "task", "label"]).copy()
    old["parent_key"] = old.jobrole_id.astype(str) + "\x1f" + old.keytask_content.map(norm)
    new["parent_key"] = new.jobrole_id.astype(str) + "\x1f" + new.kt_content.map(norm)
    old_parent = old.sort_values("parent_key").drop_duplicates("parent_key")
    new_parent = new.groupby("parent_key", as_index=False).agg(new_role_title=("jobrole_title", "first"), atomic_children=("task", "size"))
    parents = old_parent.merge(new_parent, on="parent_key", how="inner")
    parents = parents.loc[parents.jobrole_title.map(norm).eq(parents.new_role_title.map(norm))].copy()
    parents["panel_parent_id"] = parents.parent_key.map(stable)
    if args.sampling_method == "simple_random":
        if args.sample_size > len(parents):
            raise ValueError("--sample-size exceeds eligible matched parents")
        selected = parents.sample(n=args.sample_size, random_state=args.seed).copy()
    else:
        roles: dict[str, int] = defaultdict(int)
        selected = pd.concat([
            sample(parents.loc[parents.openai_label.eq(label)], min(QUOTAS[label], int(parents.openai_label.eq(label).sum())), args.role_cap, roles, args.seed + i)
            for i, label in enumerate(LABELS)
        ], ignore_index=True)
    children = new.loc[new.parent_key.isin(set(selected.parent_key))].merge(
        selected[["parent_key", "panel_parent_id", "jobrole_title", "openai_label", "ai_impact_score"]],
        on="parent_key", how="inner", suffixes=("_new", "_canonical"),
    )
    if not children.jobrole_title_new.map(norm).eq(children.jobrole_title_canonical.map(norm)).all():
        raise ValueError("Role-title mismatch in selected children")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parent_panel = selected[["panel_parent_id", "parent_key", "jobrole_id", "jobrole_title", "sector_title", "keytask_content", "openai_label", "ai_impact_score", "atomic_children"]].rename(columns={"jobrole_title": "canonical_jobrole_title", "keytask_content": "raw_parent_task"})
    child_panel = children[["panel_parent_id", "parent_key", "jobrole_id", "jobrole_title_canonical", "sector_title", "kt_content", "task_no", "task", "label", "score", "openai_label", "ai_impact_score"]].rename(columns={"jobrole_title_canonical": "canonical_jobrole_title", "label": "new_label", "score": "new_score"})
    conditions = [(p, m, alias) for p in ("paper_2023", "new_production") for m, alias in (("gpt_4o", "gpt-4o"), ("gpt_5_2", "gpt-5.2"))]
    rows = []
    for parent in parent_panel.itertuples(index=False):
        for prompt, family, model in conditions:
            rows.append({"call_id": stable(f"{parent.panel_parent_id}\x1fraw\x1f{prompt}\x1f{family}"), "panel_parent_id": parent.panel_parent_id, "panel_child_id": "", "prompt_id": prompt, "model_family": family, "requested_model": model, "task_grain": "raw_parent", "occupation": parent.canonical_jobrole_title, "task_text": parent.raw_parent_task})
    for child in child_panel.itertuples(index=False):
        for prompt, family, model in conditions:
            rows.append({"call_id": stable(f"{child.panel_parent_id}\x1f{child.task_no}\x1f{prompt}\x1f{family}"), "panel_parent_id": child.panel_parent_id, "panel_child_id": stable(f"{child.panel_parent_id}\x1f{child.task_no}\x1f{child.task}"), "prompt_id": prompt, "model_family": family, "requested_model": model, "task_grain": "atomic_child", "occupation": child.canonical_jobrole_title, "task_text": child.task})
    manifest = pd.DataFrame(rows)
    forbidden = {"openai_label", "ai_impact_score", "new_label", "new_score"}
    if forbidden & set(manifest.columns) or manifest.call_id.duplicated().any():
        raise ValueError("Manifest leakage or duplicate call ID")
    parent_panel.to_csv(out / "parent_analysis_panel.csv", index=False)
    child_panel.to_csv(out / "atomic_child_analysis_panel.csv", index=False)
    manifest.to_csv(out / "request_manifest.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({"sampled_parents": len(parent_panel), "sampled_atomic_children": len(child_panel), "outbound_calls": len(manifest), "seed": args.seed, "role_cap": args.role_cap, "sampling_method": args.sampling_method, "sample_size": args.sample_size, "quotas": QUOTAS if args.sampling_method == "label_stratified" else None}, indent=2))
    print(json.dumps({"sampled_parents": len(parent_panel), "sampled_atomic_children": len(child_panel), "outbound_calls": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
