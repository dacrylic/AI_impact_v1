from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class LeakageReport:
    passed: bool
    messages: List[str]
    warnings: List[str]


def _overlap(a: pd.Series, b: pd.Series) -> int:
    return int(len(set(a.dropna().astype(str)) & set(b.dropna().astype(str))))


def check_split_leakage(
    frame: pd.DataFrame,
    split_assignments: Dict[str, pd.Index],
    group_column: str,
    text_column: str,
    hard_id_columns: Iterable[str] = (),
    soft_id_columns: Iterable[str] = (),
) -> LeakageReport:
    messages: List[str] = []
    warnings: List[str] = []
    splits = list(split_assignments.items())

    for i, (name_a, idx_a) in enumerate(splits):
        for name_b, idx_b in splits[i + 1 :]:
            a = frame.loc[idx_a]
            b = frame.loc[idx_b]
            group_overlap = _overlap(a[group_column], b[group_column])
            if group_overlap:
                messages.append(f"group leakage between {name_a} and {name_b}: {group_overlap}")
            for col in hard_id_columns:
                overlap = _overlap(a[col], b[col])
                if overlap:
                    messages.append(f"id leakage on {col} between {name_a} and {name_b}: {overlap}")
            for col in soft_id_columns:
                overlap = _overlap(a[col], b[col])
                if overlap:
                    warnings.append(f"shared {col} across {name_a} and {name_b}: {overlap}")
            text_overlap = _overlap(a[text_column], b[text_column])
            if text_overlap:
                warnings.append(
                    f"duplicate exact target text across {name_a} and {name_b}: {text_overlap}"
                )

    return LeakageReport(passed=len(messages) == 0, messages=messages, warnings=warnings)
