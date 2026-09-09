# Historical GPT-4 Sensitivity Replay

## Purpose

This is a quick reproducibility check for the historical AI-impact labels. The
cited paper describes an early version of GPT-4, but does not publish the exact
API model ID or the original production prompt. The replay therefore tests a
plausible historical snapshot, `gpt-4-0613`, using the reconstructed paper
prompt. It cannot prove which model produced the original dashboard labels.

## Design

- **Rows:** the same 978 sampled raw parent tasks used in the representative
  factorial study.
- **Inputs:** occupation/job-role title plus the original raw task text.
- **Prompt:** `results/factorial_restart_v1/prompts/paper_2023_system_reconstruction.txt`
  and `paper_2023_user_template.txt`.
- **Model:** requested and resolved `gpt-4-0613` for every row.
- **API:** legacy Chat Completions endpoint, because this historical snapshot
  rejects the newer Responses API structured-output schema and JSON mode.
- **Target comparison:** the historical `openai_label` on the same 978 rows,
  plus the existing paper-prompt `gpt-4o-2024-08-06` replay.
- **Leakage:** no historical labels or impact scores were sent in the prompt.

## Results

| Source | E0 | Impacted (E1/E2/E3) |
|---|---:|---:|
| Historical labels | 76.38% | 23.62% |
| Paper prompt + `gpt-4o-2024-08-06` | 52.76% | 47.24% |
| Paper prompt + `gpt-4-0613` | 45.91% | 54.09% |

The historical-snapshot replay is therefore **farther from** the historical
23.6% impacted result than the existing GPT-4o replay. It does not support the
claim that the old dashboard was reproduced by simply calling the paper-era
GPT-4 model.

Against the historical labels, the `gpt-4-0613` replay had 60.74% exact
four-class agreement and 65.24% E0-versus-impacted agreement. Its confusion
matrix uses historical labels as rows and replay labels as columns:

| | Replay E0 | Replay E1 | Replay E2 | Replay E3 |
|---|---:|---:|---:|---:|
| Historical E0 | 428 | 201 | 118 | 0 |
| Historical E1 | 3 | 90 | 28 | 0 |
| Historical E2 | 17 | 14 | 75 | 0 |
| Historical E3 | 1 | 0 | 2 | 1 |

Among historical E0 rows, 319 of 747 became impacted in this replay. Among
historically impacted rows, 210 of 231 remained impacted.

## Interpretation

The old labels remain unrecovered under this plausible paper-era setup. The
gap is not explained by model age alone in this check. Remaining unknowns
include the unpreserved historical prompt, exact API parameters, sampling or
post-processing, and whether the original labels were manually adjusted.

The correct stakeholder statement is: **we can reproduce neither the exact
historical model call nor the 23.6% result from the surviving evidence.** This
replay strengthens the case for treating the historical dashboard percentage
as a legacy, non-reproducible baseline rather than as a validated benchmark.

## Artifacts

- Manifest: `results/paper_gpt4_0613_check/request_manifest.csv`
- Responses: `results/paper_gpt4_0613_check/responses.csv`
- Existing comparison arm: `results/factorial_representative_v1/final_analysis/raw_complete_cell_labels.csv`
