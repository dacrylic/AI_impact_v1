# AI Impact Classifier

CPU-first knowledge-distillation experiments for reproducing validated LLM AI-impact labels from job-role title and task text.

## Current Classification Baseline

The official target is `E0`, `E1`, and `E23`, where `E23` merges source `E2` and `E3`.

- Aggregate out-of-fold macro F1: **0.8056**.
- Inputs: `keytask_content` is required; `jobrole_title` is optional and used by the baseline.
- Evaluation: nested five-fold `StratifiedGroupKFold`, grouped by `jobrole_id`.
- Runtime: CPU-only; no GPU, embedding model, or Torch dependency is required.

Read [the model card](docs/MODEL_CARD.md) before treating the benchmark as a deployment result.

Read [the historical experiment record](docs/HISTORICAL_EXPERIMENTS.md) for
the old-target methods screened, rejected leakage paths, and the rationale for
selecting the CPU-first stack.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproduce The Baseline

```bash
python -m ai_impact_classifier.experiments.run_constrained_oof_stack_cv \
  --input "/path/to/task_ai_impact_details.xlsx" \
  --output-dir results/e0_e1_e23_constrained_oof_stack_cv5
```

The command writes fold metrics, one held-out prediction per row, and run metadata. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the split and leakage contract.

## Production Review Flags

The same sparse model can emit `needs_review` for low-margin and low-coverage
inputs without a second inference model. See the [production output schema](docs/PRODUCTION_OUTPUT_SCHEMA.md).

## Run The API

Train and package the single production artifact from the source workbook:

```bash
ai-impact-train-production --input "/path/to/task_ai_impact_details.xlsx"
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API exposes `GET /health` and `POST /v1/predict`. The latter accepts
`keytask_content`, optional `jobrole_title`, and optional `include_diagnostics`.
Its compact default response is the predicted label plus `review_level`,
`review_score`, and `needs_review`.

## Run The Demo

```bash
streamlit run streamlit_app.py
```

The Streamlit app supports a single task and a CSV batch with a required
`keytask_content` column and optional `jobrole_title` column. It uses the same
production artifact as the API.

For Streamlit Community Cloud, deploy `streamlit_app.py` from the repository
root. `requirements.txt` installs the small runtime dependency set; Docker is
not required.

## Input Contract

Model features may use only `keytask_content` and `jobrole_title`.

IDs, occupation and sector columns, scores, categories, and source labels are never model inputs. The raw `openai_label` is used only as a supervised training target and is unavailable at inference.

## Supported Commands

```bash
ai-impact-constrained-cv --input /path/to/workbook.xlsx
```

Other files under `src/ai_impact_classifier/experiments/` are retained as historical experiments. They may not support the current E0/E1/E23 target.
