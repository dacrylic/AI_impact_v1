# AI Impact Classifier

CPU-first knowledge-distillation experiments for reproducing validated LLM AI-impact labels from job-role title and task text.

## Champion

The current CPU-only champion is a nested, role-grouped sparse ensemble with
an aggregate out-of-fold macro F1 of **0.7803**.

- Target: `E0`, `E3`, and `E12`, where `E12` merges source `E1` and `E2`.
- Inputs: `keytask_content` is required; `jobrole_title` is optional but used
  by the champion.
- No GPU, embedding model, or Torch dependency is required.
- Evaluation: nested five-fold `StratifiedGroupKFold`, grouped by
  `jobrole_id`.

Read [the model card](docs/MODEL_CARD.md) before treating the benchmark as a
deployment result. It documents the method, class-level metrics, and the
remaining limitations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base installation is CPU-only. Historical embedding and Transformer
experiments require `pip install -e '.[embeddings]'` and are not part of the
supported workflow.

## Reproduce The Champion

```bash
python -m ai_impact_classifier.experiments.run_cpu_aux_retrieval_oof_cv \
  --input "/path/to/task_ai_impact_details.xlsx" \
  --output-dir results/cpu_aux_retrieval_oof_cv5
```

The command writes fold metrics, one out-of-fold prediction per row, and run
metadata. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the split
and leakage contract.

## Input Contract

Model features may use only:

- `keytask_content`
- `jobrole_title`

The following are never model inputs: IDs, occupation and sector columns,
scores, categories, and source labels. The raw `openai_label` is used only as
a training target; the champion additionally uses the source E1/E2 split as a
training-only auxiliary target. It is not present at inference.

## Supported Commands

```bash
# Current CPU champion benchmark
ai-impact-cpu-aux-retrieval-cv --input /path/to/workbook.xlsx

# Simpler CPU baseline used for comparison
ai-impact-constrained-cv --input /path/to/workbook.xlsx

# Error-review material for the E3 boundary
ai-impact-e3-audit --input /path/to/workbook.xlsx \
  --predictions results/cpu_aux_retrieval_oof_cv5/oof_predictions.csv
```

Other files under `src/ai_impact_classifier/experiments/` are retained as
historical experiments. They are not supported benchmark entry points and may
need optional dependencies.
