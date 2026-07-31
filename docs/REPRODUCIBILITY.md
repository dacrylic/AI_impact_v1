# Reproducibility And Leakage Contract

## Environment

Use Python 3.9 or newer and the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The champion requires only CPU packages. No CUDA, MPS, Torch, or embedding
model is required.

## Split Strategy

The champion uses nested group-stratified cross-validation.

- Outer loop: five `StratifiedGroupKFold` folds, grouped by `jobrole_id`.
- Inner loop: a group-stratified held-out fold from each outer development
  partition selects meta-model regularization and class offsets.
- Cross-fitting: base and auxiliary scores used by the meta-model are created
  from group-disjoint folds within the current development partition.

Consequently, a role is never present in both the training and evaluation side
of a fold, and every row receives one outer-fold prediction.

## Leakage Rules

Hard exclusion applies to `jobrole_id`: a shared role across partitions is a
failure. Other identifiers and exact duplicate task strings are audit warnings
because the same task can legitimately appear under distinct roles. The
champion does not use those identifiers as features.

## Output Files

The champion command writes:

- `fold_metrics.csv`: per-fold nested-CV scores and selected meta settings.
- `oof_predictions.csv`: one outer-fold prediction per input row.
- `run_metadata.json`: target definition, feature contract, and aggregate
  metrics.

Results are ignored by Git so local benchmark runs do not pollute commits.
