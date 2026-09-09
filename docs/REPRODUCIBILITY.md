# Reproducibility And Leakage Contract

## Environment

Use Python 3.9 or newer and the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[api,dev]'
```

The finalized classifier and API require only CPU packages. No Torch, GPU,
embedding download, or live LLM call is required at inference.

## Release Split

`ai_impact_classifier.production.train` builds a deterministic three-way split
from `StratifiedGroupKFold` using `jobrole_id` as the group:

1. one of five outer folds is held aside as the 20% sealed test set;
2. one group-stratified fold within the remaining development rows is the
   validation set; and
3. the other development rows train the fixed release configuration.

This produces roughly 64% train, 16% validation and 20% test. The source-label
proportions are preserved as closely as group sizes allow, while no job role
can appear in more than one partition.

## Full-Input Duplicate Rule

Before splitting, the trainer creates the complete inference representation:
task, optional title, and optional AOP fields. It then:

- excludes exact complete inputs with more than one target label;
- retains one deterministic representative for each repeated same-label input;
- asserts zero shared complete inputs across every pair of partitions; and
- asserts zero shared `jobrole_id` values across every pair of partitions.

The model itself never receives `jobrole_id`, source scores, labels, sector,
identifiers, or LLM reasoning as input features.

## Release Evidence

Each training run writes a `training_report.json` containing source and model
SHA-256 checksums, row counts, duplicate treatment, split seed, exact feature
weights, leakage checks, confusion matrix, class metrics, and macro/weighted
F1. Keep that report beside the release artifact in the governed artifact
store.
