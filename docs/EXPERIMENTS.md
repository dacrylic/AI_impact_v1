# Experiments And Decision Record

## Decision

The approved production label policy is **current POC guidance applied by
GPT-5.2 to normalized/decomposed AOP-style tasks**. The production classifier
distils that policy into the three-class target `E0`, `E1`, and `E23`, where
the source labels `E2` and `E3` are merged.

This is a policy decision, not a claim that the labels are independently
adjudicated ground truth. The classifier must reproduce the approved policy
consistently, cheaply and with suitable review controls.

## Why this target was selected

The factorial study held a random matched panel of tasks fixed while varying:

1. scoring guidance: historical dashboard guidance, current POC guidance, and
   current POC guidance plus historical examples;
2. LLM version: GPT-4o and GPT-5.2; and
3. task format: original parent task and normalized/decomposed atomic task.

The analysis found that scoring guidance and LLM version drive the main
distribution shift. Normalization/decomposition changes E0 by about zero to
three percentage points after parent-weighted comparison, while improving the
operational scoring unit and traceability. The approved target therefore uses
the current POC guidance, GPT-5.2, and normalized/decomposed tasks.

The historical dashboard's 23.6% AI-impacted result was strongly reproducible
when the recovered historical method was used. It remains a separately
versioned historical measure; the approved live-tool policy must not be
presented as a revision to that national statistic.

## Distillation evidence

Within the approved-target experiments, the selected production trade-off was
a field-aware sparse TF-IDF plus LinearSVC classifier. It reached **0.8188
macro F1** on a sealed test with zero shared job roles and zero shared full
input strings. A frozen MPNet blend reached 0.8250 but introduced roughly an
order of magnitude more inference latency. The sparse classifier is therefore
the release candidate for the live API.

The final release build recreated that configuration: **0.8183 macro F1**,
**0.8236 weighted F1**, and **0.8244 accuracy** on the 15,410-row sealed test.
It deduplicated 1,160 same-label complete inputs, excluded no conflicting
complete inputs, and recorded zero shared roles or complete inputs across the
split. The release report captures the source-data hash and artifact checksum.

Run it with:

```bash
.venv/bin/python -m ai_impact_classifier.production.train \
  --input "/path/to/tasks_reasoning_scores 2.csv" \
  --output-dir artifacts/releases/gpt52-current-field-aware-v1 \
  --refit-full
```

## Research archive

The curated management deck and the factorial analysis documentation are in
[`research/factorial-study/`](../research/factorial-study/). The large raw
response files, temporary slide renders, local model artifacts and source
datasets stay out of Git because they contain source material, are
reproducible from governed inputs, or are release-managed binaries.

Relevant detailed documents:

- [factorial-study archive](../research/factorial-study/)

## What is deliberately not production code

The scripts under `src/ai_impact_classifier/experiments/` preserve historical
baselines, embedding screens, regression attempts and specialist ensembles.
They are not imported by the FastAPI application or deployment code. This
keeps the production dependency footprint CPU-only and avoids deploying a
method merely because it was once explored.
