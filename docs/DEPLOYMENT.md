# Release And Deployment Guide

## Release gates

Before promoting a model artifact, record the following in the release ticket:

1. approved label policy: current POC guidance plus GPT-5.2;
2. source data version, hash, row count and label distribution;
3. strict split and duplicate-input checks from `training_report.json`;
4. sealed-test macro F1, weighted F1, class-level metrics and confusion matrix;
5. model artifact checksum and `modelVersion`;
6. responsible approver and release date.

The `model.joblib` artifact is a release binary. Keep it in the approved
artifact registry or storage location, not in ordinary Git history.

## HTTP contract

`POST /v1/predict` accepts:

```json
{
  "action": "prepare",
  "object": "monthly operational performance report",
  "purpose": "support management review",
  "jobroleTitle": "Operations Analyst",
}
```

The primary contract is `action` plus `object`, with `purpose` optional. The
service joins those values deterministically into the exact canonical task form
used in training. It does not decompose raw job-posting prose or infer A/O/P
values. `keytaskContent` is supported only as a compatibility alternative for
upstreams that already emit the composed task. Requests must not include LLM
labels, scores, reasoning, task IDs, role IDs, sector, or other restricted
source columns.

The response is always:

```json
{
  "predictedLabel": "E0 | E1 | E23",
  "reviewScore": 0.0,
  "modelVersion": "gpt52-current-guidance-e0-e1-e23-v1"
}
```

`reviewScore` is a review-priority signal. It rises when the top two class
scores are close, the input has low lexical coverage against the training
vocabulary, or the task is unusually short. It is not a probability.

## Operations

- `GET /healthz` checks process liveness.
- `GET /readyz` checks that the model artifact has loaded.
- `GET /v1/model` returns the target and model version.
- `POST /v1/predict:batch` supports up to 100 rows per request.

Alert on readiness failures, high review-rate changes, sudden class-mix
changes, elevated low-coverage requests and latency. Sample high-review and
low-review cases regularly for human quality review. Track input cohorts so
that a new upstream task extractor or job-posting source cannot silently
change the live distribution.

## Deployment boundaries

Heroku is the primary serving route and uses the root `Procfile`. SageMaker is
a separate deployment path that packages the same `model.joblib` into a
versioned S3 model archive. Do not retrain models differently for the two
platforms; deployment differences must not create label-policy drift.
