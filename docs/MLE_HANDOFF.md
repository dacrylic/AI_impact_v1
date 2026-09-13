# MLE Handoff

## What is being handed over

This repository contains the production implementation of the approved
interim scoring model:

- target labels: `E0`, `E1`, `E23` (`E2` and `E3` are merged into `E23`);
- input: AOP (`action` and `object`, with optional `purpose`) and optional
  `jobroleTitle`;
- model: frozen field-aware TF-IDF features plus a CPU `LinearSVC` classifier;
- serving artifact: `artifacts/releases/gpt52-current-basic-v2/model.joblib`;
- primary interim route: FastAPI on Heroku;
- organisational production route: SageMaker, using the separate
  `deployment/sagemaker/` path.

The API does not decompose free text. The upstream extractor supplies AOP. A
pre-composed `keytaskContent` is accepted as a compatibility alternative.

## Start here

1. Read [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) for the evidence and model
   selection history.
2. Read [`docs/MODEL_CARD.md`](MODEL_CARD.md) for intended use and limits.
3. Read [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) for release gates and monitoring.
4. Use [`deployment/sagemaker/README.md`](../deployment/sagemaker/README.md)
   for the separate endpoint path.

## Reproduce training

The approved labelled CSV is intentionally outside Git. Obtain it through the
approved internal data location and record its SHA-256, row count, and label
distribution in the release ticket.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

ai-impact-train-production \
  --input "/secure/path/tasks_reasoning_scores 2.csv" \
  --output-dir artifacts/releases/gpt52-current-basic-v2 \
  --feature-profile basic \
  --model-version gpt52-current-guidance-e0-e1-e23-basic-v2 \
  --refit-full
```

The training command:

- maps source `E2` and `E3` to `E23`;
- excludes contradictory exact duplicate model inputs rather than choosing a
  label arbitrarily;
- collapses unanimous duplicate inputs;
- holds out whole `jobrole_id` groups;
- verifies zero shared roles and zero shared full model inputs across splits;
- writes `training_report.json`, including class metrics, confusion matrix,
  split diagnostics, data hash, and artifact hash;
- writes `sealed_test_rows.csv` for audit review.

Do not promote an artifact unless the report has been reviewed and the sealed
test and leakage checks pass. The `--refit-full` option is used only after the
sealed evaluation is accepted; it refits the selected recipe on all approved,
deduplicated rows for serving.

## Artifact and runtime checks

```bash
python -m pytest -q
python scripts/infer_local.py \
  --model artifacts/releases/gpt52-current-basic-v2/model.joblib \
  --action "prepare" \
  --object "monthly operational performance report"
```

The serving runtime is pinned to scikit-learn `1.6.1` because serialized
scikit-learn artifacts are version-sensitive. The release report's SHA-256 is
the source of truth; do not silently substitute a different artifact.

## SageMaker port

Package the model only:

```bash
cd artifacts/releases/gpt52-current-basic-v2
tar -czf model.tar.gz model.joblib
aws s3 cp model.tar.gz s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz
cd /path/to/AI_impact_v1
python -m pip install -e '.[sagemaker]'
python deployment/sagemaker/deploy.py \
  --model-data s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz \
  --role <sagemaker-execution-role-arn> \
  --endpoint-name ai-impact-gpt52-current-basic-v2
```

The deployment launcher stages the inference handler and the package needed
to unpickle the model. It does not upload the training data or research
artifacts. Before using an approved organisational image, confirm its
scikit-learn version can load the release artifact; otherwise rebuild and
revalidate inside that image.

The SageMaker handler returns the same three fields as the interim API:

```json
{
  "predictedLabel": "E1",
  "reviewScore": 0.31,
  "modelVersion": "gpt52-current-guidance-e0-e1-e23-basic-v2"
}
```

## P3 implementation responsibilities

- move the artifact to the approved model registry/S3 location;
- confirm the approved SageMaker image and scikit-learn compatibility;
- apply the organisation's IAM, VPC, encryption, logging, monitoring, and
  data-retention controls;
- create endpoint autoscaling and rollback policy;
- add contract, load, and parity tests in the organisational CI/CD pipeline;
- register the model version and release evidence;
- keep the Heroku route as the temporary P2 consumption path until cutover.

No API key is required for SageMaker by this repository handler; authentication
and authorisation must be supplied by the organisation's gateway/IAM layer.
