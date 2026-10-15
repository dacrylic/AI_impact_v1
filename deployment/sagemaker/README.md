# SageMaker Deployment

This is intentionally separate from Heroku. Both routes load the same
`model.joblib`, but SageMaker serves it from a versioned S3 model archive and
keeps endpoint lifecycle inside the organisation's ML stack.

## Package the approved artifact

The archive must contain `model.joblib` at its root:

```bash
.venv/bin/python deployment/sagemaker/package_model.py \
  --model models/gpt52-current-basic-v2.joblib \
  --manifest models/gpt52-current-basic-v2.manifest.json \
  --output /tmp/ai-impact-model.tar.gz
aws s3 cp /tmp/ai-impact-model.tar.gz s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz
```

The packaging utility validates the release checksum before creating the
archive. It includes only the model binary; no source dataset, API key,
training report, or research material is uploaded.

## Deploy

```bash
python -m pip install -e '.[sagemaker]'
python deployment/sagemaker/deploy.py \
  --model-data s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz \
  --role <sagemaker-execution-role-arn> \
  --endpoint-name ai-impact-gpt52-current-basic-v2 \
  --framework-version <approved-1.6.1-compatible-tag>
```

The launcher stages `deployment/sagemaker/inference.py` together with the
project package required to load the artifact; MLEs do not need to manually
copy source files into the model archive. The endpoint accepts the same JSON
shape as FastAPI's `/v1/predict` endpoint. Use SageMaker Model Registry,
endpoint data capture, and an approved container image policy when integrating
this path into the organisational platform.

SageMaker requests use the same input alternatives as FastAPI. Send either
`keytaskContent` plus optional `jobroleTitle`, or `action` and `object` plus
optional `purpose` and `jobroleTitle`. Do not send both alternatives together.
The response contains `predictedLabel`, `reviewScore`, and `modelVersion`.

The artifact was trained and serialized with scikit-learn `1.6.1`. There is no
default framework version in the launcher: the deployment command requires an
explicit approved container tag. The platform owner must confirm that the
selected image contains scikit-learn `1.6.1` compatibility before deployment.
If the approved image cannot load the artifact without version warnings or
drift, build an approved custom image with the pinned dependencies in
`pyproject.toml`, then rerun the sealed-test and parity checks before
promotion. Do not silently deploy the artifact in a `1.5.x` runtime.
