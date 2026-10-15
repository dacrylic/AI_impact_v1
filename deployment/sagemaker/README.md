# SageMaker Deployment

This is intentionally separate from Heroku. Both routes load the same
`model.joblib`, but SageMaker serves it from a versioned S3 model archive and
keeps endpoint lifecycle inside the organisation's ML stack.

## Package the approved artifact

The archive must contain `model.joblib` at its root:

```bash
rm -rf /tmp/ai-impact-model
mkdir -p /tmp/ai-impact-model
cp models/gpt52-current-basic-v2.joblib /tmp/ai-impact-model/model.joblib
tar -czf /tmp/ai-impact-model.tar.gz -C /tmp/ai-impact-model model.joblib
aws s3 cp /tmp/ai-impact-model.tar.gz s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz
```

## Deploy

```bash
python -m pip install -e '.[sagemaker]'
python deployment/sagemaker/deploy.py \
  --model-data s3://<approved-bucket>/ai-impact/gpt52-current-basic-v2/model.tar.gz \
  --role <sagemaker-execution-role-arn> \
  --endpoint-name ai-impact-gpt52-current-basic-v2
```

The launcher stages `deployment/sagemaker/inference.py` together with the
project package required to load the artifact; MLEs do not need to manually
copy source files into the model archive. The endpoint accepts the same JSON
shape as FastAPI's `/v1/predict` endpoint. Use SageMaker Model Registry,
endpoint data capture, and an approved container image policy when integrating
this path into the organisational platform.

The artifact was trained with scikit-learn `1.6.1`. The default SageMaker
container tag is retained for compatibility with the current launcher, but the
platform owner must confirm the approved image's scikit-learn version before
deployment. If the image cannot load the artifact without warnings or version
drift, rebuild the release artifact inside the approved image and rerun the
sealed-test checks before promotion.
