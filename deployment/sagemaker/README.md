# SageMaker Deployment

This is intentionally separate from Heroku. Both routes load the same
`model.joblib`, but SageMaker serves it from a versioned S3 model archive and
keeps endpoint lifecycle inside the organisation's ML stack.

## Package the approved artifact

The archive must contain `model.joblib` at its root:

```bash
cd artifacts/releases/gpt52-current-field-aware-v1
tar -czf model.tar.gz model.joblib
aws s3 cp model.tar.gz s3://<approved-bucket>/ai-impact/gpt52-current-field-aware-v1/model.tar.gz
```

## Deploy

```bash
python -m pip install -e '.[sagemaker]'
python deployment/sagemaker/deploy.py \
  --model-data s3://<approved-bucket>/ai-impact/gpt52-current-field-aware-v1/model.tar.gz \
  --role <sagemaker-execution-role-arn> \
  --endpoint-name ai-impact-gpt52-current-field-aware-v1
```

The endpoint accepts the same JSON shape as FastAPI's `/v1/predict` endpoint.
Use SageMaker Model Registry, endpoint data capture, and an approved container
image policy when integrating this stub into the organisational platform.
