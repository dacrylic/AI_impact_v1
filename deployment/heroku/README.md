# Heroku Deployment

The Heroku web service runs the FastAPI application in `ai_impact_classifier.api`.
It uses the exact same `model.joblib` artifact as local and SageMaker serving.

## Release preparation

1. Train a release artifact with the sealed-test report recorded:

   ```bash
   ai-impact-train-production \
     --input "/path/to/tasks_reasoning_scores 2.csv" \
     --output-dir artifacts/releases/gpt52-current-field-aware-v1 \
     --refit-full
   ```

2. Verify `training_report.json`, then copy the approved artifact to
   `artifacts/current/model.joblib` in the controlled release workspace. Do
   not commit the binary. The Docker build includes that local file even though
   Git ignores it.
3. Set `AI_IMPACT_MODEL_PATH` only when the artifact is mounted elsewhere.

## Deploy

The primary release route is a Docker image built on the controlled machine
that holds the verified artifact. This prevents a GitHub build from silently
deploying an API with no model binary:

```bash
heroku create <app-name>
cp artifacts/releases/gpt52-current-field-aware-v1/model.joblib artifacts/current/model.joblib
heroku container:login
heroku container:push web --app <app-name>
heroku container:release web --app <app-name>
```

After deployment, call `/readyz` before directing traffic to the service. The
root `Procfile` remains available for a platform-managed artifact mount or a
separate build-time artifact retrieval process; it is not the default GitHub
deployment route because the model binary is deliberately excluded from Git.

For a production service, run more than one Uvicorn worker only after checking
memory usage: each worker loads a copy of the sparse model artifact. Use
`/healthz` for liveness and `/readyz` for a model-loaded readiness check.

## Required configuration

| Variable | Purpose |
| --- | --- |
| `AI_IMPACT_MODEL_PATH` | Optional artifact location; default is `artifacts/current/model.joblib`. |
| `WEB_CONCURRENCY` | Optional worker count. Start with `1` for a memory-constrained dyno. |

Heroku is the primary consumption route. It should receive task content and,
when available, title plus AOP context; do not send teacher labels, scores,
reasoning, or identifiers in prediction requests.
