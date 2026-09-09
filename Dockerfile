FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"
COPY artifacts/current ./artifacts/current

ENV AI_IMPACT_MODEL_PATH=artifacts/current/model.joblib
ENV PORT=8000
CMD ["sh", "-c", "uvicorn ai_impact_classifier.api:app --host 0.0.0.0 --port ${PORT}"]
