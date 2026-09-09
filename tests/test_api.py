from __future__ import annotations

import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from ai_impact_classifier.production.contracts import CLASS_ORDER


def _artifact(path):
    texts = [
        "[TASK] inspect physical equipment",
        "[TASK] write a customer summary",
        "[TASK] analyze operational performance data",
    ]
    labels = ["E0", "E1", "E23"]
    word = TfidfVectorizer(ngram_range=(1, 2))
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    features = FeatureUnion([("word", word), ("char", char)])
    classifier = LinearSVC().fit(features.fit_transform(texts), labels)
    joblib.dump(
        {
            "artifact_schema_version": 1,
            "model_version": "test-v1",
            "classes": list(CLASS_ORDER),
            "features": features,
            "classifier": classifier,
            "word_vectorizer": features.transformer_list[0][1],
        },
        path,
    )


def test_predict_and_health(monkeypatch, tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact)
    monkeypatch.setenv("AI_IMPACT_MODEL_PATH", str(artifact))
    from ai_impact_classifier.api import app, get_predictor

    get_predictor.cache_clear()
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    response = client.post("/v1/predict", json={"keytaskContent": "Write a customer summary", "jobroleTitle": "Analyst"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predictedLabel"] in CLASS_ORDER
    assert 0 <= payload["reviewScore"] <= 1
    assert payload["modelVersion"] == "test-v1"
    assert set(payload) == {"predictedLabel", "reviewScore", "modelVersion"}


def test_rejects_unknown_fields(monkeypatch, tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact)
    monkeypatch.setenv("AI_IMPACT_MODEL_PATH", str(artifact))
    from ai_impact_classifier.api import app, get_predictor

    get_predictor.cache_clear()
    response = TestClient(app).post("/v1/predict", json={"keytaskContent": "Write a report", "teacherScore": 0.7})
    assert response.status_code == 422


def test_accepts_aop_without_precomposed_task(monkeypatch, tmp_path):
    artifact = tmp_path / "model.joblib"
    _artifact(artifact)
    monkeypatch.setenv("AI_IMPACT_MODEL_PATH", str(artifact))
    from ai_impact_classifier.api import app, get_predictor

    get_predictor.cache_clear()
    response = TestClient(app).post(
        "/v1/predict",
        json={"action": "Write", "object": "a customer summary", "purpose": "support a review"},
    )
    assert response.status_code == 200
    assert response.json()["predictedLabel"] in CLASS_ORDER
