import importlib

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "age": 0.02,
    "sex": -0.044,
    "bmi": 0.06,
    "bp": -0.03,
    "s1": -0.02,
    "s2": 0.03,
    "s3": -0.02,
    "s4": 0.02,
    "s5": 0.02,
    "s6": -0.001,
}


def client_for_version(monkeypatch, tmp_path, version="v0.1"):
    dataset = load_diabetes(as_frame=True)
    model = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    model.fit(dataset.data, dataset.target)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_VERSION", version)
    import app.main

    importlib.reload(app.main)
    return TestClient(app.main.app)


def test_health(monkeypatch, tmp_path):
    client = client_for_version(monkeypatch, tmp_path, "v0.1")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "v0.1"}


def test_prediction_is_numeric(monkeypatch, tmp_path):
    client = client_for_version(monkeypatch, tmp_path)
    response = client.post("/predict", json=FEATURES)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], (int, float))
    assert np.isfinite(response.json()["prediction"])


def test_missing_feature_returns_422_json(monkeypatch, tmp_path):
    client = client_for_version(monkeypatch, tmp_path)
    payload = dict(FEATURES)
    payload.pop("s6")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert "s6" in response.text


def test_non_finite_feature_returns_422(monkeypatch, tmp_path):
    client = client_for_version(monkeypatch, tmp_path)
    payload = dict(FEATURES)
    payload["bmi"] = "NaN"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422