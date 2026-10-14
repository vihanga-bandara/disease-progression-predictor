import json

from app.train import train


def test_train_writes_model_and_metrics(tmp_path):
    metrics = train(tmp_path)

    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    assert metrics["rmse"] > 0


def test_train_is_deterministic(tmp_path):
    first = train(tmp_path / "first")
    second = train(tmp_path / "second")

    assert first["rmse"] == second["rmse"]
