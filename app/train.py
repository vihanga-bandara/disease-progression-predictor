import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("artifacts")


def train(output_dir: Path) -> dict:
    data = load_diabetes(as_frame=True).frame
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    rmse = root_mean_squared_error(y_test, model.predict(X_test))

    metrics = {"rmse": round(rmse, 3), "n_train": len(X_train), "n_test": len(X_test)}
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(OUTPUT_DIR), indent=2))
