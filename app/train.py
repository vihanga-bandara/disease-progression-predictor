import json
from pathlib import Path

import joblib
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_score, recall_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("artifacts")
ALPHAS = [0.01, 0.1, 1, 10, 100]
HIGH_RISK_THRESHOLD = 200


def train(output_dir: Path) -> dict:
    data = load_diabetes(as_frame=True).frame
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    search = GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": ALPHAS},
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    predicted = model.predict(X_test)
    high_risk = y_test >= HIGH_RISK_THRESHOLD
    flagged = predicted >= HIGH_RISK_THRESHOLD
    metrics = {
        "rmse": round(root_mean_squared_error(y_test, predicted), 3),
        "alpha": search.best_params_["ridge__alpha"],
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "precision": round(precision_score(high_risk, flagged), 3),
        "recall": round(recall_score(high_risk, flagged), 3),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(OUTPUT_DIR), indent=2))
