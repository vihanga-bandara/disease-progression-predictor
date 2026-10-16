# Disease progression predictor

![CI](https://github.com/vihanga-bandara/disease-progression-predictor/actions/workflows/ci.yml/badge.svg)

Predicts a patient's short-term diabetes progression so a virtual clinic can sort its follow-up
calls by risk. Trained on the scikit-learn diabetes dataset.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
```

## Train the model

```bash
uv run python -m app.train
```

Writes `artifacts/model.joblib` and `artifacts/metrics.json`. The split uses a fixed seed, so every
run gives the same RMSE.

## Run the API

```bash
uv run uvicorn app.main:app --port 8000
```

```bash
curl http://localhost:8000/health
# {"status":"ok","model_version":"dev"}

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
# {"prediction":206.24632515432327,"high_risk":true}
```

`high_risk` is true when the prediction is 200 or more.

All ten features are required numbers. A missing or invalid one gives HTTP 422 with a JSON body
listing each problem, its location and a message:

```json
{"detail":[{"type":"missing","loc":["body","s6"],"msg":"Field required"}],"error":"Invalid request input"}
```

## Checks

```bash
uv run ruff check .
uv run pytest
```

## Docker

Build the self-contained image:

```bash
docker build --build-arg MODEL_VERSION=v0.2 -t disease-progression-predictor:latest .
```

Run the container:

```bash
docker run -d --name disease-predictor -p 8080:80 disease-progression-predictor:latest
```

Smoke test the running container:

```bash
./scripts/smoke_test.sh http://localhost:8080
```

Released images are on GHCR and run with:

```bash
docker pull ghcr.io/vihanga-bandara/disease-progression-predictor:v0.2
docker run -it -p 80:80 ghcr.io/vihanga-bandara/disease-progression-predictor:v0.2
```

## Release

Push a version tag. The release workflow trains the model, builds the Docker image, smoke-tests the
running container, pushes the image to GHCR and publishes a GitHub release with the metrics and the
changelog.

```bash
git tag v0.2
git push origin v0.2
```
