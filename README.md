# Disease progression predictor

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

## Checks

```bash
uv run ruff check .
uv run pytest
```
