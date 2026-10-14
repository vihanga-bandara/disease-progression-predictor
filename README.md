# Disease progression predictor

Service for a virtual diabetes clinic. It predicts a patient's short-term disease progression so the
nurses can sort their follow-up calls by risk. The model is trained on the scikit-learn diabetes
dataset and served as a small HTTP API packaged in a Docker image.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync --frozen
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Working on the repo

- Branch from `main`, one pull request per task, merged with a merge commit after one review.
- Run the checks above before opening the pull request.
- Version tags (`v0.1`, `v0.2`) and the GHCR package are pushed by the repository owner.
