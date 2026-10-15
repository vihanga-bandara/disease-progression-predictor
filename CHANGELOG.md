# Changelog

## v0.2

- Ridge instead of LinearRegression, alpha 10 picked by 5-fold cross-validation. Same split and seed as v0.1.
- New `high_risk` flag in the predict response, true when the predicted progression is 200 or more.

| | v0.1 | v0.2 |
|---|---|---|
| RMSE (89 held-out rows) | 53.853 | 53.626 |
| Precision at 200 | - | 0.833 |
| Recall at 200 | - | 0.417 |

The dataset is small and some features are correlated, so the penalty on the coefficients generalises
a bit better than plain least squares. The flag gives the nurses something to filter on.

## v0.1

Baseline model: StandardScaler + LinearRegression on the scikit-learn diabetes dataset, 80/20 split
with seed 42.

RMSE on the 89 held-out rows: 53.853
