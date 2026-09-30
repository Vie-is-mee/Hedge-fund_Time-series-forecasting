# Technical Notes

## 1. Competition-aligned metric

The notebook evaluates predictions with the weighted normalized score used during development:

```python
def weighted_rmse_score(y_target, y_pred, w):
    denom = np.sum(w * y_target ** 2)
    if denom == 0:
        return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))
```

Sample weights are passed to LightGBM during training and to the evaluation function. They are not used as predictive features.

## 2. Why walk-forward CV

A random split mixes earlier and later observations and can underestimate temporal distribution shift. The solution instead validates on progressively later `ts_index` ranges.

The third fold is closest to the hidden future period, while the mean and dispersion across folds provide a rough robustness check.

## 3. Final boosting-round selection

Every fold uses early stopping. Rather than selecting the best iteration from a single fold, the solution takes the median of the three best-iteration values. This reduces sensitivity to one validation window.

## 4. Regularization

The main LightGBM controls are:

- `num_leaves = 63`
- `max_depth = 6`
- `min_data_in_leaf = 500`
- `lambda_l1 = 1.0`
- `lambda_l2 = 1.0`
- `feature_fraction = 0.70`
- `bagging_fraction = 0.80`
- `bagging_freq = 5`
- `learning_rate = 0.05`

The design favors a conservative model because the features are anonymized and the public leaderboard can be noisy.

## 5. Tail clipping

The clipping bounds are computed from P0.1 and P99.9 of the training target, then expanded by 10% of the percentile range. This only targets extreme extrapolation.

## 6. Memory strategy

The pipeline uses:

- Polars Parquet scans
- Float32 feature/target arrays
- Int16/Int32 metadata where appropriate
- explicit garbage collection
- inference batches of 200,000 rows

The expected test size in the original solution is 1,447,107 rows.

## 7. Caveat: fold preprocessing

Imputation statistics and category mappings are learned once from the full training file before the chronological CV masks are applied. This is a mild source of validation contamination for model-selection estimates, even though no hidden test targets are used.

For a stricter research backtest, fit those preprocessing statistics independently inside each fold.
