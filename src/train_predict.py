"""Top-7 Kaggle solution for Hedge fund - Time series forecasting.

This script is a cleaned, function-oriented version of the competition notebook.
It preserves the original modeling logic while making the workflow easier to read
and run from a GitHub repository.
"""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import psutil

RANDOM_SEED = 42
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 150
EXPECTED_TEST_ROWS = 1_447_107
PRED_BATCH_SIZE = 200_000
WF_FOLDS = [(0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
CLIP_LOW_PCTILE = 0.1
CLIP_HIGH_PCTILE = 99.9
CLIP_BUFFER = 0.10

META_COLS = ["id", "code", "sub_code", "sub_category", "horizon", "ts_index"]
TARGET_COL = "y_target"
WEIGHT_COL = "weight"
CAT_COLS = ["code", "sub_code", "sub_category"]


def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def resolve_data_dir() -> Path:
    candidates = [
        os.environ.get("HF_DATA_DIR"),
        "/kaggle/input/ts-forecasting",
        "/kaggle/input/competitions/ts-forecasting",
        "./data",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "train.parquet").exists():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not find train.parquet. Set HF_DATA_DIR or place data under ./data."
    )


def weighted_rmse_score(y_target: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    denom = np.sum(w * y_target**2)
    if denom == 0:
        return 0.0
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    return float(np.sqrt(1.0 - np.clip(ratio, 0.0, 1.0)))


def detect_device() -> str:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return "cuda" if result.returncode == 0 else "cpu"
    except Exception:
        return "cpu"


def lightgbm_params(device: str) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 6,
        "min_data_in_leaf": 500,
        "min_sum_hessian_in_leaf": 10.0,
        "min_gain_to_split": 0.01,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "bagging_freq": 5,
        "learning_rate": 0.05,
        "max_cat_threshold": 256,
        "device_type": device,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": -1,
        "seed": RANDOM_SEED,
        "num_threads": -1,
    }


def fit_with_cpu_fallback(params: dict, *args, **kwargs) -> lgb.Booster:
    try:
        return lgb.train(params, *args, **kwargs)
    except lgb.basic.LightGBMError as exc:
        if params.get("device_type") != "cuda":
            raise
        print(f"GPU training failed ({exc}); retrying on CPU.")
        cpu_params = dict(params)
        cpu_params["device_type"] = "cpu"
        return lgb.train(cpu_params, *args, **kwargs)


def main() -> None:
    data_dir = resolve_data_dir()
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    output_path = Path("/kaggle/working/submission_v3.csv") if Path("/kaggle/working").exists() else Path("submission_v3.csv")

    schema = pl.scan_parquet(train_path).collect_schema()
    all_cols = schema.names()
    feature_cols = [c for c in all_cols if c.startswith("feature_")]
    needed_cols = [
        c for c in META_COLS + feature_cols + [TARGET_COL, WEIGHT_COL] if c in all_cols
    ]

    print(f"LightGBM: {lgb.__version__}")
    print(f"Polars:   {pl.__version__}")
    print(f"Features: {len(feature_cols)}")
    print(f"RAM:      {ram_mb():.0f} MB")

    train = (
        pl.scan_parquet(train_path)
        .select(needed_cols)
        .with_columns(
            [pl.col(c).cast(pl.Float32) for c in feature_cols]
            + [
                pl.col(TARGET_COL).cast(pl.Float32),
                pl.col(WEIGHT_COL).cast(pl.Float32),
                pl.col("horizon").cast(pl.Int16),
                pl.col("ts_index").cast(pl.Int32),
            ]
        )
        .collect()
    )

    y_for_clip = train[TARGET_COL].to_numpy()
    clip_lo_raw = float(np.percentile(y_for_clip, CLIP_LOW_PCTILE))
    clip_hi_raw = float(np.percentile(y_for_clip, CLIP_HIGH_PCTILE))
    clip_range = clip_hi_raw - clip_lo_raw
    clip_low = clip_lo_raw - CLIP_BUFFER * clip_range
    clip_high = clip_hi_raw + CLIP_BUFFER * clip_range
    del y_for_clip

    impute_vals: dict[str, float] = {}
    for col in feature_cols:
        series = train[col].drop_nulls()
        try:
            series = series.drop_nans()
        except Exception:
            pass
        median = series.median()
        impute_vals[col] = float(median) if median is not None else 0.0

    train = train.with_columns(
        [
            pl.col(c).fill_null(impute_vals[c]).fill_nan(impute_vals[c])
            for c in feature_cols
        ]
    )

    cat_maps: dict[str, dict[str, int]] = {}
    enc_cols: list[str] = []
    for col in CAT_COLS:
        uniques = sorted(train[col].cast(pl.Utf8).unique().to_list())
        mapping = {value: i for i, value in enumerate(uniques)}
        cat_maps[col] = mapping
        enc_col = f"{col}_enc"
        enc_cols.append(enc_col)
        train = train.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .replace(mapping, default=-1)
            .cast(pl.Int16)
            .alias(enc_col)
        )

    all_features = feature_cols + enc_cols + ["horizon", "ts_index"]
    x_arr = train.select(all_features).to_numpy(allow_copy=True).astype(np.float32)
    y_arr = train[TARGET_COL].to_numpy().astype(np.float32)
    w_arr = train[WEIGHT_COL].to_numpy().astype(np.float32)
    ts_arr = train["ts_index"].to_numpy().astype(np.int32)
    del train
    gc.collect()

    device = detect_device()
    params = lightgbm_params(device)
    dataset_params = {"max_bin": 127, "histogram_pool_size": 512}
    cat_indices = [all_features.index(c) for c in enc_cols]

    ts_min = int(ts_arr.min())
    ts_max = int(ts_arr.max())
    ts_span = ts_max - ts_min
    fold_best_iters: list[int] = []
    fold_val_scores: list[float] = []

    print(f"\nWalk-forward CV on ts_index [{ts_min}, {ts_max}]")
    for fold_num, (train_end_pct, val_end_pct) in enumerate(WF_FOLDS, start=1):
        ts_train_end = ts_min + ts_span * train_end_pct
        ts_val_end = ts_min + ts_span * val_end_pct
        train_mask = ts_arr <= ts_train_end
        val_mask = (ts_arr > ts_train_end) & (ts_arr <= ts_val_end)

        x_val = x_arr[val_mask].copy()
        y_val = y_arr[val_mask].copy()
        w_val = w_arr[val_mask].copy()

        ds_train = lgb.Dataset(
            x_arr[train_mask],
            label=y_arr[train_mask],
            weight=w_arr[train_mask],
            feature_name=all_features,
            categorical_feature=cat_indices,
            params=dataset_params,
            free_raw_data=True,
        )
        ds_val = lgb.Dataset(
            x_val,
            label=y_val,
            weight=w_val,
            feature_name=all_features,
            categorical_feature=cat_indices,
            params=dataset_params,
            reference=ds_train,
            free_raw_data=True,
        )

        fold_model = fit_with_cpu_fallback(
            params,
            ds_train,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[ds_train, ds_val],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=300),
            ],
        )

        pred = fold_model.predict(x_val, num_iteration=fold_model.best_iteration)
        pred = np.clip(pred, clip_low, clip_high)
        score = weighted_rmse_score(y_val, pred, w_val)
        fold_best_iters.append(fold_model.best_iteration)
        fold_val_scores.append(score)
        print(f"Fold {fold_num}: score={score:.6f}, best_iter={fold_model.best_iteration}")

        del ds_train, ds_val, fold_model, x_val, y_val, w_val, pred
        gc.collect()

    final_best_iter = int(np.median(fold_best_iters))
    print(f"CV mean={np.mean(fold_val_scores):.6f}, std={np.std(fold_val_scores):.6f}")
    print(f"Median best iteration={final_best_iter}")

    # Final chronological 90/10 fit, matching the original notebook.
    final_split = ts_min + ts_span * WF_FOLDS[-1][0]
    final_train_mask = ts_arr <= final_split
    final_val_mask = ts_arr > final_split

    x_final_val = x_arr[final_val_mask].copy()
    y_final_val = y_arr[final_val_mask].copy()
    w_final_val = w_arr[final_val_mask].copy()

    ds_final_train = lgb.Dataset(
        x_arr[final_train_mask],
        label=y_arr[final_train_mask],
        weight=w_arr[final_train_mask],
        feature_name=all_features,
        categorical_feature=cat_indices,
        params=dataset_params,
        free_raw_data=True,
    )
    ds_final_val = lgb.Dataset(
        x_final_val,
        label=y_final_val,
        weight=w_final_val,
        feature_name=all_features,
        categorical_feature=cat_indices,
        params=dataset_params,
        reference=ds_final_train,
        free_raw_data=True,
    )

    model = fit_with_cpu_fallback(
        params,
        ds_final_train,
        num_boost_round=final_best_iter + 50,
        valid_sets=[ds_final_train, ds_final_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )

    val_pred = model.predict(x_final_val, num_iteration=model.best_iteration)
    val_pred = np.clip(val_pred, clip_low, clip_high)
    final_score = weighted_rmse_score(y_final_val, val_pred, w_final_val)
    print(f"Final chronological validation score={final_score:.6f}")

    del x_arr, y_arr, w_arr, ts_arr
    del x_final_val, y_final_val, w_final_val, val_pred
    del ds_final_train, ds_final_val
    gc.collect()

    importance = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 20 features by gain:")
    print(importance.head(20).to_string(index=False))

    test_schema = set(pl.scan_parquet(test_path).collect_schema().names())
    test_needed = [c for c in META_COLS + feature_cols if c not in [TARGET_COL, WEIGHT_COL]]
    test = (
        pl.scan_parquet(test_path)
        .select([c for c in test_needed if c in test_schema])
        .with_columns(
            [
                pl.col(c)
                .fill_null(impute_vals[c])
                .fill_nan(impute_vals[c])
                .cast(pl.Float32)
                for c in feature_cols
            ]
            + [pl.col("horizon").cast(pl.Int16), pl.col("ts_index").cast(pl.Int32)]
        )
        .collect()
    )

    for col in CAT_COLS:
        enc_col = f"{col}_enc"
        test = test.with_columns(
            pl.col(col)
            .cast(pl.Utf8)
            .replace(cat_maps[col], default=-1)
            .cast(pl.Int16)
            .alias(enc_col)
        )

    test_ids = test["id"].to_numpy()
    x_test = test.select(all_features).to_numpy(allow_copy=True).astype(np.float32)
    del test
    gc.collect()

    if len(test_ids) != EXPECTED_TEST_ROWS:
        print(f"Warning: expected {EXPECTED_TEST_ROWS:,} rows, found {len(test_ids):,}.")

    preds = np.empty(len(test_ids), dtype=np.float32)
    for start in range(0, len(test_ids), PRED_BATCH_SIZE):
        end = min(start + PRED_BATCH_SIZE, len(test_ids))
        preds[start:end] = model.predict(
            x_test[start:end], num_iteration=model.best_iteration
        ).astype(np.float32)

    del x_test
    gc.collect()
    preds = np.clip(preds, clip_low, clip_high)

    submission = pd.DataFrame({"id": test_ids, "prediction": preds})
    if submission["prediction"].isna().any() or np.isinf(submission["prediction"].values).any():
        raise ValueError("Submission contains NaN or Inf predictions.")
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
