# Hedge Fund Time-Series Forecasting — Kaggle Top 7/982 (Top 1%)

Competition solution for **Hedge fund - Time series forecasting** on Kaggle.

- **Final standing:** 7th / 982 teams (Top 1%)
- **Team:** `TIU`
- **Final leaderboard score:** 0.1748
- **Core model:** LightGBM GBDT
- **Competition:** https://www.kaggle.com/competitions/ts-forecasting
- **Leaderboard:** https://www.kaggle.com/competitions/ts-forecasting/leaderboard

## Problem

The task is to forecast `y_target` from anonymized time-series/tabular signals for rows identified by `code`, `sub_code`, `sub_category`, forecast `horizon`, and `ts_index`.

The competition setup makes validation design especially important: the model must generalize forward in time, so a random train/validation split can give an overly optimistic estimate.

## Solution overview

The final pipeline uses a regularized **LightGBM gradient-boosted tree model** over:

- 86 anonymized numerical features (`feature_*`)
- encoded `code`, `sub_code`, and `sub_category`
- `horizon`
- `ts_index`

### What mattered most

1. **Time-aware walk-forward validation**
   - Fold 1: train first 70%, validate next 10%
   - Fold 2: train first 80%, validate next 10%
   - Fold 3: train first 90%, validate final 10%

2. **Stable model-complexity selection**
   - Train every fold with early stopping.
   - Use the **median best iteration** across folds for the final training configuration instead of trusting one split.

3. **Strong regularization**
   - bounded tree depth / leaf count
   - large minimum leaf size
   - L1/L2 penalties
   - feature subsampling and row bagging

4. **Robust prediction clipping**
   - derive the clipping range from the training target's P0.1/P99.9 range
   - expand it with a 10% buffer to reduce extreme extrapolation without aggressively truncating normal predictions

5. **Memory-efficient large-scale inference**
   - Polars Parquet loading
   - Float32 arrays
   - explicit memory cleanup
   - batched prediction over ~1.45M test rows
   - GPU training when available, with CPU fallback

## Validation design

```text
Time -------------------------------------------------------------->

Fold 1: [============= TRAIN 70% =============][ VAL 10% ][        ]
Fold 2: [================ TRAIN 80% =================][ VAL 10% ][  ]
Fold 3: [==================== TRAIN 90% ====================][VAL 10%]
```

This is intentionally chronological rather than randomized.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── hedge_fund_top7_walk_forward.ipynb
├── src/
│   └── train_predict.py
└── docs/
    └── technical_notes.md
```

## Quick start

The competition dataset is **not included**. Download `train.parquet` and `test.parquet` from Kaggle after accepting the competition rules.

The script looks for data in this order:

1. `HF_DATA_DIR` environment variable
2. `/kaggle/input/ts-forecasting`
3. `/kaggle/input/competitions/ts-forecasting`
4. local `./data`

### Local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export HF_DATA_DIR=/path/to/competition-data
python src/train_predict.py
```

### Kaggle

Upload the repository as a Kaggle notebook/dataset or copy the notebook into a Kaggle Notebook, then point the paths to the mounted competition data.

## Reproducibility note

The original saved notebook did not preserve exact package versions, so `requirements.txt` intentionally avoids invented version pins.

There is also one methodological caveat worth documenting: median-imputation values and category vocabularies are computed once from the full training set before the walk-forward folds. This does **not** expose hidden test labels and does not invalidate final test-time training, but it means the internal fold scores are best interpreted as model-selection estimates rather than a perfectly isolated backtest.

## Why this repository is kept simple

This was a competition solution, not a production trading system. The repository therefore focuses on the actual model-selection and inference logic used for the submission instead of adding unrelated abstractions or dependencies after the fact.
