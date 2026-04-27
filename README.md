# Overnight WTI Loss Tracking

[Start Here in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/notebooks/overnight_wti_baseline_runner.ipynb)

This repository is structured for one main purpose:

- run overnight WTI experiments in Colab
- save all artifacts to persistent storage
- review train loss, validation loss, forecast plots, and metrics the next morning

## Main Workflow

1. Open [notebooks/overnight_wti_baseline_runner.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/overnight_wti_baseline_runner.ipynb).
2. Leave `SAVE_TO_GOOGLE_DRIVE = True` so outputs survive overnight.
3. Run the notebook from top to bottom.
4. In the morning, read the saved batch summary and the per-run plots.

The default overnight batch is:

- `uni_daily`
- `uni_weekly`
- `multi_daily`
- `multi_weekly`

Each scenario runs:

- `GRU`
- `TimeXer`
- `iTransformer`

## What Gets Saved

Each run writes:

- `loss_history.csv`
- `loss_curve.png`
- `forecast_plot.png`
- `metrics.csv`
- `predictions.csv`
- `config_snapshot.yaml`
- `summary.json`

Each batch writes:

- `summary.csv`
- `report.html`
- `report.md`
- `batch_config_snapshot.yaml`

## Current Baseline Assumptions

- target is treated as `WTI`
- dataset column used for WTI is `Com_CrudeOil`
- start date is fixed at `2011-01-01`
- baseline keeps model hyperparameters at library defaults
- only experiment-level settings are fixed in the batch config:
  - scenario type
  - horizon
  - validation size
  - test size

## Multivariate Feature Sets

The multivariate feature manifests live in:

- [configs/manifests/wti_feature_sets.yaml](/Users/jaeholee/Desktop/newoil/configs/manifests/wti_feature_sets.yaml)

Notes:

- daily uses the exact feature names you specified
- weekly uses the weekly dataset equivalents where needed, for example `*_lag1`
- both daily and weekly manifests become fully usable from early January 2011 after forward fill

## Important Configs

- [configs/batches/overnight_wti_baseline.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_baseline.yaml)
- [configs/batches/overnight_wti_diff.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_diff.yaml)
- [configs/batches/overnight_wti_weight_decay.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_weight_decay.yaml)
- [configs/batches/overnight_wti_diff_weight_decay.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_diff_weight_decay.yaml)
- [configs/manifests/wti_feature_sets.yaml](/Users/jaeholee/Desktop/newoil/configs/manifests/wti_feature_sets.yaml)
- [configs/papers_raw.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_raw.yaml)
- [configs/papers_weekly.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_weekly.yaml)

Use order:

1. `overnight_wti_baseline.yaml`
2. `overnight_wti_weight_decay.yaml`
3. `overnight_wti_diff.yaml`
4. `overnight_wti_diff_weight_decay.yaml`

## Repository Layout

- `data/`: committed raw source files used by the overnight runs
- `configs/manifests/`: target and feature manifests
- `configs/batches/`: batch definitions for overnight experiments
- `src/newoil/`: reusable loading, running, plotting, and reporting code
- `notebooks/`: Colab notebooks for execution and review

## Existing Single-Run Notebook

The earlier single-config notebook is still available here:

- [notebooks/neuralforecast_model_comparison.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/neuralforecast_model_comparison.ipynb)

Use that one for ad hoc debugging.
Use the overnight runner notebook for the main batch workflow.
