# Overnight WTI Loss Tracking

[Start Here in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/notebooks/overnight_wti_baseline_runner.ipynb)

[WTI Review in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/notebooks/weekly_wti_review_runner.ipynb)

This repository is structured for one main purpose:

- run overnight WTI experiments in Colab
- save all artifacts to persistent storage
- review train loss, validation loss, forecast plots, and metrics the next morning

## Main Workflow

1. Open [notebooks/overnight_wti_baseline_runner.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/overnight_wti_baseline_runner.ipynb).
2. For a quick risk check, first set `BATCH_CONFIG_RELATIVE_PATH = "configs/batches/smoke_wti_baseline.yaml"`.
3. Leave `SAVE_TO_GOOGLE_DRIVE = True` so outputs survive overnight.
4. Run the notebook from top to bottom.
5. If the smoke batch finishes cleanly, switch back to `configs/batches/overnight_wti_baseline.yaml` and start the real overnight run.
6. In the morning, read the saved batch summary and the per-run plots.

## WTI Review Workflow

1. Open [notebooks/weekly_wti_review_runner.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/weekly_wti_review_runner.ipynb).
2. If you received an updated weekly CSV, upload it to Google Drive and set `UPDATED_WEEKLY_DATA_SOURCE_PATH`.
3. Leave the default `BATCH_CONFIG_RELATIVE_PATHS` as-is for the company-facing four-run plan:
   `daily_wti_h12_mse_report.yaml`, `daily_wti_h12_mse_scaled_regularized.yaml`, `weekly_wti_h2_mse_report.yaml`, and `weekly_wti_h2_mse_scaled_regularized.yaml`.
4. Run the notebook from top to bottom.
5. Read `wti_review_combined_summary.csv` plus each batch `report.html` to see which change improved the issue.

## Local VSCode Workflow

1. Install the runtime once in your selected VSCode interpreter:
   `pip install neuralforecast==3.1.7 pandas matplotlib openpyxl pyyaml`
2. Open the repo in VSCode and run the task `Run Company Plan`.
3. The task auto-detects `gpu`, `mps`, or `cpu` and runs the same daily+weekly company plan locally.
4. Read `outputs/company_plan_*/combined_summary.csv` and each batch `report.html`.

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

Loss-curve scaling is configurable through the batch `report.loss_scale` field:

- `linear`
- `log`
- `symlog`

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

- [configs/batches/smoke_wti_baseline.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/smoke_wti_baseline.yaml)
- [configs/batches/overnight_wti_baseline.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_baseline.yaml)
- [configs/batches/overnight_wti_diff.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_diff.yaml)
- [configs/batches/overnight_wti_weight_decay.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_weight_decay.yaml)
- [configs/batches/overnight_wti_diff_weight_decay.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_diff_weight_decay.yaml)
- [configs/batches/smoke_wti_short_window_log.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/smoke_wti_short_window_log.yaml)
- [configs/batches/overnight_wti_short_window_log.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_short_window_log.yaml)
- [configs/batches/smoke_wti_myoil_h3.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/smoke_wti_myoil_h3.yaml)
- [configs/batches/overnight_wti_myoil_h3.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/overnight_wti_myoil_h3.yaml)
- [configs/batches/weekly_wti_h2_mse_report.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/weekly_wti_h2_mse_report.yaml)
- [configs/batches/weekly_wti_h2_mse_raw_report.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/weekly_wti_h2_mse_raw_report.yaml)
- [configs/batches/weekly_wti_h2_mse_scaled_regularized.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/weekly_wti_h2_mse_scaled_regularized.yaml)
- [configs/batches/weekly_wti_h2_mse_scaled_val_sweep.yaml](/Users/jaeholee/Desktop/newoil/configs/batches/weekly_wti_h2_mse_scaled_val_sweep.yaml)
- [configs/manifests/wti_feature_sets.yaml](/Users/jaeholee/Desktop/newoil/configs/manifests/wti_feature_sets.yaml)
- [configs/papers_raw.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_raw.yaml)
- [configs/papers_weekly.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_weekly.yaml)

Use order:

1. `smoke_wti_baseline.yaml`
2. `overnight_wti_baseline.yaml`
3. `overnight_wti_weight_decay.yaml`
4. `overnight_wti_diff.yaml`
5. `overnight_wti_diff_weight_decay.yaml`
6. `smoke_wti_short_window_log.yaml`
7. `overnight_wti_short_window_log.yaml`
8. `smoke_wti_myoil_h3.yaml`
9. `overnight_wti_myoil_h3.yaml`
10. `weekly_wti_h2_mse_report.yaml`
11. `weekly_wti_h2_mse_raw_report.yaml`
12. `weekly_wti_h2_mse_scaled_regularized.yaml`
13. `weekly_wti_h2_mse_scaled_val_sweep.yaml`

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
