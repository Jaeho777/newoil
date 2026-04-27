# NeuralForecast Colab Experiment

[Open in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/notebooks/neuralforecast_model_comparison.ipynb)

This repository is kept intentionally small:

- [configs/experiment.yaml](/Users/jaeholee/Desktop/newoil/configs/experiment.yaml)
- [configs/papers_weekly.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_weekly.yaml)
- [configs/papers_raw.yaml](/Users/jaeholee/Desktop/newoil/configs/papers_raw.yaml)
- [configs/tuning_overrides.example.yaml](/Users/jaeholee/Desktop/newoil/configs/tuning_overrides.example.yaml)
- [notebooks/neuralforecast_model_comparison.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/neuralforecast_model_comparison.ipynb)

Structure:

- `configs/experiment.yaml`: dataset schema, frequency, horizon, validation split, test split
- `configs/papers_weekly.yaml`: ready-to-run config for `data/papers_db_weekly.csv`
- `configs/papers_raw.yaml`: ready-to-run config for `data/papers_db_raw.csv`
- `configs/tuning_overrides.example.yaml`: optional reference for later tuning overrides
- `notebooks/neuralforecast_model_comparison.ipynb`: Colab notebook that reads the YAML and runs the experiment

What the notebook does:

- installs `neuralforecast` directly from the Nixtla GitHub repository
- loads your data from `csv/xlsx` upload or from a configured repo path such as `data/my_dataset.csv`
- trains `GRU`, `TimeXer`, and `iTransformer` with library defaults unless optional overrides are added
- outputs train/validation loss curves
- outputs forecast plots
- outputs `MAE`, `RMSE`, `MAPE`, `sMAPE` tables

Why YAML here:

- keeps dataset-specific settings out of the notebook
- makes reruns reproducible
- keeps the repo clean without vendoring the full `neuralforecast` source tree

Recommended workflow:

1. Open the notebook and choose `CONFIG_RELATIVE_PATH`.
2. Start with `configs/papers_weekly.yaml` or `configs/papers_raw.yaml`.
3. Check train / validation loss and forecast quality.
4. Only if validation loss stalls or overfits, copy selective fields from [configs/tuning_overrides.example.yaml](/Users/jaeholee/Desktop/newoil/configs/tuning_overrides.example.yaml) into the chosen config.

Notes:

- `TimeXer` and `iTransformer` are multivariate models, so the notebook aligns the panel by timestamp before training
- the included `papers_weekly` and `papers_raw` configs use `forward fill` and then trim to the first timestamp where all 228 series are available
- if `data.file_path` is empty, the notebook asks you to upload the file in Colab
- if you commit a dataset into this repository, place it under `data/` and set `data.file_path` in [configs/experiment.yaml](/Users/jaeholee/Desktop/newoil/configs/experiment.yaml), for example `data/my_dataset.csv`
- if the dataset is large or private, do not commit it; leave `data.file_path` empty and upload it in Colab instead
- optional `runtime:` and `training:` sections can be added to `experiment.yaml` when you want to override the baseline
- the notebook currently supports safe shared overrides such as `random_seed`, `input_size`, `batch_size`, `max_steps`, `loss`, and `optimizer`
- older pipeline features such as Optuna studies or external multi-GPU schedulers are intentionally not wired into the baseline notebook yet
