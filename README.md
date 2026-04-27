# NeuralForecast Colab Experiment

[Open in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/notebooks/neuralforecast_model_comparison.ipynb)

This repository is kept intentionally small:

- [configs/experiment.yaml](/Users/jaeholee/Desktop/newoil/configs/experiment.yaml)
- [notebooks/neuralforecast_model_comparison.ipynb](/Users/jaeholee/Desktop/newoil/notebooks/neuralforecast_model_comparison.ipynb)

Structure:

- `configs/experiment.yaml`: dataset schema, frequency, horizon, validation split, test split
- `notebooks/neuralforecast_model_comparison.ipynb`: Colab notebook that reads the YAML and runs the experiment

What the notebook does:

- installs `neuralforecast` directly from the Nixtla GitHub repository
- loads your data from `csv/xlsx` upload or from a configured path
- trains `GRU`, `TimeXer`, and `iTransformer` with library defaults
- outputs train/validation loss curves
- outputs forecast plots
- outputs `MAE`, `RMSE`, `MAPE`, `sMAPE` tables

Why YAML here:

- keeps dataset-specific settings out of the notebook
- makes reruns reproducible
- keeps the repo clean without vendoring the full `neuralforecast` source tree

Notes:

- `TimeXer` and `iTransformer` are multivariate models, so the notebook aligns the panel by timestamp and drops timestamps with missing values across series before training
- if `data.file_path` is empty, the notebook asks you to upload the file in Colab
