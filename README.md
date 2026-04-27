# NeuralForecast Colab Notebook

[Open in Colab](https://colab.research.google.com/github/Jaeho777/newoil/blob/main/neuralforecast_gru_timexer_itransformer_colab.ipynb)

This repository contains a Colab-ready notebook that compares `GRU`, `TimeXer`, and `iTransformer` from `neuralforecast`.

The notebook is designed for user-provided data and produces:

- train / validation loss curves
- forecast plots on the test horizon
- metrics tables with `MAE`, `RMSE`, `MAPE`, and `sMAPE`

Supported data layouts:

- long format: one row per timestamp and series, for example `unique_id`, `ds`, `y`
- wide format: one timestamp column and one value column per series

Notes:

- model-specific hyperparameters are left at the library defaults
- only required experiment settings are provided: forecast horizon, computed input window, number of series, validation split, and test split
- the notebook aligns the panel by timestamp and drops rows with missing values across series before training
