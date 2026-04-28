from __future__ import annotations

import json
import math
import traceback
import inspect
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models import GRU, TimeXer, iTransformer


MODEL_REGISTRY = {
    "GRU": GRU,
    "TimeXer": TimeXer,
    "iTransformer": iTransformer,
}

LOSS_REGISTRY = {
    "mae": MAE,
    "mse": MSE,
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

LR_SCHEDULER_REGISTRY = {
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def patch_neuralforecast_reduce_on_plateau_monitor() -> None:
    if getattr(BaseModel.configure_optimizers, "_newoil_patched", False):
        return

    def configure_optimizers(self):
        if self.optimizer:
            optimizer_signature = inspect.signature(self.optimizer)
            optimizer_kwargs = deepcopy(self.optimizer_kwargs)
            if "lr" in optimizer_signature.parameters:
                if "lr" in optimizer_kwargs:
                    warnings.warn(
                        "ignoring learning rate passed in optimizer_kwargs, using the model's learning rate"
                    )
                optimizer_kwargs["lr"] = self.learning_rate
            optimizer = self.optimizer(params=self.parameters(), **optimizer_kwargs)
        else:
            if self.optimizer_kwargs:
                warnings.warn("ignoring optimizer_kwargs as the optimizer is not specified")
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler = {"frequency": 1, "interval": "step"}
        if self.lr_scheduler:
            lr_scheduler_signature = inspect.signature(self.lr_scheduler)
            lr_scheduler_kwargs = deepcopy(self.lr_scheduler_kwargs)
            if "optimizer" in lr_scheduler_signature.parameters and "optimizer" in lr_scheduler_kwargs:
                warnings.warn("ignoring optimizer passed in lr_scheduler_kwargs, using the model's optimizer")
                del lr_scheduler_kwargs["optimizer"]
            if "optimizer" in lr_scheduler_signature.parameters:
                lr_scheduler["scheduler"] = self.lr_scheduler(optimizer=optimizer, **lr_scheduler_kwargs)
            else:
                lr_scheduler["scheduler"] = self.lr_scheduler(**lr_scheduler_kwargs)

            if issubclass(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler["monitor"] = self.val_monitor
        else:
            if self.lr_scheduler_kwargs:
                warnings.warn("ignoring lr_scheduler_kwargs as the lr_scheduler is not specified")
            lr_scheduler["scheduler"] = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=self.lr_decay_steps,
                gamma=0.5,
            )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    configure_optimizers._newoil_patched = True
    BaseModel.configure_optimizers = configure_optimizers


patch_neuralforecast_reduce_on_plateau_monitor()


@dataclass
class BatchResult:
    batch_dir: Path
    summary_df: pd.DataFrame
    report_html: Path
    report_markdown: Path


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def normalize_transform_name(transform_name: Optional[str]) -> str:
    if transform_name is None:
        return "none"
    return str(transform_name).lower().replace("_", "-").strip()


def default_input_size(horizon: int, patch_len: int = 16) -> int:
    base = max(3 * horizon, patch_len)
    return int(math.ceil(base / patch_len) * patch_len)


def estimate_steps_per_epoch(
    train_length: int,
    input_size: int,
    horizon: int,
    n_series: int,
    training_cfg: Dict[str, Any],
) -> int:
    windows_per_series = max(train_length - input_size - horizon + 1, 1)
    total_windows = max(windows_per_series * max(n_series, 1), 1)

    windows_batch_size = training_cfg.get("windows_batch_size")
    if windows_batch_size is not None and int(windows_batch_size) > 0:
        effective_batch_size = int(windows_batch_size)
    else:
        effective_batch_size = int(training_cfg.get("batch_size") or total_windows)

    return max(int(math.ceil(total_windows / max(effective_batch_size, 1))), 1)


def resolve_epoch_compatible_training_cfg(
    training_cfg: Dict[str, Any],
    train_length: int,
    input_size: int,
    horizon: int,
    n_series: int,
) -> Dict[str, Any]:
    resolved = deepcopy(training_cfg)
    trainer_kwargs = deepcopy(resolved.get("trainer_kwargs") or {})
    trainer_max_epochs = trainer_kwargs.pop("max_epochs", None)
    resolved["trainer_kwargs"] = trainer_kwargs

    if resolved.get("max_epochs") is None and trainer_max_epochs is not None:
        resolved["max_epochs"] = trainer_max_epochs

    max_epochs = resolved.get("max_epochs")
    if max_epochs is None:
        return resolved

    steps_per_epoch = estimate_steps_per_epoch(
        train_length=train_length,
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        training_cfg=resolved,
    )
    epoch_equivalent_max_steps = int(max_epochs) * steps_per_epoch

    current_max_steps = resolved.get("max_steps")
    if current_max_steps is None:
        resolved["max_steps"] = epoch_equivalent_max_steps
    else:
        resolved["max_steps"] = min(int(current_max_steps), epoch_equivalent_max_steps)

    resolved["resolved_from_max_epochs"] = True
    resolved["estimated_steps_per_epoch"] = steps_per_epoch
    resolved["epoch_equivalent_max_steps"] = epoch_equivalent_max_steps
    resolved.pop("max_epochs", None)
    return resolved


def read_tabular_file(path: Path, sheet_name: int = 0) -> pd.DataFrame:
    lower_path = path.name.lower()
    if lower_path.endswith(".csv"):
        return pd.read_csv(path)
    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported file type: {path}")


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def select_columns(
    feature_manifest: Dict[str, Any], dataset_key: str, mode: str
) -> Dict[str, Any]:
    target_cfg = feature_manifest["target"]
    target_column = target_cfg["source_column"]
    if mode == "univariate":
        return {
            "target_column": target_column,
            "selected_columns": [target_column],
            "display_target_name": target_cfg.get("display_name", target_column),
        }

    feature_sets = feature_manifest["feature_sets"]
    feature_key = f"multivariate_{dataset_key}"
    feature_columns = list(feature_sets[feature_key])
    return {
        "target_column": target_column,
        "selected_columns": [target_column, *feature_columns],
        "display_target_name": target_cfg.get("display_name", target_column),
    }


def prepare_panel(
    repo_root: Path,
    dataset_cfg: Dict[str, Any],
    selected_columns: List[str],
    target_column: str,
    start_date: str,
    preprocess_cfg: Dict[str, Any],
) -> pd.DataFrame:
    data_cfg = dataset_cfg["data"]
    file_path = resolve_path(repo_root, data_cfg["file_path"])
    frame = read_tabular_file(file_path, sheet_name=data_cfg.get("sheet_name", 0))
    time_col = data_cfg["time_col"]
    freq = data_cfg["freq"]

    missing = [col for col in [time_col, *selected_columns] if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_path.name}: {missing}")

    panel = frame[[time_col, *selected_columns]].copy()
    panel = panel.rename(columns={time_col: "ds"})
    panel["ds"] = pd.to_datetime(panel["ds"])
    panel = panel.sort_values("ds").drop_duplicates(subset=["ds"]).set_index("ds")
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.loc[pd.Timestamp(start_date) :]

    if preprocess_cfg.get("reindex_full_range", False):
        full_index = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
        panel = panel.reindex(full_index)
        panel.index.name = "ds"

    fill_method = preprocess_cfg.get("fill_method")
    feature_columns = [column for column in panel.columns if column != target_column]
    if fill_method == "ffill":
        if feature_columns:
            panel[feature_columns] = panel[feature_columns].ffill()
    elif fill_method == "bfill":
        if feature_columns:
            panel[feature_columns] = panel[feature_columns].bfill()
    elif fill_method == "ffill_bfill":
        if feature_columns:
            panel[feature_columns] = panel[feature_columns].ffill().bfill()
    elif fill_method in (None, "none"):
        pass
    else:
        raise ValueError(f"Unsupported fill_method: {fill_method}")

    if preprocess_cfg.get("drop_before_first_complete", False):
        complete_mask = ~panel.isna().any(axis=1)
        if not complete_mask.any():
            raise ValueError("No complete timestamp available after preprocessing.")
        first_complete_ts = complete_mask[complete_mask].index[0]
        panel = panel.loc[first_complete_ts:]

    panel = panel.dropna(how="any")
    if panel.empty:
        raise ValueError("No aligned rows remain after preprocessing.")
    return panel


def transform_series(series: pd.Series, transform_name: str) -> pd.Series:
    transform_name = normalize_transform_name(transform_name)
    if transform_name == "none":
        return series.copy()
    if transform_name == "diff":
        return series.diff()
    if transform_name == "log-diff":
        non_positive = series <= 0
        if non_positive.any():
            min_value = float(series.min())
            raise ValueError(
                f"log-diff requires strictly positive values, but {series.name} has "
                f"{int(non_positive.sum())} non-positive rows (min={min_value})."
            )
        return np.log(series).diff()
    raise ValueError(f"Unsupported transform: {transform_name}")


def apply_transforms(
    panel: pd.DataFrame,
    target_column: str,
    target_transform: str,
    exog_transform: str,
) -> pd.DataFrame:
    transformed_columns = {}
    for column in panel.columns:
        transform_name = target_transform if column == target_column else exog_transform
        transformed_columns[column] = transform_series(panel[column], transform_name)

    transformed = pd.DataFrame(transformed_columns, index=panel.index).dropna()
    if transformed.empty:
        raise ValueError("Configured transforms removed all rows.")
    return transformed


def build_single_target_nf_df(
    panel: pd.DataFrame,
    target_column: str,
    display_target_name: str,
    hist_exog_columns: List[str],
) -> pd.DataFrame:
    ordered_columns = [target_column, *hist_exog_columns]
    nf_df = panel[ordered_columns].reset_index().rename(columns={"ds": "ds", target_column: "y"}).copy()
    nf_df["unique_id"] = display_target_name
    nf_df = nf_df[["unique_id", "ds", "y", *hist_exog_columns]]
    return nf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def assert_no_leakage_risk(
    preprocess_cfg: Dict[str, Any],
    panel: pd.DataFrame,
    freq: str,
    train_dates: pd.DatetimeIndex,
    val_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> None:
    fill_method = preprocess_cfg.get("fill_method")
    if fill_method in {"bfill", "ffill_bfill"}:
        raise ValueError(
            f"Potential leakage risk: fill_method={fill_method} uses backward filling, which is not allowed."
        )

    if panel.index.has_duplicates:
        raise ValueError("Potential leakage risk: duplicate timestamps remain after preprocessing.")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("Potential leakage risk: timestamps are not strictly increasing after preprocessing.")

    if len(train_dates) and len(val_dates) and train_dates.max() >= val_dates.min():
        raise ValueError("Potential leakage risk: validation starts before training ends.")
    if len(val_dates) and len(test_dates) and val_dates.max() >= test_dates.min():
        raise ValueError("Potential leakage risk: test starts before validation ends.")

    expected_index = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
    if len(expected_index) != len(panel.index) or not expected_index.equals(panel.index):
        raise ValueError(
            "Potential leakage or alignment risk: the post-processed panel is not contiguous on the configured frequency."
        )


def loss_name_to_instance(loss_name: Optional[str]):
    if loss_name is None:
        return None
    key = str(loss_name).lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss: {loss_name}")
    return LOSS_REGISTRY[key]()


def build_common_model_kwargs(
    model_name: str,
    horizon: int,
    input_size: int,
    n_series: int,
    hist_exog_columns: List[str],
    training_cfg: Dict[str, Any],
    random_seed: int,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "h": horizon,
        "input_size": input_size,
        "random_seed": random_seed,
    }

    loss_instance = loss_name_to_instance(training_cfg.get("loss"))
    if loss_instance is not None:
        kwargs["loss"] = loss_instance
        kwargs["valid_loss"] = loss_instance.__class__()  # fresh instance

    optional_ints = [
        "batch_size",
        "valid_batch_size",
        "max_steps",
        "val_check_steps",
        "early_stop_patience_steps",
        "step_size",
    ]
    for key in optional_ints:
        if training_cfg.get(key) is not None:
            kwargs[key] = int(training_cfg[key])

    if training_cfg.get("model_step_size") is not None:
        kwargs["step_size"] = int(training_cfg["model_step_size"])

    if training_cfg.get("val_monitor") is not None:
        kwargs["val_monitor"] = str(training_cfg["val_monitor"])

    for key in ["windows_batch_size", "inference_windows_batch_size"]:
        if key in training_cfg and training_cfg[key] is not None:
            kwargs[key] = training_cfg[key]

    for key in ["scaler_type", "learning_rate"]:
        if training_cfg.get(key) is not None:
            kwargs[key] = training_cfg[key]

    optimizer_cfg = training_cfg.get("optimizer") or {}
    optimizer_name = str(optimizer_cfg.get("name", "")).lower().strip()
    if optimizer_name:
        if optimizer_name not in OPTIMIZER_REGISTRY:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        kwargs["optimizer"] = OPTIMIZER_REGISTRY[optimizer_name]
        kwargs["optimizer_kwargs"] = optimizer_cfg.get("kwargs", {})

    lr_scheduler_cfg = training_cfg.get("lr_scheduler") or {}
    lr_scheduler_name = str(lr_scheduler_cfg.get("name", "")).lower().replace("_", "").strip()
    if lr_scheduler_name:
        if lr_scheduler_name not in LR_SCHEDULER_REGISTRY:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler_cfg.get('name')}")
        kwargs["lr_scheduler"] = LR_SCHEDULER_REGISTRY[lr_scheduler_name]
        kwargs["lr_scheduler_kwargs"] = {
            key: value
            for key, value in lr_scheduler_cfg.items()
            if key not in {"name", "max_lr"} and value is not None
        }

    trainer_kwargs = training_cfg.get("trainer_kwargs") or {}
    for key, value in trainer_kwargs.items():
        kwargs[key] = value

    model_kwargs = training_cfg.get("model_kwargs") or {}
    for key, value in model_kwargs.items():
        if value is not None:
            kwargs[key] = value

    if hist_exog_columns:
        kwargs["hist_exog_list"] = list(hist_exog_columns)

    if model_name in {"TimeXer", "iTransformer"}:
        kwargs["n_series"] = n_series
    return kwargs


def save_loss_history(model, run_dir: Path) -> pd.DataFrame:
    train_df = pd.DataFrame(model.train_trajectories, columns=["step", "train_loss"])
    valid_df = pd.DataFrame(model.valid_trajectories, columns=["step", "valid_loss"])
    history_df = train_df.merge(valid_df, on="step", how="outer").sort_values("step")
    final_epoch_index = getattr(model, "current_epoch", None)
    max_logged_step = pd.to_numeric(history_df["step"], errors="coerce").dropna().max()
    if final_epoch_index is not None and pd.notna(max_logged_step) and float(max_logged_step) > 0:
        completed_epochs = float(final_epoch_index) + 1.0
        history_df["epoch"] = history_df["step"].astype(float) * (completed_epochs / float(max_logged_step))
    history_df.to_csv(run_dir / "loss_history.csv", index=False)
    return history_df


def plot_loss_curves(
    history_df: pd.DataFrame,
    title: str,
    output_path: Path,
    y_scale: str = "linear",
    x_column: str = "step",
    x_label: str = "Global step",
) -> None:
    y_scale = str(y_scale or "linear").lower().strip()
    if y_scale not in {"linear", "log", "symlog"}:
        raise ValueError(f"Unsupported loss curve y-scale: {y_scale}")
    if x_column not in history_df.columns:
        x_column = "step"
        x_label = "Global step"

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False

    if "train_loss" in history_df:
        train_loss = history_df["train_loss"].where(history_df["train_loss"] > 0) if y_scale == "log" else history_df["train_loss"]
        if not train_loss.dropna().empty:
            ax.plot(history_df[x_column], train_loss, label="Train loss", linewidth=1.4)
            plotted = True

    if "valid_loss" in history_df:
        valid_rows = history_df.dropna(subset=["valid_loss"])
        if not valid_rows.empty:
            valid_loss = (
                valid_rows["valid_loss"].where(valid_rows["valid_loss"] > 0)
                if y_scale == "log"
                else valid_rows["valid_loss"]
            )
            if not valid_loss.dropna().empty:
                ax.plot(
                    valid_rows[x_column],
                    valid_loss,
                    label="Validation loss",
                    linewidth=1.8,
                    marker="o",
                )
                plotted = True

    if y_scale != "linear":
        ax.set_yscale(y_scale)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss")
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_run_issues(summary: Dict[str, Any]) -> Dict[str, str]:
    status = str(summary.get("status", "unknown"))
    mode = str(summary.get("mode", ""))
    train_drop = summary.get("train_drop_ratio")
    val_drop = summary.get("val_drop_ratio")
    rebound = summary.get("val_rebound_ratio")

    if status == "overfits":
        issue_note = "Train loss improves, but validation loss does not hold the gain and rebounds."
        next_action_note = "Keep the validation monitor, tighten early stopping, and reduce optimization aggressiveness."
    elif status == "stalls":
        issue_note = "Both train and validation losses improve slowly, suggesting limited learning progress."
        next_action_note = "Adjust horizon, input window, or optimizer schedule before adding more complexity."
    elif status == "learns":
        issue_note = "Validation loss improves without a material rebound, so the run is learning stably."
        next_action_note = "Use this run as the reference and compare uni vs multi with final forecast metrics."
    elif status == "invalid":
        issue_note = "Loss history is incomplete, so the learning pattern cannot be judged from this run alone."
        next_action_note = "Re-run with stable logging before comparing forecasting quality."
    else:
        issue_note = "Loss history needs manual review."
        next_action_note = "Check the saved loss history and forecast outputs before deciding the next change."

    if mode == "multivariate":
        issue_note += " This multivariate run uses a single target with historical exogenous features, so validation loss is target-aligned."

    if pd.notna(train_drop) and pd.notna(val_drop):
        issue_note += f" Train drop={float(train_drop):.3f}, validation drop={float(val_drop):.3f}."
    if pd.notna(rebound):
        issue_note += f" Validation rebound={float(rebound):.3f}."

    return {
        "issue_note": issue_note,
        "next_action_note": next_action_note,
    }


def invert_predictions(
    transform_name: str,
    target_predictions: np.ndarray,
    original_panel: pd.DataFrame,
    transformed_panel: pd.DataFrame,
    target_column: str,
    test_size: int,
) -> np.ndarray:
    transform_name = normalize_transform_name(transform_name)
    if transform_name == "none":
        return target_predictions
    if transform_name == "diff":
        train_end_date = transformed_panel.index[-test_size - 1]
        last_actual = float(original_panel[target_column].loc[train_end_date])
        return last_actual + np.cumsum(target_predictions)
    if transform_name == "log-diff":
        train_end_date = transformed_panel.index[-test_size - 1]
        last_actual = float(original_panel[target_column].loc[train_end_date])
        last_log_actual = math.log(last_actual)
        return np.exp(last_log_actual + np.cumsum(target_predictions))
    raise ValueError(f"Unsupported transform: {transform_name}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape_denom = np.where(smape_denom == 0, np.nan, smape_denom)
    smape = float(np.nanmean(2.0 * np.abs(y_pred - y_true) / smape_denom) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}


def diagnose_run(
    history_df: pd.DataFrame,
    improvement_ratio_threshold: float = 0.02,
    overfit_rebound_ratio: float = 0.05,
) -> Dict[str, Any]:
    valid_rows = history_df.dropna(subset=["valid_loss"])
    train_rows = history_df.dropna(subset=["train_loss"])
    if valid_rows.empty or train_rows.empty:
        return {"status": "invalid", "train_drop_ratio": np.nan, "val_drop_ratio": np.nan}

    train_start = float(train_rows["train_loss"].iloc[0])
    train_end = float(train_rows["train_loss"].iloc[-1])
    valid_start = float(valid_rows["valid_loss"].iloc[0])
    valid_end = float(valid_rows["valid_loss"].iloc[-1])
    valid_min = float(valid_rows["valid_loss"].min())

    train_drop_ratio = (train_start - train_end) / max(abs(train_start), 1e-8)
    val_drop_ratio = (valid_start - valid_min) / max(abs(valid_start), 1e-8)
    rebound_ratio = (valid_end - valid_min) / max(abs(valid_min), 1e-8)

    if val_drop_ratio >= improvement_ratio_threshold and rebound_ratio <= overfit_rebound_ratio:
        status = "learns"
    elif train_drop_ratio >= improvement_ratio_threshold and val_drop_ratio < improvement_ratio_threshold:
        status = "overfits"
    elif train_drop_ratio >= improvement_ratio_threshold and rebound_ratio > overfit_rebound_ratio:
        status = "overfits"
    else:
        status = "stalls"

    min_row = valid_rows.loc[valid_rows["valid_loss"].idxmin()]
    return {
        "status": status,
        "train_drop_ratio": train_drop_ratio,
        "val_drop_ratio": val_drop_ratio,
        "val_rebound_ratio": rebound_ratio,
        "min_val_step": int(min_row["step"]),
        "min_val_loss": valid_min,
        "final_train_loss": train_end,
        "final_val_loss": valid_end,
    }


def plot_forecast(
    history_target: pd.Series,
    actual_target: pd.Series,
    pred_target: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history_target.index, history_target.values, label="History", color="black", linewidth=1.4)
    ax.plot(actual_target.index, actual_target.values, label="Actual", color="#1f77b4", linewidth=2.0)
    ax.plot(
        pred_target.index,
        pred_target.values,
        label="Forecast",
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        marker="o",
    )
    ax.axvline(actual_target.index.min(), color="gray", linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _date_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _float_text(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _mode_label(mode: str) -> str:
    return "단변량" if mode == "univariate" else "다변량"


def _dataset_label(dataset: str) -> str:
    return "Daily" if dataset == "daily" else "Weekly"


def _markdown_file_link(label: str, path: Path) -> str:
    resolved = path.resolve()
    return f"[{label}](<{resolved}>)"


def _markdown_image(path: Path, alt: str) -> str:
    resolved = path.resolve()
    return f"![{alt}](<{resolved}>)"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) if not pd.isna(row[col]) else "" for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def select_best_run(summary_df: pd.DataFrame, dataset: str, mode: str) -> Optional[pd.Series]:
    sub = summary_df[
        (summary_df["dataset"] == dataset)
        & (summary_df["mode"] == mode)
        & (summary_df["status"] != "failed")
    ].copy()
    if sub.empty:
        return None
    sub = sub.sort_values(["RMSE", "MAE", "MAPE", "sMAPE"], na_position="last")
    return sub.iloc[0]


def save_uni_multi_comparison_plot(
    dataset: str,
    merged_df: pd.DataFrame,
    output_path: Path,
    uni_model: str,
    multi_model: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = pd.to_datetime(merged_df["ds"])
    ax.plot(x, merged_df["actual_target"], marker="o", linewidth=2.5, label="Actual")
    ax.plot(x, merged_df["uni_pred"], marker="o", linewidth=2.0, label=f"Uni ({uni_model})")
    ax.plot(x, merged_df["multi_pred"], marker="o", linewidth=2.0, label=f"Multi ({multi_model})")
    ax.set_title(f"{_dataset_label(dataset)} actual vs uni vs multi")
    ax.set_xlabel("Date")
    ax.set_ylabel("Target value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def prepare_batch_artifacts(summary_df: pd.DataFrame, batch_dir: Path) -> Dict[str, Any]:
    successful = summary_df[summary_df["status"] != "failed"].copy()
    artifacts: Dict[str, Any] = {
        "best_model_summary_df": pd.DataFrame(),
        "best_model_summary_csv": None,
        "comparison_by_dataset": {},
    }
    if successful.empty:
        return artifacts

    best_rows: List[Dict[str, Any]] = []
    for dataset in sorted(successful["dataset"].dropna().unique()):
        for mode in ["univariate", "multivariate"]:
            best = select_best_run(successful, dataset=dataset, mode=mode)
            if best is None:
                continue
            best_rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "best_model": best["model"],
                    "MAE": round(float(best["MAE"]), 6),
                    "RMSE": round(float(best["RMSE"]), 6),
                    "MAPE": round(float(best["MAPE"]), 6),
                    "sMAPE": round(float(best["sMAPE"]), 6),
                    "min_val_loss": round(float(best["min_val_loss"]), 6),
                    "final_val_loss": round(float(best["final_val_loss"]), 6),
                    "artifact_dir": best["artifact_dir"],
                    "scenario_name": best["scenario_name"],
                    "horizon": int(best["horizon"]),
                }
            )

        uni_best = select_best_run(successful, dataset=dataset, mode="univariate")
        multi_best = select_best_run(successful, dataset=dataset, mode="multivariate")
        if uni_best is None or multi_best is None:
            continue

        uni_pred_path = Path(str(uni_best["artifact_dir"])) / "predictions.csv"
        multi_pred_path = Path(str(multi_best["artifact_dir"])) / "predictions.csv"
        if not uni_pred_path.exists() or not multi_pred_path.exists():
            continue

        uni_pred = pd.read_csv(uni_pred_path)
        multi_pred = pd.read_csv(multi_pred_path)
        merged = (
            uni_pred[["ds", "actual_target", "predicted_target"]]
            .rename(columns={"predicted_target": "uni_pred"})
            .merge(
                multi_pred[["ds", "predicted_target"]].rename(columns={"predicted_target": "multi_pred"}),
                on="ds",
                how="inner",
            )
        )
        merged["uni_model"] = uni_best["model"]
        merged["multi_model"] = multi_best["model"]
        comparison_csv = batch_dir / f"{dataset}_uni_vs_multi_comparison.csv"
        comparison_png = batch_dir / f"{dataset}_actual_vs_uni_multi.png"
        merged.to_csv(comparison_csv, index=False)
        save_uni_multi_comparison_plot(
            dataset=dataset,
            merged_df=merged,
            output_path=comparison_png,
            uni_model=str(uni_best["model"]),
            multi_model=str(multi_best["model"]),
        )
        artifacts["comparison_by_dataset"][dataset] = {
            "csv": comparison_csv,
            "png": comparison_png,
            "uni_model": str(uni_best["model"]),
            "multi_model": str(multi_best["model"]),
            "horizon": int(uni_best["horizon"]),
        }

    best_df = pd.DataFrame(best_rows)
    if not best_df.empty:
        best_csv = batch_dir / "best_model_summary.csv"
        best_df.to_csv(best_csv, index=False)
        artifacts["best_model_summary_df"] = best_df
        artifacts["best_model_summary_csv"] = best_csv
    return artifacts


def build_experiment_methodology(batch_cfg: Dict[str, Any], summary_df: pd.DataFrame) -> str:
    parts = []
    description = str(batch_cfg.get("description", "")).strip()
    if description:
        parts.append(description)
    datasets = sorted(summary_df["dataset"].dropna().unique())
    modes = sorted(summary_df["mode"].dropna().unique())
    horizons = sorted({int(v) for v in summary_df["horizon"].dropna().tolist()}) if "horizon" in summary_df else []
    losses = sorted({str(v) for v in summary_df["loss_name"].dropna().tolist()}) if "loss_name" in summary_df else []
    scalers = sorted({str(v) for v in summary_df["scaler_type"].dropna().tolist()}) if "scaler_type" in summary_df else []
    if datasets:
        parts.append(f"dataset={','.join(datasets)}")
    if modes:
        parts.append(f"mode={','.join(modes)}")
    if horizons:
        parts.append(f"horizon={','.join(map(str, horizons))}")
    if losses:
        parts.append(f"loss={','.join(losses)}")
    if scalers:
        parts.append(f"scaler={','.join(scalers)}")
    return " / ".join(parts)


def build_main_result_lines(summary_df: pd.DataFrame, artifacts: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    best_df = artifacts.get("best_model_summary_df", pd.DataFrame())
    if best_df.empty:
        return ["성공적으로 완료된 run이 없어 주요 결과를 요약할 수 없습니다."]

    for dataset in sorted(best_df["dataset"].unique()):
        sub = best_df[best_df["dataset"] == dataset]
        uni = sub[sub["mode"] == "univariate"]
        multi = sub[sub["mode"] == "multivariate"]
        if not uni.empty and not multi.empty:
            uni_row = uni.iloc[0]
            multi_row = multi.iloc[0]
            if float(multi_row["RMSE"]) < float(uni_row["RMSE"]):
                diff = float(uni_row["RMSE"]) - float(multi_row["RMSE"])
                lines.append(
                    f"{_dataset_label(dataset)}에서는 다변량 {multi_row['best_model']}가 단변량 {uni_row['best_model']} 대비 "
                    f"RMSE {diff:.3f} 개선을 보였습니다."
                )
            else:
                diff = float(multi_row["RMSE"]) - float(uni_row["RMSE"])
                lines.append(
                    f"{_dataset_label(dataset)}에서는 단변량 {uni_row['best_model']}가 다변량 {multi_row['best_model']} 대비 "
                    f"RMSE {diff:.3f} 우위를 보였습니다."
                )
        else:
            row = sub.iloc[0]
            lines.append(
                f"{_dataset_label(dataset)}에서는 {_mode_label(row['mode'])} {row['best_model']}가 "
                f"RMSE {float(row['RMSE']):.3f}, MAE {float(row['MAE']):.3f}를 기록했습니다."
            )
    return lines


def build_insight_lines(summary_df: pd.DataFrame, artifacts: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    best_df = artifacts.get("best_model_summary_df", pd.DataFrame())
    if best_df.empty:
        return lines

    if {"dataset", "mode", "min_val_loss"}.issubset(best_df.columns):
        for dataset in sorted(best_df["dataset"].unique()):
            uni = best_df[(best_df["dataset"] == dataset) & (best_df["mode"] == "univariate")]
            multi = best_df[(best_df["dataset"] == dataset) & (best_df["mode"] == "multivariate")]
            if not uni.empty and not multi.empty:
                uni_val = float(uni.iloc[0]["min_val_loss"])
                multi_val = float(multi.iloc[0]["min_val_loss"])
                if uni_val > 0 and multi_val / uni_val >= 100:
                    lines.append(
                        f"{_dataset_label(dataset)} multivariate loss 절대치는 단변량 대비 매우 크게 나타나므로, "
                        f"raw loss 숫자보다 target metric과 실제 예측값 기준 해석이 더 적절합니다."
                    )
                    break

    rebound_rows = summary_df[
        (summary_df["status"] == "overfits")
        | ((summary_df["final_val_loss"] > summary_df["min_val_loss"] * 1.1) & summary_df["min_val_loss"].notna())
    ]
    if not rebound_rows.empty:
        lines.append("일부 run에서 validation loss가 최저점 이후 다시 반등해, 과적합 또는 validation split 민감도가 의심됩니다.")
    return lines


def build_conclusion_sections(summary_df: pd.DataFrame, artifacts: Dict[str, Any]) -> Dict[str, str]:
    best_df = artifacts.get("best_model_summary_df", pd.DataFrame())
    successful = summary_df[summary_df["status"] != "failed"]
    current_status = "이번 배치는 요청된 실험 조건 기준으로 성공한 run들을 중심으로 비교 가능한 결과를 생성했습니다."
    if successful.empty:
        current_status = "이번 배치는 실패한 run만 있어 현재 상태를 비교할 수 없습니다."

    meaning = "현재 결과는 baseline 또는 개선 run의 상대 비교 기준으로 해석하는 것이 적절합니다."
    if not best_df.empty and {"dataset", "mode", "RMSE"}.issubset(best_df.columns):
        datasets = sorted(best_df["dataset"].unique())
        if datasets:
            meaning = (
                f"{', '.join(_dataset_label(d) for d in datasets)} 기준으로 "
                "loss curve와 target metric을 함께 보며 단변량/다변량 및 모델별 차이를 해석할 수 있습니다."
            )

    next_actions = []
    for note in successful.get("next_action_note", pd.Series(dtype=object)).dropna().astype(str):
        if note not in next_actions:
            next_actions.append(note)
    next_action = next_actions[0] if next_actions else "다음 실험에서는 개선 대상 이슈를 가장 직접적으로 건드리는 설정 변경을 추가 비교합니다."
    return {
        "current_status": current_status,
        "meaning": meaning,
        "next_action": next_action,
    }


def build_html_report(batch_cfg: Dict[str, Any], summary_df: pd.DataFrame, batch_dir: Path) -> str:
    rows_html = summary_df.to_html(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
    sections = [
        "<html><head><meta charset='utf-8'><title>Overnight WTI Report</title></head><body>",
        f"<h1>{batch_cfg['name']}</h1>",
        f"<p>Generated at {datetime.utcnow().isoformat()}Z</p>",
        "<h2>Run Summary</h2>",
        rows_html,
    ]
    for _, row in summary_df.iterrows():
        run_name = row["run_name"]
        sections.append(f"<h3>{run_name}</h3>")
        if row["status"] == "failed":
            sections.append(f"<p><strong>Failure:</strong> {row.get('error_message', '')}</p>")
            continue
        sections.append("<ul>")
        for key in ["status", "min_val_loss", "final_val_loss", "MAE", "RMSE", "MAPE", "sMAPE"]:
            if key in row:
                sections.append(f"<li>{key}: {row[key]}</li>")
        if "issue_note" in row and pd.notna(row["issue_note"]):
            sections.append(f"<li><strong>Issue</strong>: {row['issue_note']}</li>")
        if "next_action_note" in row and pd.notna(row["next_action_note"]):
            sections.append(f"<li><strong>Next action</strong>: {row['next_action_note']}</li>")
        sections.append("</ul>")
        loss_curve = Path(str(row["artifact_dir"])) / "loss_curve.png"
        forecast_plot = Path(str(row["artifact_dir"])) / "forecast_plot.png"
        if loss_curve.exists():
            rel_loss_curve = loss_curve.relative_to(batch_dir) if batch_dir in loss_curve.parents else loss_curve
            sections.append(f"<img src='{rel_loss_curve.as_posix()}' width='900'>")
        if forecast_plot.exists():
            rel_forecast_plot = (
                forecast_plot.relative_to(batch_dir) if batch_dir in forecast_plot.parents else forecast_plot
            )
            sections.append(f"<img src='{rel_forecast_plot.as_posix()}' width='900'>")
    sections.append("</body></html>")
    return "\n".join(sections)


def build_markdown_report(
    batch_cfg: Dict[str, Any],
    summary_df: pd.DataFrame,
    batch_dir: Path,
    artifacts: Dict[str, Any],
) -> str:
    successful = summary_df[summary_df["status"] != "failed"].copy()
    target_columns = sorted(successful["target_column"].dropna().unique()) if "target_column" in successful else []
    models = [str(model) for model in batch_cfg.get("models", [])]
    methodology = build_experiment_methodology(batch_cfg, successful if not successful.empty else summary_df)
    main_results = build_main_result_lines(successful if not successful.empty else summary_df, artifacts)
    insight_lines = build_insight_lines(successful if not successful.empty else summary_df, artifacts)
    conclusion = build_conclusion_sections(successful if not successful.empty else summary_df, artifacts)

    lines = [
        f"# {batch_cfg['name']}",
        "",
        f"- Generated at: {datetime.utcnow().isoformat()}Z",
        f"- Start date filter: {batch_cfg['start_date']}",
        "",
        f"- 예측 타깃: {', '.join(target_columns) if target_columns else ''}",
        f"- 모델 종류: {', '.join(models)}",
        f"- 실험 방법론({batch_cfg['name']}): {methodology}",
    ]

    if not successful.empty:
        for dataset in sorted(successful["dataset"].dropna().unique()):
            row = successful[successful["dataset"] == dataset].iloc[0]
            lines.append(
                "- 실제 날짜 세팅: "
                f"{_dataset_label(dataset)} / "
                f"Train {_date_text(row.get('train_start_date'))} ~ {_date_text(row.get('train_end_date'))}, "
                f"Val {_date_text(row.get('val_start_date'))} ~ {_date_text(row.get('val_end_date'))}, "
                f"Test {_date_text(row.get('test_start_date'))} ~ {_date_text(row.get('test_end_date'))}"
            )

    for result_line in main_results:
        lines.append(f"- 주요 실험 결과: {result_line}")
    for insight_line in insight_lines:
        lines.append(f"- 내가 생각한 인사이트: {insight_line}")

    for dataset in ["weekly", "daily"]:
        dataset_rows = successful[successful["dataset"] == dataset]
        if dataset_rows.empty:
            continue
        lines.extend(["", f"## {_dataset_label(dataset)} Loss Curve", ""])
        for mode in ["univariate", "multivariate"]:
            mode_rows = dataset_rows[dataset_rows["mode"] == mode].copy()
            if mode_rows.empty:
                continue
            mode_rows = mode_rows.sort_values("model")
            lines.extend(["", f"### {_mode_label(mode)} 모델별 그래프", ""])
            for _, row in mode_rows.iterrows():
                loss_curve_path = Path(str(row["artifact_dir"])) / "loss_curve.png"
                if loss_curve_path.exists():
                    alt_text = f"{row['run_name']} loss curve"
                    lines.append(f"- {row['model']}: {_markdown_image(loss_curve_path, alt_text)}")

    for dataset in ["daily", "weekly"]:
        comparison = artifacts.get("comparison_by_dataset", {}).get(dataset)
        if comparison is None:
            continue
        horizon = comparison.get("horizon", "")
        lines.extend(
            [
                "",
                f"## {_dataset_label(dataset)} 실제값 vs 예측값 (horizon {horizon})",
                "",
                f"- actual vs uni vs multi: {_markdown_image(comparison['png'], f'{dataset} actual vs uni vs multi')}",
            ]
        )

    lines.extend(["", "## 지표 요약", ""])
    best_summary_df = artifacts.get("best_model_summary_df", pd.DataFrame()).copy()
    if not best_summary_df.empty:
        metrics_cols = [
            "dataset",
            "mode",
            "best_model",
            "MAE",
            "RMSE",
            "MAPE",
            "sMAPE",
            "min_val_loss",
            "final_val_loss",
        ]
        metrics_df = best_summary_df[metrics_cols].copy()
        for col in ["MAE", "RMSE", "MAPE", "sMAPE", "min_val_loss", "final_val_loss"]:
            metrics_df[col] = metrics_df[col].map(lambda x: _float_text(x, digits=6))
        lines.append(dataframe_to_markdown_table(metrics_df))
        if artifacts.get("best_model_summary_csv"):
            lines.extend(["", f"- CSV: {_markdown_file_link('best_model_summary.csv', artifacts['best_model_summary_csv'])}"])
    else:
        lines.append("요약 가능한 성공 run이 없습니다.")

    lines.extend(
        [
            "",
            "## 결론 및 향후 계획",
            "",
            "### 현재 상태",
            f"- {conclusion['current_status']}",
            "",
            "### 의미",
            f"- {conclusion['meaning']}",
            "",
            "### 다음 액션",
            f"- {conclusion['next_action']}",
        ]
    )
    return "\n".join(lines)


def run_single_experiment(
    repo_root: Path,
    batch_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    feature_manifest: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    model_name: str,
    run_dir: Path,
) -> Dict[str, Any]:
    selection = select_columns(feature_manifest, scenario_cfg["dataset"], scenario_cfg["mode"])
    target_column = selection["target_column"]
    selected_columns = selection["selected_columns"]
    display_target_name = selection["display_target_name"]
    hist_exog_columns = [column for column in selected_columns if column != target_column]

    preprocess_cfg = dict(dataset_cfg.get("preprocess", {}))
    preprocess_cfg.update(scenario_cfg.get("preprocess", {}))

    original_panel = prepare_panel(
        repo_root=repo_root,
        dataset_cfg=dataset_cfg,
        selected_columns=selected_columns,
        target_column=target_column,
        start_date=batch_cfg["start_date"],
        preprocess_cfg=preprocess_cfg,
    )

    scenario_cfg = dict(scenario_cfg)
    runtime_cfg = dict(batch_cfg.get("runtime", {}))
    horizon = int(scenario_cfg["horizon"])
    val_size = int(scenario_cfg["val_size"])
    test_size = int(scenario_cfg["test_size"])
    patch_len = int(scenario_cfg.get("patch_len", 16))
    training_cfg = deepcopy(batch_cfg.get("training_defaults", {}))
    deep_update(training_cfg, scenario_cfg.get("training", {}))
    deep_update(training_cfg, batch_cfg.get("model_training_overrides", {}).get(model_name, {}))
    deep_update(training_cfg, scenario_cfg.get("model_training_overrides", {}).get(model_name, {}))
    report_cfg = dict(batch_cfg.get("report", {}))
    report_cfg.update(scenario_cfg.get("report", {}))
    random_seed = int(runtime_cfg.get("random_seed", 1))

    legacy_transform = normalize_transform_name(scenario_cfg.get("transform"))
    target_transform = normalize_transform_name(
        scenario_cfg.get("target_transform", runtime_cfg.get("transformations_target", legacy_transform))
    )
    exog_transform = normalize_transform_name(
        scenario_cfg.get("exog_transform", runtime_cfg.get("transformations_exog", legacy_transform))
    )
    transformed_panel = apply_transforms(
        original_panel,
        target_column=target_column,
        target_transform=target_transform,
        exog_transform=exog_transform,
    )
    transform_name = (
        target_transform
        if target_transform == exog_transform
        else f"target:{target_transform}|exog:{exog_transform}"
    )

    input_size = int(training_cfg.get("input_size", default_input_size(horizon, patch_len)))
    lookback_plot = int(scenario_cfg.get("lookback_plot", report_cfg.get("lookback_plot", 120)))
    loss_scale = str(report_cfg.get("loss_scale", "linear")).lower().strip()
    x_axis_mode = str(report_cfg.get("x_axis", "step")).lower().strip()

    if test_size != horizon:
        raise ValueError(
            f"test_size must match horizon for direct predict() evaluation. "
            f"Received horizon={horizon}, test_size={test_size}"
        )

    if len(transformed_panel) <= val_size + test_size + input_size:
        raise ValueError(
            f"Not enough rows after preprocessing for {scenario_cfg['name']} / {model_name}. "
            f"rows={len(transformed_panel)}, input_size={input_size}, val_size={val_size}, test_size={test_size}"
        )

    train_dates = transformed_panel.index[: -(val_size + test_size)]
    val_dates = transformed_panel.index[-(val_size + test_size) : -test_size]
    test_dates = transformed_panel.index[-test_size:]
    train_val_panel = transformed_panel.iloc[:-test_size]
    assert_no_leakage_risk(
        preprocess_cfg=preprocess_cfg,
        panel=transformed_panel,
        freq=dataset_cfg["data"]["freq"],
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
    )

    train_val_df = build_single_target_nf_df(
        panel=train_val_panel,
        target_column=target_column,
        display_target_name=display_target_name,
        hist_exog_columns=hist_exog_columns,
    )
    n_series = 1
    training_cfg = resolve_epoch_compatible_training_cfg(
        training_cfg=training_cfg,
        train_length=len(train_val_panel),
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
    )

    kwargs = build_common_model_kwargs(
        model_name=model_name,
        horizon=horizon,
        input_size=input_size,
        n_series=n_series,
        hist_exog_columns=hist_exog_columns,
        training_cfg=training_cfg,
        random_seed=random_seed,
    )
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**kwargs)
    nf = NeuralForecast(models=[model], freq=dataset_cfg["data"]["freq"])
    nf.fit(df=train_val_df, val_size=val_size)
    preds_df = nf.predict().sort_values(["unique_id", "ds"]).reset_index(drop=True)

    target_pred_rows = preds_df[preds_df["unique_id"] == target_column].copy()
    target_pred_rows["ds"] = pd.to_datetime(target_pred_rows["ds"])
    target_predictions = target_pred_rows[model_name].to_numpy()
    actual_target = original_panel[target_column].loc[test_dates]
    predicted_target = invert_predictions(
        transform_name=target_transform,
        target_predictions=target_predictions,
        original_panel=original_panel,
        transformed_panel=transformed_panel,
        target_column=target_column,
        test_size=test_size,
    )
    pred_target_series = pd.Series(predicted_target, index=test_dates, name="prediction")
    history_target = original_panel[target_column].iloc[-(test_size + lookback_plot) : -test_size]

    metrics = compute_metrics(actual_target.to_numpy(), pred_target_series.to_numpy())
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(run_dir / "metrics.csv", index=False)

    predictions_out = pd.DataFrame(
        {
            "ds": test_dates,
            "actual_target": actual_target.to_numpy(),
            "predicted_target": pred_target_series.to_numpy(),
            "model": model_name,
            "scenario": scenario_cfg["name"],
            "target_column": target_column,
        }
    )
    predictions_out.to_csv(run_dir / "predictions.csv", index=False)

    fitted_model = nf.models[0]
    history_df = save_loss_history(fitted_model, run_dir)
    diagnostics = diagnose_run(history_df)
    x_column = "epoch" if x_axis_mode == "epoch" and "epoch" in history_df.columns else "step"
    x_label = "Epoch" if x_column == "epoch" else "Global step"
    plot_loss_curves(
        history_df,
        f"{scenario_cfg['name']} - {model_name} loss curves",
        run_dir / "loss_curve.png",
        y_scale=loss_scale,
        x_column=x_column,
        x_label=x_label,
    )
    plot_forecast(
        history_target=history_target,
        actual_target=actual_target,
        pred_target=pred_target_series,
        title=f"{scenario_cfg['name']} - {model_name} forecast ({display_target_name})",
        output_path=run_dir / "forecast_plot.png",
    )

    snapshot = {
        "batch_name": batch_cfg["name"],
        "scenario": scenario_cfg,
        "model_name": model_name,
        "dataset_name": dataset_cfg.get("name"),
        "target_column": target_column,
        "selected_columns": selected_columns,
        "hist_exog_columns": hist_exog_columns,
        "transform": transform_name,
        "target_transform": target_transform,
        "exog_transform": exog_transform,
        "input_size": input_size,
        "loss_scale": loss_scale,
        "rows_after_preprocessing": int(len(original_panel)),
        "rows_after_transform": int(len(transformed_panel)),
        "leakage_checks_passed": True,
        "resolved_training_cfg": training_cfg,
    }
    dump_yaml(run_dir / "config_snapshot.yaml", snapshot)

    summary = {
        "run_name": f"{scenario_cfg['name']}__{model_name}",
        "scenario_name": scenario_cfg["name"],
        "dataset": scenario_cfg["dataset"],
        "mode": scenario_cfg["mode"],
        "model": model_name,
        "transform": transform_name,
        "target_transform": target_transform,
        "exog_transform": exog_transform,
        "target_column": target_column,
        "n_series": n_series,
        "n_hist_exog": len(hist_exog_columns),
        "rows_after_preprocessing": int(len(original_panel)),
        "rows_after_transform": int(len(transformed_panel)),
        "horizon": horizon,
        "val_size": val_size,
        "test_size": test_size,
        "input_size": input_size,
        "loss_scale": loss_scale,
        "x_axis": x_label,
        "loss_name": str(training_cfg.get("loss", "default")).lower(),
        "scaler_type": training_cfg.get("scaler_type", "default"),
        "train_start_date": train_dates.min().strftime("%Y-%m-%d") if len(train_dates) else "",
        "train_end_date": train_dates.max().strftime("%Y-%m-%d") if len(train_dates) else "",
        "val_start_date": val_dates.min().strftime("%Y-%m-%d") if len(val_dates) else "",
        "val_end_date": val_dates.max().strftime("%Y-%m-%d") if len(val_dates) else "",
        "test_start_date": test_dates.min().strftime("%Y-%m-%d") if len(test_dates) else "",
        "test_end_date": test_dates.max().strftime("%Y-%m-%d") if len(test_dates) else "",
        "completed_epochs": float(history_df["epoch"].max()) if "epoch" in history_df.columns else np.nan,
        "artifact_dir": str(run_dir),
        **diagnostics,
        **metrics,
    }
    summary.update(summarize_run_issues(summary))
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def run_batch_from_config(
    batch_config_path: Path,
    repo_root: Path,
    output_root: Optional[Path] = None,
) -> BatchResult:
    batch_cfg = load_yaml(batch_config_path)
    batch_name = batch_cfg["name"]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_output_root = output_root if output_root is not None else repo_root / "outputs"
    batch_dir = ensure_dir(base_output_root / f"{batch_name}_{timestamp}")

    feature_manifest_path = resolve_path(repo_root, batch_cfg["feature_manifest_path"])
    feature_manifest = load_yaml(feature_manifest_path)
    dataset_cfgs = {
        key: load_yaml(resolve_path(repo_root, value))
        for key, value in batch_cfg["dataset_refs"].items()
    }

    summary_rows: List[Dict[str, Any]] = []
    for scenario_cfg in batch_cfg["scenarios"]:
        for model_name in batch_cfg["models"]:
            run_name = f"{scenario_cfg['name']}__{model_name}"
            run_dir = ensure_dir(batch_dir / run_name)
            print(f"[RUN] {run_name}")
            try:
                summary = run_single_experiment(
                    repo_root=repo_root,
                    batch_cfg=batch_cfg,
                    dataset_cfg=dataset_cfgs[scenario_cfg["dataset"]],
                    feature_manifest=feature_manifest,
                    scenario_cfg=scenario_cfg,
                    model_name=model_name,
                    run_dir=run_dir,
                )
            except Exception as exc:  # noqa: BLE001
                error_message = str(exc)
                traceback_text = traceback.format_exc()
                with (run_dir / "error.txt").open("w", encoding="utf-8") as f:
                    f.write(traceback_text)
                summary = {
                    "run_name": run_name,
                    "scenario_name": scenario_cfg["name"],
                    "dataset": scenario_cfg["dataset"],
                    "mode": scenario_cfg["mode"],
                    "model": model_name,
                    "transform": scenario_cfg.get("transform", "none"),
                    "target_transform": scenario_cfg.get(
                        "target_transform",
                        batch_cfg.get("runtime", {}).get("transformations_target", scenario_cfg.get("transform", "none")),
                    ),
                    "exog_transform": scenario_cfg.get(
                        "exog_transform",
                        batch_cfg.get("runtime", {}).get("transformations_exog", scenario_cfg.get("transform", "none")),
                    ),
                    "target_column": feature_manifest["target"]["source_column"],
                    "artifact_dir": str(run_dir),
                    "status": "failed",
                    "error_message": error_message,
                }
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(batch_dir / "summary.csv", index=False)
    artifacts = prepare_batch_artifacts(summary_df, batch_dir)

    report_html = batch_dir / "report.html"
    report_markdown = batch_dir / "report.md"
    report_html.write_text(build_html_report(batch_cfg, summary_df, batch_dir), encoding="utf-8")
    report_markdown.write_text(build_markdown_report(batch_cfg, summary_df, batch_dir, artifacts), encoding="utf-8")
    dump_yaml(batch_dir / "batch_config_snapshot.yaml", batch_cfg)
    return BatchResult(
        batch_dir=batch_dir,
        summary_df=summary_df,
        report_html=report_html,
        report_markdown=report_markdown,
    )
