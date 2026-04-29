from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback
import inspect
import types
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


class _RayTunePlaceholder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> "_RayTunePlaceholder":
        return _RayTunePlaceholder(*args, **kwargs)

    def __getattr__(self, name: str) -> "_RayTunePlaceholder":
        return _RayTunePlaceholder(name)


class _UnavailableRayClass:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "ray[tune] is required only for NeuralForecast Auto models. "
            "This pipeline uses fixed GRU/TimeXer/iTransformer models, so install "
            "ray[tune] only if you explicitly run Auto models."
        )


def patch_missing_ray_for_neuralforecast_import() -> None:
    if "ray" in sys.modules:
        return
    try:
        ray_spec = importlib.util.find_spec("ray")
    except ValueError:
        ray_spec = None
    if ray_spec is not None:
        return

    ray_module = types.ModuleType("ray")
    air_module = types.ModuleType("ray.air")
    tune_module = types.ModuleType("ray.tune")
    tune_integration_module = types.ModuleType("ray.tune.integration")
    tune_pl_module = types.ModuleType("ray.tune.integration.pytorch_lightning")
    tune_search_module = types.ModuleType("ray.tune.search")
    basic_variant_module = types.ModuleType("ray.tune.search.basic_variant")
    for module in (ray_module, tune_module, tune_integration_module, tune_search_module):
        module.__path__ = []  # type: ignore[attr-defined]

    def placeholder_factory(*args: Any, **kwargs: Any) -> _RayTunePlaceholder:
        return _RayTunePlaceholder(*args, **kwargs)

    def module_getattr(name: str) -> _RayTunePlaceholder:
        return _RayTunePlaceholder(name)

    for module in (ray_module, air_module, tune_module):
        module.__getattr__ = module_getattr  # type: ignore[attr-defined]

    for attr in (
        "choice",
        "grid_search",
        "lograndint",
        "loguniform",
        "qlograndint",
        "qloguniform",
        "qrandint",
        "quniform",
        "randint",
        "sample_from",
        "uniform",
    ):
        setattr(tune_module, attr, placeholder_factory)

    air_module.RunConfig = _RayTunePlaceholder
    tune_module.TuneConfig = _RayTunePlaceholder
    tune_pl_module.TuneReportCallback = _UnavailableRayClass
    basic_variant_module.BasicVariantGenerator = _UnavailableRayClass

    ray_module.air = air_module
    ray_module.tune = tune_module
    tune_module.integration = tune_integration_module
    tune_module.search = tune_search_module
    tune_integration_module.pytorch_lightning = tune_pl_module
    tune_search_module.basic_variant = basic_variant_module

    sys.modules.setdefault("ray", ray_module)
    sys.modules.setdefault("ray.air", air_module)
    sys.modules.setdefault("ray.tune", tune_module)
    sys.modules.setdefault("ray.tune.integration", tune_integration_module)
    sys.modules.setdefault("ray.tune.integration.pytorch_lightning", tune_pl_module)
    sys.modules.setdefault("ray.tune.search", tune_search_module)
    sys.modules.setdefault("ray.tune.search.basic_variant", basic_variant_module)

NeuralForecast = None
BaseModel = None
MODEL_REGISTRY = {"GRU": None, "TimeXer": None, "iTransformer": None}
LOSS_REGISTRY = {"mae": None, "mse": None}
_NEURALFORECAST_RUNTIME_READY = False

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

LR_SCHEDULER_REGISTRY = {
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def build_loss_csv_logger(run_dir: Path):
    try:
        from lightning.pytorch.loggers import CSVLogger
    except Exception:
        try:
            from pytorch_lightning.loggers import CSVLogger
        except Exception:
            return None
    return CSVLogger(save_dir=str(run_dir), name="loss_logs")


def patch_neuralforecast_reduce_on_plateau_monitor() -> None:
    if BaseModel is None:
        raise RuntimeError("NeuralForecast runtime has not been loaded yet.")
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


def ensure_neuralforecast_runtime() -> None:
    global BaseModel
    global LOSS_REGISTRY
    global MODEL_REGISTRY
    global NeuralForecast
    global _NEURALFORECAST_RUNTIME_READY

    if _NEURALFORECAST_RUNTIME_READY:
        return

    patch_missing_ray_for_neuralforecast_import()

    from neuralforecast import NeuralForecast as NeuralForecastCls
    from neuralforecast.common._base_model import BaseModel as BaseModelCls
    from neuralforecast.losses.pytorch import MAE, MSE
    from neuralforecast.models import GRU, TimeXer, iTransformer

    NeuralForecast = NeuralForecastCls
    BaseModel = BaseModelCls
    MODEL_REGISTRY = {
        "GRU": GRU,
        "TimeXer": TimeXer,
        "iTransformer": iTransformer,
    }
    LOSS_REGISTRY = {
        "mae": MAE,
        "mse": MSE,
    }
    patch_neuralforecast_reduce_on_plateau_monitor()
    _NEURALFORECAST_RUNTIME_READY = True


@dataclass
class BatchResult:
    batch_dir: Path
    summary_df: pd.DataFrame
    report_html: Path
    report_markdown: Path
    batch_config: Dict[str, Any]


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

    requested_max_epochs = int(max_epochs)
    steps_per_epoch = estimate_steps_per_epoch(
        train_length=train_length,
        input_size=input_size,
        horizon=horizon,
        n_series=n_series,
        training_cfg=resolved,
    )
    epoch_equivalent_max_steps = requested_max_epochs * steps_per_epoch

    current_max_steps = resolved.get("max_steps")
    if current_max_steps is None:
        resolved["max_steps"] = epoch_equivalent_max_steps
    else:
        resolved["max_steps"] = min(int(current_max_steps), epoch_equivalent_max_steps)

    resolved["resolved_from_max_epochs"] = True
    resolved["requested_max_epochs"] = requested_max_epochs
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
    ensure_neuralforecast_runtime()
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
    kwargs.setdefault("logger", False)

    model_kwargs = training_cfg.get("model_kwargs") or {}
    for key, value in model_kwargs.items():
        if value is not None:
            kwargs[key] = value

    if hist_exog_columns and model_name in {"GRU", "TimeXer"}:
        kwargs["hist_exog_list"] = list(hist_exog_columns)

    if model_name in {"TimeXer", "iTransformer"}:
        kwargs["n_series"] = n_series
    return kwargs


def save_loss_history(
    model,
    run_dir: Path,
    training_cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    training_cfg = training_cfg or {}

    csv_history = load_csv_logger_loss_history(run_dir)
    if not csv_history.empty:
        csv_history.to_csv(run_dir / "loss_history.csv", index=False)
        return csv_history

    train_df = pd.DataFrame(model.train_trajectories, columns=["step", "train_loss"])
    valid_df = pd.DataFrame(model.valid_trajectories, columns=["step", "valid_loss"])
    train_df["log_index"] = np.arange(1, len(train_df) + 1)
    valid_df["log_index"] = np.arange(1, len(valid_df) + 1)
    train_df["epoch"] = derive_loss_epoch_axis(train_df)
    valid_df["epoch"] = derive_loss_epoch_axis(valid_df)

    history_df = train_df.merge(valid_df, on="log_index", how="outer", suffixes=("_train", "_valid"))
    train_step_col = "step_train" if "step_train" in history_df.columns else None
    valid_step_col = "step_valid" if "step_valid" in history_df.columns else None
    if train_step_col and valid_step_col:
        history_df["step"] = history_df[train_step_col].combine_first(history_df[valid_step_col])
    elif train_step_col:
        history_df["step"] = history_df[train_step_col]
    elif valid_step_col:
        history_df["step"] = history_df[valid_step_col]
    else:
        history_df["step"] = history_df["log_index"].astype(float)

    train_epoch_col = "epoch_train" if "epoch_train" in history_df.columns else None
    valid_epoch_col = "epoch_valid" if "epoch_valid" in history_df.columns else None
    if train_epoch_col and valid_epoch_col:
        history_df["epoch"] = history_df[train_epoch_col].combine_first(history_df[valid_epoch_col])
    elif train_epoch_col:
        history_df["epoch"] = history_df[train_epoch_col]
    elif valid_epoch_col:
        history_df["epoch"] = history_df[valid_epoch_col]
    else:
        history_df["epoch"] = history_df["log_index"].astype(float)

    drop_cols = [
        column
        for column in ["step_train", "step_valid", "epoch_train", "epoch_valid"]
        if column in history_df.columns
    ]
    if drop_cols:
        history_df = history_df.drop(columns=drop_cols)

    history_df.to_csv(run_dir / "loss_history.csv", index=False)
    return history_df


def load_csv_logger_loss_history(run_dir: Path) -> pd.DataFrame:
    candidates = sorted(run_dir.glob("**/metrics.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            metrics = pd.read_csv(path)
        except Exception:
            continue

        train_col = next(
            (
                column
                for column in ["train_loss_epoch", "train_loss", "train_loss_step"]
                if column in metrics.columns and metrics[column].notna().any()
            ),
            None,
        )
        valid_col = next(
            (
                column
                for column in ["valid_loss", "ptl/val_loss", "val_loss"]
                if column in metrics.columns and metrics[column].notna().any()
            ),
            None,
        )
        if train_col is None and valid_col is None:
            continue

        history = pd.DataFrame(index=metrics.index)
        if "step" in metrics.columns:
            history["step"] = pd.to_numeric(metrics["step"], errors="coerce")
        else:
            history["step"] = np.arange(1, len(metrics) + 1, dtype=float)

        if "epoch" in metrics.columns:
            history["epoch"] = pd.to_numeric(metrics["epoch"], errors="coerce")
        else:
            history["epoch"] = history["step"]

        history["train_loss"] = (
            pd.to_numeric(metrics[train_col], errors="coerce") if train_col is not None else np.nan
        )
        history["valid_loss"] = (
            pd.to_numeric(metrics[valid_col], errors="coerce") if valid_col is not None else np.nan
        )
        history = history.dropna(subset=["train_loss", "valid_loss"], how="all").copy()
        if history.empty:
            continue

        history["step"] = history["step"].ffill().bfill()
        history["epoch"] = history["epoch"].ffill().bfill()
        history["log_index"] = np.arange(1, len(history) + 1)
        history["loss_history_source"] = f"csv_logger:{path.relative_to(run_dir).as_posix()}"
        return history[["train_loss", "log_index", "valid_loss", "step", "epoch", "loss_history_source"]]

    return pd.DataFrame()


def derive_loss_epoch_axis(history_df: pd.DataFrame) -> pd.Series:
    if history_df.empty:
        return pd.Series(dtype=float)

    # NeuralForecast trajectories expose a raw step id, but its meaning changes
    # with windowing and validation cadence. The log order is the stable epoch
    # proxy for our report curves.
    return history_df["log_index"].astype(float)


def plot_loss_curves(
    history_df: pd.DataFrame,
    title: str,
    output_path: Path,
    y_scale: str = "linear",
    x_column: str = "step",
    x_label: str = "Global step",
    x_max: Optional[float] = None,
    normalize_mode: str = "none",
    allow_dual_axis: bool = True,
) -> None:
    y_scale = str(y_scale or "linear").lower().strip()
    if y_scale not in {"linear", "log", "symlog"}:
        raise ValueError(f"Unsupported loss curve y-scale: {y_scale}")
    normalize_mode = str(normalize_mode or "none").lower().replace("_", "-").strip()
    if normalize_mode not in {"none", "initial", "first"}:
        raise ValueError(f"Unsupported loss curve normalization: {normalize_mode}")
    if x_column not in history_df.columns:
        x_column = "step"
        x_label = "Global step"

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    train_loss = pd.Series(dtype=float)
    valid_rows = pd.DataFrame()
    valid_loss = pd.Series(dtype=float)

    if "train_loss" in history_df:
        train_loss = (
            history_df["train_loss"].where(history_df["train_loss"] > 0)
            if y_scale == "log"
            else history_df["train_loss"]
        )

    if "valid_loss" in history_df:
        valid_rows = history_df.dropna(subset=["valid_loss"])
        if not valid_rows.empty:
            valid_loss = (
                valid_rows["valid_loss"].where(valid_rows["valid_loss"] > 0)
                if y_scale == "log"
                else valid_rows["valid_loss"]
            )

    if normalize_mode in {"initial", "first"}:
        train_loss = _normalize_loss_series(train_loss)
        valid_loss = _normalize_loss_series(valid_loss)

    train_scale = _series_scale(train_loss)
    valid_scale = _series_scale(valid_loss)
    use_dual_axis = (
        allow_dual_axis
        and
        normalize_mode == "none"
        and y_scale == "linear"
        and math.isfinite(train_scale)
        and math.isfinite(valid_scale)
        and min(train_scale, valid_scale) > 0
        and max(train_scale, valid_scale) / min(train_scale, valid_scale) >= 25.0
    )

    if use_dual_axis:
        valid_ax = ax.twinx()
        if not train_loss.dropna().empty:
            ax.plot(
                history_df[x_column],
                train_loss,
                label="Train loss (left)",
                color="#1f77b4",
                linewidth=1.4,
            )
            plotted = True
        if not valid_loss.dropna().empty:
            valid_ax.plot(
                valid_rows[x_column],
                valid_loss,
                label="Validation loss (right)",
                color="#ff7f0e",
                linewidth=1.8,
                marker="o",
            )
            plotted = True
        ax.set_ylabel("Train loss")
        valid_ax.set_ylabel("Validation loss")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        valid_ax.tick_params(axis="y", labelcolor="#ff7f0e")
        handles, labels = ax.get_legend_handles_labels()
        valid_handles, valid_labels = valid_ax.get_legend_handles_labels()
        handles.extend(valid_handles)
        labels.extend(valid_labels)
        ax.text(
            0.01,
            0.96,
            "Separate y-axes due to loss scale gap",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
        )
    else:
        if not train_loss.dropna().empty:
            ax.plot(history_df[x_column], train_loss, label="Train loss", linewidth=1.4)
            plotted = True
        if not valid_loss.dropna().empty:
            ax.plot(
                valid_rows[x_column],
                valid_loss,
                label="Validation loss",
                linewidth=1.8,
                marker="o",
            )
            plotted = True
        handles, labels = ax.get_legend_handles_labels()

    if y_scale != "linear":
        ax.set_yscale(y_scale)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    if not use_dual_axis:
        ylabel = "Normalized loss (first logged value = 1.0)" if normalize_mode != "none" else "Loss"
        ax.set_ylabel(ylabel)
    if x_max is not None and math.isfinite(float(x_max)) and float(x_max) > 0:
        ax.set_xlim(left=0, right=float(x_max))
    if plotted:
        ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _series_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    numeric = numeric[numeric > 0]
    if numeric.empty:
        return float("nan")
    return float(numeric.median())


def _normalize_loss_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    baseline_candidates = numeric.dropna()
    baseline_candidates = baseline_candidates[baseline_candidates.abs() > 0]
    if baseline_candidates.empty:
        return numeric
    return numeric / float(baseline_candidates.iloc[0])


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
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape_denom = np.where(smape_denom == 0, np.nan, smape_denom)
    smape = float(np.nanmean(2.0 * np.abs(y_pred - y_true) / smape_denom) * 100)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}


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
                    "MSE": round(float(best.get("MSE", np.nan)), 6),
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
        for key in ["status", "min_val_loss", "final_val_loss", "MSE", "MAE", "RMSE", "MAPE", "sMAPE"]:
            if key in row:
                sections.append(f"<li>{key}: {row[key]}</li>")
        if "issue_note" in row and pd.notna(row["issue_note"]):
            sections.append(f"<li><strong>Issue</strong>: {row['issue_note']}</li>")
        if "next_action_note" in row and pd.notna(row["next_action_note"]):
            sections.append(f"<li><strong>Next action</strong>: {row['next_action_note']}</li>")
        sections.append("</ul>")
        loss_curve = Path(str(row["artifact_dir"])) / "loss_curve.png"
        normalized_loss_curve = Path(str(row["artifact_dir"])) / "loss_curve_normalized.png"
        forecast_plot = Path(str(row["artifact_dir"])) / "forecast_plot.png"
        if loss_curve.exists():
            rel_loss_curve = loss_curve.relative_to(batch_dir) if batch_dir in loss_curve.parents else loss_curve
            sections.append("<p><strong>Raw MSE loss curve</strong></p>")
            sections.append(f"<img src='{rel_loss_curve.as_posix()}' width='900'>")
        if normalized_loss_curve.exists():
            rel_norm_loss_curve = (
                normalized_loss_curve.relative_to(batch_dir)
                if batch_dir in normalized_loss_curve.parents
                else normalized_loss_curve
            )
            sections.append("<p><strong>Normalized loss curve (first logged value = 1.0)</strong></p>")
            sections.append(f"<img src='{rel_norm_loss_curve.as_posix()}' width='900'>")
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
            "MSE",
            "MAE",
            "RMSE",
            "MAPE",
            "sMAPE",
            "min_val_loss",
            "final_val_loss",
        ]
        for col in metrics_cols:
            if col not in best_summary_df.columns:
                best_summary_df[col] = np.nan
        metrics_df = best_summary_df[metrics_cols].copy()
        for col in ["MSE", "MAE", "RMSE", "MAPE", "sMAPE", "min_val_loss", "final_val_loss"]:
            if col in metrics_df:
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


def normalize_company_experiment_key(batch_name: str) -> str:
    for prefix in ["daily_wti_h12_", "weekly_wti_h2_", "company_wti_"]:
        if batch_name.startswith(prefix):
            return batch_name[len(prefix) :]
    return batch_name


def company_experiment_title(experiment_key: str) -> str:
    mapping = {
        "mse_report": "Requested MSE Report Setting",
        "mse_scaled_regularized": "Scaled + Regularized Improvement Setting",
        "mse_raw_report": "Raw Reference Setting",
        "mse_scaled_val_sweep": "Validation Size Sweep Setting",
    }
    return mapping.get(experiment_key, experiment_key.replace("_", " ").title())


def load_best_model_summary(batch_dir: Path) -> pd.DataFrame:
    path = batch_dir / "best_model_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_dataset_result_line(
    best_df: pd.DataFrame,
    dataset: str,
    baseline_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    if best_df.empty or "dataset" not in best_df.columns:
        return [f"{_dataset_label(dataset)} 결과를 요약할 성공 run이 없습니다."]
    rows = best_df[best_df["dataset"] == dataset].copy()
    if rows.empty:
        return [f"{_dataset_label(dataset)} 결과를 요약할 성공 run이 없습니다."]

    lines: List[str] = []
    uni = rows[rows["mode"] == "univariate"]
    multi = rows[rows["mode"] == "multivariate"]
    if not uni.empty:
        row = uni.iloc[0]
        lines.append(
            f"{_dataset_label(dataset)} 단변량에서는 {row['best_model']}가 "
            f"MAE {_float_text(row['MAE'])}, RMSE {_float_text(row['RMSE'])}로 가장 좋았습니다."
        )
    if not multi.empty:
        row = multi.iloc[0]
        lines.append(
            f"{_dataset_label(dataset)} 다변량에서는 {row['best_model']}가 "
            f"MAE {_float_text(row['MAE'])}, RMSE {_float_text(row['RMSE'])}로 가장 좋았습니다."
        )

    if not uni.empty and not multi.empty:
        uni_row = uni.iloc[0]
        multi_row = multi.iloc[0]
        if float(multi_row["RMSE"]) < float(uni_row["RMSE"]):
            lines.append(
                f"{_dataset_label(dataset)}에서는 다변량이 RMSE 기준으로 단변량보다 "
                f"{float(uni_row['RMSE']) - float(multi_row['RMSE']):.6f} 개선됐습니다."
            )
        else:
            lines.append(
                f"{_dataset_label(dataset)}에서는 단변량이 RMSE 기준으로 다변량보다 "
                f"{float(multi_row['RMSE']) - float(uni_row['RMSE']):.6f} 더 안정적이었습니다."
            )

    if baseline_df is not None and not baseline_df.empty:
        baseline_rows = baseline_df[baseline_df["dataset"] == dataset]
        for mode in ["univariate", "multivariate"]:
            current_mode = rows[rows["mode"] == mode]
            baseline_mode = baseline_rows[baseline_rows["mode"] == mode]
            if current_mode.empty or baseline_mode.empty:
                continue
            current_rmse = float(current_mode.iloc[0]["RMSE"])
            baseline_rmse = float(baseline_mode.iloc[0]["RMSE"])
            delta = baseline_rmse - current_rmse
            if abs(delta) > 1e-9:
                lines.append(
                    f"{_dataset_label(dataset)} {mode}에서는 baseline 대비 RMSE가 "
                    f"{delta:+.6f} 변했습니다."
                )
    return lines


def summarize_dataset_insights(best_df: pd.DataFrame, summary_df: pd.DataFrame, dataset: str) -> List[str]:
    if best_df.empty or "dataset" not in best_df.columns:
        rows = pd.DataFrame()
    else:
        rows = best_df[best_df["dataset"] == dataset].copy()
    dataset_rows = summary_df[(summary_df["dataset"] == dataset) & (summary_df["status"] != "failed")].copy()
    lines: List[str] = []
    if not rows.empty:
        if {"min_val_loss", "final_val_loss"}.issubset(rows.columns):
            improving_modes = []
            for _, row in rows.iterrows():
                if float(row["final_val_loss"]) <= float(row["min_val_loss"]) * 1.05:
                    improving_modes.append(_mode_label(str(row["mode"])))
            if improving_modes:
                lines.append(
                    f"{_dataset_label(dataset)}에서는 {', '.join(improving_modes)} 설정이 validation rebound 없이 비교적 안정적으로 수렴했습니다."
                )
    if not dataset_rows.empty and "issue_note" in dataset_rows.columns:
        issue_note = str(dataset_rows.sort_values(["RMSE", "MAE"]).iloc[0].get("issue_note", "")).strip()
        if issue_note:
            lines.append(issue_note)
    if not lines:
        lines.append(f"{_dataset_label(dataset)} 결과는 최종 지표와 예측 그래프를 함께 봐야 해석이 가능합니다.")
    return lines[:3]


def build_company_conclusion_sections(
    group_df: pd.DataFrame,
    best_df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame],
) -> Dict[str, List[str]]:
    datasets = list(best_df["dataset"].dropna().unique()) if not best_df.empty else []
    current_status = [
        f"이번 실험군은 {', '.join(datasets) if datasets else '선택된 dataset'}에 대해 단변량과 다변량을 같은 보고 형식으로 비교합니다.",
        f"성공 run 기준 모델 수는 {int((group_df['status'] != 'failed').sum())}개이고 실패 run 수는 {int((group_df['status'] == 'failed').sum())}개입니다.",
        "Multivariate는 단일 타깃에 historical exogenous feature를 붙이는 구조로 계산되어 loss와 metric이 타깃 기준으로 정렬됩니다.",
        "리포트의 실제값 대 예측값 그래프와 최종 metric 표를 함께 보고 결론을 내려야 합니다.",
    ]

    meaning = [
        "단변량과 다변량의 우열은 raw loss 숫자보다 target 기준 MAE, RMSE, MAPE, sMAPE에서 판단하는 것이 맞습니다.",
        "Validation loss와 최종 예측 metric이 같이 좋아질 때만 구조 변경이나 정규화가 실제로 도움이 되었다고 해석할 수 있습니다.",
        "Daily와 Weekly는 horizon과 데이터 밀도가 달라서 같은 모델이라도 학습 양상과 metric 해석이 달라질 수 있습니다.",
        "이번 결과는 이후 seed ensemble이나 추가 튜닝을 붙이기 전에 baseline과 개선 방향을 가르는 기준점 역할을 합니다.",
    ]
    if baseline_df is not None and not baseline_df.empty and not best_df.empty:
        meaning[1] = "개선 실험은 baseline 대비 metric 변화량을 함께 보면서 정규화와 exogenous 구조가 실제 성능 개선으로 이어졌는지 판단합니다."

    next_action = [
        "다음 단계에서는 결과가 안정적인 설정을 기준 run으로 고정하고 seed ensemble로 재현성을 확인합니다.",
        "Daily horizon과 Weekly horizon은 각각 실무 의사결정 단위에 맞춰 별도로 조정하는 것이 좋습니다.",
        "Non-lag 거시 변수는 실제 공표 시점 기준으로 한 번 더 검토해 semantic leakage 가능성을 줄이는 것이 필요합니다.",
        "최종 공유본에는 통합 report.md와 함께 핵심 metric 표, 실제 예측 그래프, loss curve만 남기고 나머지 탐색 결과는 부록으로 두는 것이 좋습니다.",
    ]
    return {"current_status": current_status, "meaning": meaning, "next_action": next_action}


def build_company_master_report(batch_results: List[BatchResult], output_dir: Path) -> Path:
    grouped: Dict[str, Dict[str, BatchResult]] = {}
    ordered_keys: List[str] = []
    for result in batch_results:
        experiment_key = normalize_company_experiment_key(result.batch_config["name"])
        dataset_names = [str(v) for v in result.summary_df["dataset"].dropna().unique()]
        dataset_name = dataset_names[0] if dataset_names else (
            "daily" if "daily" in result.batch_config["name"] else "weekly"
        )
        if experiment_key not in grouped:
            grouped[experiment_key] = {}
            ordered_keys.append(experiment_key)
        grouped[experiment_key][dataset_name] = result

    baseline_key = "mse_report" if "mse_report" in grouped else (ordered_keys[0] if ordered_keys else "")
    lines = [
        "# WTI Combined Review Report",
        "",
        f"- Generated at: {datetime.utcnow().isoformat()}Z",
        "",
    ]

    dataset_order = {"daily": 0, "weekly": 1}
    mode_order = {"univariate": 0, "multivariate": 1}
    model_order = {name: idx for idx, name in enumerate(MODEL_REGISTRY.keys())}

    if not ordered_keys:
        output_path = output_dir / "master_report.md"
        output_path.write_text("\n".join(lines + ["실행된 배치가 없어 통합 리포트를 생성하지 못했습니다."]) + "\n", encoding="utf-8")
        return output_path

    for idx, experiment_key in enumerate(ordered_keys, start=1):
        dataset_map = grouped[experiment_key]
        summary_frames = [result.summary_df.copy() for result in dataset_map.values()]
        group_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        all_success = group_df[group_df["status"] != "failed"].copy() if not group_df.empty else pd.DataFrame()
        any_result = next(iter(dataset_map.values()))
        best_frames = [load_best_model_summary(result.batch_dir) for result in dataset_map.values()]
        non_empty_best_frames = [df for df in best_frames if not df.empty]
        best_df = pd.concat(non_empty_best_frames, ignore_index=True) if non_empty_best_frames else pd.DataFrame()
        baseline_best_df = None
        if experiment_key != baseline_key and baseline_key in grouped:
            baseline_frames = [load_best_model_summary(result.batch_dir) for result in grouped[baseline_key].values()]
            non_empty_baseline_frames = [df for df in baseline_frames if not df.empty]
            baseline_best_df = (
                pd.concat(non_empty_baseline_frames, ignore_index=True) if non_empty_baseline_frames else pd.DataFrame()
            )

        if not best_df.empty:
            best_df["dataset_order"] = best_df["dataset"].map(dataset_order).fillna(99)
            best_df["mode_order"] = best_df["mode"].map(mode_order).fillna(99)
            best_df = best_df.sort_values(["dataset_order", "mode_order"]).drop(
                columns=["dataset_order", "mode_order"]
            )

        methodology_parts = []
        for dataset in ["daily", "weekly"]:
            if dataset not in dataset_map:
                continue
            result = dataset_map[dataset]
            successful = result.summary_df[result.summary_df["status"] != "failed"]
            horizon = int(successful["horizon"].iloc[0]) if not successful.empty else ""
            loss_name = str(successful["loss_name"].iloc[0]).upper() if not successful.empty else "default"
            scaler_name = str(successful["scaler_type"].iloc[0]) if not successful.empty else "default"
            methodology_parts.append(
                f"{_dataset_label(dataset)} horizon={horizon}, loss={loss_name}, scaler={scaler_name}"
            )
        methodology_text = ", ".join(methodology_parts)
        methodology_suffix = (
            "단변량은 target-only, 다변량은 single-target + historical exogenous features 구조로 비교했습니다."
        )
        if experiment_key == "mse_scaled_regularized":
            methodology_text += ", ReVIN + weight decay/dropout/model downsizing"
        else:
            methodology_text += ", MSE + scheduler + early stopping"
        methodology_text = f"{methodology_text}. {methodology_suffix}"

        lines.extend(
            [
                f"## 실험 {idx}. {company_experiment_title(experiment_key)}",
                "",
                "- 예측 타깃: WTI proxy (Com_CrudeOil)",
                f"- 모델 종류: {', '.join(any_result.batch_config.get('models', []))}",
                f"- 실험 방법론({experiment_key}): {methodology_text}",
            ]
        )

        for dataset in ["daily", "weekly"]:
            if dataset not in dataset_map:
                continue
            result = dataset_map[dataset]
            successful = result.summary_df[result.summary_df["status"] != "failed"]
            if successful.empty:
                continue
            row = successful.iloc[0]
            lines.append(
                "- 실제 날짜 세팅: "
                f"{_dataset_label(dataset)} / "
                f"Train {_date_text(row.get('train_start_date'))} ~ {_date_text(row.get('train_end_date'))}, "
                f"Val {_date_text(row.get('val_start_date'))} ~ {_date_text(row.get('val_end_date'))}, "
                f"Test {_date_text(row.get('test_start_date'))} ~ {_date_text(row.get('test_end_date'))}"
            )

        for dataset in ["daily", "weekly"]:
            if best_df.empty or dataset not in set(best_df["dataset"]):
                continue
            for line in summarize_dataset_result_line(best_df, dataset, baseline_best_df):
                lines.append(f"- 주요 실험 결과: {line}")

        insight_lines: List[str] = []
        for dataset in ["daily", "weekly"]:
            if dataset in dataset_map:
                insight_lines.extend(summarize_dataset_insights(best_df, dataset_map[dataset].summary_df, dataset))
        for insight in insight_lines[:6]:
            lines.append(f"- 내가 생각한 인사이트: {insight}")

        if "weekly" in dataset_map:
            result = dataset_map["weekly"]
            weekly_rows = result.summary_df[result.summary_df["status"] != "failed"].copy()
            if not weekly_rows.empty:
                weekly_rows["mode_order"] = weekly_rows["mode"].map(mode_order).fillna(99)
                weekly_rows["model_order"] = weekly_rows["model"].map(model_order).fillna(99)
                weekly_rows = weekly_rows.sort_values(["mode_order", "model_order"]).drop(
                    columns=["mode_order", "model_order"]
                )
            lines.extend(["", "## Weekly Loss Curve", ""])
            for mode in ["univariate", "multivariate"]:
                mode_rows = weekly_rows[weekly_rows["mode"] == mode].copy()
                lines.extend(["", f"### {_mode_label(mode)} 모델별 그래프", ""])
                for _, row in mode_rows.iterrows():
                    loss_curve_path = Path(str(row["artifact_dir"])) / "loss_curve.png"
                    normalized_loss_curve_path = Path(str(row["artifact_dir"])) / "loss_curve_normalized.png"
                    if loss_curve_path.exists():
                        alt_text = f"{row['run_name']} loss curve"
                        lines.append(f"- {row['model']} raw MSE: {_markdown_image(loss_curve_path, alt_text)}")
                    if normalized_loss_curve_path.exists():
                        alt_text = f"{row['run_name']} normalized loss curve"
                        lines.append(
                            f"- {row['model']} normalized: {_markdown_image(normalized_loss_curve_path, alt_text)}"
                        )

        if "daily" in dataset_map:
            result = dataset_map["daily"]
            daily_rows = result.summary_df[result.summary_df["status"] != "failed"].copy()
            if not daily_rows.empty:
                daily_rows["mode_order"] = daily_rows["mode"].map(mode_order).fillna(99)
                daily_rows["model_order"] = daily_rows["model"].map(model_order).fillna(99)
                daily_rows = daily_rows.sort_values(["mode_order", "model_order"]).drop(
                    columns=["mode_order", "model_order"]
                )
            lines.extend(["", "## Daily Loss Curve", ""])
            for mode in ["univariate", "multivariate"]:
                mode_rows = daily_rows[daily_rows["mode"] == mode].copy()
                lines.extend(["", f"### {_mode_label(mode)} 모델별 그래프", ""])
                for _, row in mode_rows.iterrows():
                    loss_curve_path = Path(str(row["artifact_dir"])) / "loss_curve.png"
                    normalized_loss_curve_path = Path(str(row["artifact_dir"])) / "loss_curve_normalized.png"
                    if loss_curve_path.exists():
                        alt_text = f"{row['run_name']} loss curve"
                        lines.append(f"- {row['model']} raw MSE: {_markdown_image(loss_curve_path, alt_text)}")
                    if normalized_loss_curve_path.exists():
                        alt_text = f"{row['run_name']} normalized loss curve"
                        lines.append(
                            f"- {row['model']} normalized: {_markdown_image(normalized_loss_curve_path, alt_text)}"
                        )

        if "daily" in dataset_map:
            best_daily = (
                best_df[best_df["dataset"] == "daily"].copy()
                if (not best_df.empty and "dataset" in best_df.columns)
                else pd.DataFrame()
            )
            horizon = int(best_daily["horizon"].iloc[0]) if not best_daily.empty else ""
            comp_path = dataset_map["daily"].batch_dir / "daily_actual_vs_uni_multi.png"
            if comp_path.exists():
                lines.extend(
                    [
                        "",
                        f"## Daily 실제값 vs 예측값 (horizon {horizon})",
                        "",
                        f"- actual vs uni vs multi: {_markdown_image(comp_path, 'daily actual vs uni vs multi')}",
                    ]
                )

        if "weekly" in dataset_map:
            best_weekly = (
                best_df[best_df["dataset"] == "weekly"].copy()
                if (not best_df.empty and "dataset" in best_df.columns)
                else pd.DataFrame()
            )
            horizon = int(best_weekly["horizon"].iloc[0]) if not best_weekly.empty else ""
            comp_path = dataset_map["weekly"].batch_dir / "weekly_actual_vs_uni_multi.png"
            if comp_path.exists():
                lines.extend(
                    [
                        "",
                        f"## Weekly 실제값 vs 예측값 (horizon {horizon})",
                        "",
                        f"- actual vs uni vs multi: {_markdown_image(comp_path, 'weekly actual vs uni vs multi')}",
                    ]
                )

        lines.extend(["", "## 지표 요약", ""])
        if not best_df.empty:
            metrics_cols = [
                "dataset",
                "mode",
                "best_model",
                "MSE",
                "MAE",
                "RMSE",
                "MAPE",
                "sMAPE",
                "min_val_loss",
                "final_val_loss",
            ]
            for col in metrics_cols:
                if col not in best_df.columns:
                    best_df[col] = np.nan
            metrics_df = best_df[metrics_cols].copy()
            metrics_df["dataset_order"] = metrics_df["dataset"].map(dataset_order).fillna(99)
            metrics_df["mode_order"] = metrics_df["mode"].map(mode_order).fillna(99)
            metrics_df = metrics_df.sort_values(["dataset_order", "mode_order"]).drop(
                columns=["dataset_order", "mode_order"]
            )
            for col in ["MSE", "MAE", "RMSE", "MAPE", "sMAPE", "min_val_loss", "final_val_loss"]:
                if col in metrics_df:
                    metrics_df[col] = metrics_df[col].map(lambda x: _float_text(x, digits=6))
            lines.append(dataframe_to_markdown_table(metrics_df))
        else:
            lines.append("요약 가능한 성공 run이 없습니다.")

        conclusion = build_company_conclusion_sections(all_success, best_df, baseline_best_df)
        lines.extend(["", "## 결론 및 향후 계획", "", "### 현재 상태"])
        for sentence in conclusion["current_status"][:4]:
            lines.append(f"- {sentence}")
        lines.extend(["", "### 의미"])
        for sentence in conclusion["meaning"][:4]:
            lines.append(f"- {sentence}")
        lines.extend(["", "### 다음 액션"])
        for sentence in conclusion["next_action"][:4]:
            lines.append(f"- {sentence}")
        if idx != len(ordered_keys):
            lines.extend(["", "---", ""])

    output_path = output_dir / "master_report.md"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def run_single_experiment(
    repo_root: Path,
    batch_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    feature_manifest: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    model_name: str,
    run_dir: Path,
) -> Dict[str, Any]:
    ensure_neuralforecast_runtime()
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
    loss_csv_logger = build_loss_csv_logger(run_dir)
    if loss_csv_logger is not None:
        # Match the reference notebook: plot target-level epoch metrics from CSVLogger
        # instead of raw NeuralForecast trajectory buffers when available.
        kwargs["logger"] = loss_csv_logger
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**kwargs)
    nf = NeuralForecast(models=[model], freq=dataset_cfg["data"]["freq"])
    nf.fit(df=train_val_df, val_size=val_size)
    preds_df = nf.predict().sort_values(["unique_id", "ds"]).reset_index(drop=True)

    target_pred_rows = preds_df[preds_df["unique_id"] == display_target_name].copy()
    target_pred_rows["ds"] = pd.to_datetime(target_pred_rows["ds"])
    target_predictions = target_pred_rows[model_name].to_numpy()
    if len(target_predictions) != len(test_dates):
        raise ValueError(
            f"Prediction length mismatch for {scenario_cfg['name']} / {model_name}: "
            f"received {len(target_predictions)} predictions for {len(test_dates)} test timestamps."
        )
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
    history_df = save_loss_history(fitted_model, run_dir, training_cfg=training_cfg)
    diagnostics = diagnose_run(history_df)
    loss_history_source = (
        str(history_df["loss_history_source"].dropna().iloc[0])
        if "loss_history_source" in history_df.columns and history_df["loss_history_source"].notna().any()
        else "neuralforecast_trajectories"
    )
    x_column = "epoch" if x_axis_mode == "epoch" and "epoch" in history_df.columns else "step"
    x_label = "Epoch" if x_column == "epoch" else "Global step"
    x_max = training_cfg.get("requested_max_epochs") if x_column == "epoch" else None
    plot_loss_curves(
        history_df,
        f"{scenario_cfg['name']} - {model_name} loss curves",
        run_dir / "loss_curve.png",
        y_scale=loss_scale,
        x_column=x_column,
        x_label=x_label,
        x_max=x_max,
    )
    plot_loss_curves(
        history_df,
        f"{scenario_cfg['name']} - {model_name} normalized loss curves",
        run_dir / "loss_curve_normalized.png",
        y_scale="linear",
        x_column=x_column,
        x_label=x_label,
        x_max=x_max,
        normalize_mode="initial",
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
        "loss_history_source": loss_history_source,
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
        "loss_history_source": loss_history_source,
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
        batch_config=batch_cfg,
    )
