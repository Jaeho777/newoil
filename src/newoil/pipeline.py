from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from neuralforecast import NeuralForecast
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


def default_input_size(horizon: int, patch_len: int = 16) -> int:
    base = max(3 * horizon, patch_len)
    return int(math.ceil(base / patch_len) * patch_len)


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
    if fill_method == "ffill":
        panel = panel.ffill()
    elif fill_method == "bfill":
        panel = panel.bfill()
    elif fill_method == "ffill_bfill":
        panel = panel.ffill().bfill()
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


def apply_transform(panel: pd.DataFrame, transform_name: str) -> pd.DataFrame:
    if transform_name == "none":
        return panel.copy()
    if transform_name == "diff":
        transformed = panel.diff().dropna()
        if transformed.empty:
            raise ValueError("Differencing removed all rows.")
        return transformed
    raise ValueError(f"Unsupported transform: {transform_name}")


def panel_to_long(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.reset_index()
        .melt(id_vars="ds", var_name="unique_id", value_name="y")
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
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

    if model_name in {"TimeXer", "iTransformer"}:
        kwargs["n_series"] = n_series
    return kwargs


def save_loss_history(model, run_dir: Path) -> pd.DataFrame:
    train_df = pd.DataFrame(model.train_trajectories, columns=["step", "train_loss"])
    valid_df = pd.DataFrame(model.valid_trajectories, columns=["step", "valid_loss"])
    history_df = train_df.merge(valid_df, on="step", how="outer").sort_values("step")
    history_df.to_csv(run_dir / "loss_history.csv", index=False)
    return history_df


def plot_loss_curves(history_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if "train_loss" in history_df:
        ax.plot(history_df["step"], history_df["train_loss"], label="Train loss", linewidth=1.4)
    if "valid_loss" in history_df:
        valid_rows = history_df.dropna(subset=["valid_loss"])
        if not valid_rows.empty:
            ax.plot(
                valid_rows["step"],
                valid_rows["valid_loss"],
                label="Validation loss",
                linewidth=1.8,
                marker="o",
            )
    ax.set_title(title)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def invert_predictions(
    transform_name: str,
    target_predictions: np.ndarray,
    original_panel: pd.DataFrame,
    transformed_panel: pd.DataFrame,
    target_column: str,
    test_size: int,
) -> np.ndarray:
    if transform_name == "none":
        return target_predictions
    if transform_name == "diff":
        train_end_date = transformed_panel.index[-test_size - 1]
        last_actual = float(original_panel[target_column].loc[train_end_date])
        return last_actual + np.cumsum(target_predictions)
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


def build_markdown_report(batch_cfg: Dict[str, Any], summary_df: pd.DataFrame) -> str:
    lines = [
        f"# {batch_cfg['name']}",
        "",
        f"- Generated at: {datetime.utcnow().isoformat()}Z",
        f"- Start date filter: {batch_cfg['start_date']}",
        "",
        "## Summary",
        "",
        summary_df.to_csv(index=False),
    ]
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

    preprocess_cfg = dict(dataset_cfg.get("preprocess", {}))
    preprocess_cfg.update(scenario_cfg.get("preprocess", {}))

    original_panel = prepare_panel(
        repo_root=repo_root,
        dataset_cfg=dataset_cfg,
        selected_columns=selected_columns,
        start_date=batch_cfg["start_date"],
        preprocess_cfg=preprocess_cfg,
    )

    transform_name = scenario_cfg.get("transform", "none")
    transformed_panel = apply_transform(original_panel, transform_name)
    scenario_cfg = dict(scenario_cfg)
    horizon = int(scenario_cfg["horizon"])
    val_size = int(scenario_cfg["val_size"])
    test_size = int(scenario_cfg["test_size"])
    patch_len = int(scenario_cfg.get("patch_len", 16))
    training_cfg = dict(batch_cfg.get("training_defaults", {}))
    training_cfg.update(scenario_cfg.get("training", {}))
    random_seed = int(batch_cfg.get("runtime", {}).get("random_seed", 1))
    input_size = int(training_cfg.get("input_size", default_input_size(horizon, patch_len)))
    lookback_plot = int(scenario_cfg.get("lookback_plot", batch_cfg.get("report", {}).get("lookback_plot", 120)))

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

    train_val_panel = transformed_panel.iloc[:-test_size]
    transformed_long = panel_to_long(transformed_panel)
    train_val_df = panel_to_long(train_val_panel)
    n_series = len(selected_columns)

    kwargs = build_common_model_kwargs(
        model_name=model_name,
        horizon=horizon,
        input_size=input_size,
        n_series=n_series,
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
    test_dates = transformed_panel.index[-test_size:]
    actual_target = original_panel[target_column].loc[test_dates]
    predicted_target = invert_predictions(
        transform_name=transform_name,
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
    plot_loss_curves(history_df, f"{scenario_cfg['name']} - {model_name} loss curves", run_dir / "loss_curve.png")
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
        "transform": transform_name,
        "input_size": input_size,
        "rows_after_preprocessing": int(len(original_panel)),
        "rows_after_transform": int(len(transformed_panel)),
    }
    dump_yaml(run_dir / "config_snapshot.yaml", snapshot)

    summary = {
        "run_name": f"{scenario_cfg['name']}__{model_name}",
        "scenario_name": scenario_cfg["name"],
        "dataset": scenario_cfg["dataset"],
        "mode": scenario_cfg["mode"],
        "model": model_name,
        "transform": transform_name,
        "target_column": target_column,
        "n_series": n_series,
        "rows_after_preprocessing": int(len(original_panel)),
        "rows_after_transform": int(len(transformed_panel)),
        "horizon": horizon,
        "val_size": val_size,
        "test_size": test_size,
        "artifact_dir": str(run_dir),
        **diagnostics,
        **metrics,
    }
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
                    "target_column": feature_manifest["target"]["source_column"],
                    "artifact_dir": str(run_dir),
                    "status": "failed",
                    "error_message": error_message,
                }
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(batch_dir / "summary.csv", index=False)

    report_html = batch_dir / "report.html"
    report_markdown = batch_dir / "report.md"
    report_html.write_text(build_html_report(batch_cfg, summary_df, batch_dir), encoding="utf-8")
    report_markdown.write_text(build_markdown_report(batch_cfg, summary_df), encoding="utf-8")
    dump_yaml(batch_dir / "batch_config_snapshot.yaml", batch_cfg)
    return BatchResult(
        batch_dir=batch_dir,
        summary_df=summary_df,
        report_html=report_html,
        report_markdown=report_markdown,
    )
