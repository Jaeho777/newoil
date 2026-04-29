from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


VARIABLES = [
    "USEPUINDXD",
    "Idx_SnPVIX",
    "Com_DubaiOil",
    "Com_OmanOil",
    "Com_BrentCrudeOil",
    "Com_CrudeOil",
    "Com_Petronet_Kerosene",
    "Com_Petronet_HSFO_180cst_3_5pct",
    "Com_Petronet_Naphtha",
    "Com_Petronet_Gasoline_92RON",
    "GPRD_THREAT",
    "GPRD",
    "GPRD_ACT",
    "N10D",
]

TARGET_COL = "Com_CrudeOil"
UNIQUE_ID = "WTI_log_return"
DEFAULT_MODELS = ["GRU", "TimeXer", "CustomITransformer"]


def detect_accelerator(requested_devices: int) -> tuple[str, int]:
    if torch.cuda.is_available():
        device_count = max(torch.cuda.device_count(), 1)
        return "gpu", max(min(int(requested_devices), device_count), 1)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def weekly_panel(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    data_path = repo_root / "data" / "0428DB_weekly.csv"
    frame = pd.read_csv(data_path)
    missing = [column for column in ["dt", *VARIABLES] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required weekly columns: {missing}")

    panel = frame[["dt", *VARIABLES]].copy()
    panel["dt"] = pd.to_datetime(panel["dt"])
    panel = panel.sort_values("dt").drop_duplicates("dt").set_index("dt")
    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="W-MON"))
    panel.index.name = "ds"

    exog_cols = [column for column in VARIABLES if column != TARGET_COL]
    panel[exog_cols] = panel[exog_cols].ffill()
    panel = panel.dropna(subset=[TARGET_COL])

    non_positive = panel[TARGET_COL] <= 0
    if non_positive.any():
        last_bad_date = panel.index[non_positive][-1]
        panel = panel.loc[panel.index > last_bad_date].copy()
    else:
        last_bad_date = None

    panel["target_log_return"] = np.log(panel[TARGET_COL]).diff()
    model_panel = panel.dropna(subset=["target_log_return", *exog_cols]).copy()
    if model_panel.empty:
        raise ValueError("No rows remain after target log-return transform.")

    metadata = {
        "data_path": str(data_path),
        "target_col": TARGET_COL,
        "variables": VARIABLES,
        "exog_cols": exog_cols,
        "last_non_positive_target_date": str(last_bad_date.date()) if last_bad_date is not None else "",
        "model_start_date": str(model_panel.index.min().date()),
        "model_end_date": str(model_panel.index.max().date()),
        "rows_after_log_return": int(len(model_panel)),
    }
    return panel, model_panel, exog_cols, metadata


def make_nf_df(model_panel: pd.DataFrame, exog_cols: List[str]) -> pd.DataFrame:
    nf_df = model_panel[["target_log_return", *exog_cols]].reset_index().rename(
        columns={"target_log_return": "y"}
    )
    nf_df["unique_id"] = UNIQUE_ID
    return nf_df[["unique_id", "ds", "y", *exog_cols]]


class OilWindowDataset(Dataset):
    def __init__(self, arr: np.ndarray, input_size: int, horizon: int, indices: np.ndarray):
        self.arr = np.asarray(arr, dtype=np.float32)
        self.input_size = int(input_size)
        self.horizon = int(horizon)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.indices[idx])
        x = self.arr[start : start + self.input_size, :]
        y = self.arr[start + self.input_size : start + self.input_size + self.horizon, 0]
        return torch.from_numpy(x), torch.from_numpy(y)


class InvertedDataEmbedding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x))


class TargetOnlyITransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enc_embedding = InvertedDataEmbedding(seq_len, d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.projector = nn.Linear(d_model, pred_len)
        self.n_vars = n_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = centered / stdev
        encoded = self.encoder(self.enc_embedding(x_norm))
        out = self.projector(encoded).permute(0, 2, 1)
        out = out * stdev[:, 0, :].unsqueeze(1)
        out = out + means[:, 0, :].unsqueeze(1)
        return out[:, :, 0]


def make_train_val_window_indices(n_rows: int, input_size: int, horizon: int, val_size: int) -> tuple[np.ndarray, np.ndarray]:
    max_start = n_rows - input_size - horizon + 1
    if max_start <= 0:
        raise ValueError("Not enough rows for custom iTransformer windowing.")
    val_start_idx = n_rows - val_size
    train_idx: List[int] = []
    val_idx: List[int] = []
    for start in range(max_start):
        target_start = start + input_size
        target_end = target_start + horizon
        if target_end <= val_start_idx:
            train_idx.append(start)
        elif target_start >= val_start_idx and target_end <= n_rows:
            val_idx.append(start)
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


def torch_device_for(accelerator: str) -> torch.device:
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if accelerator == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    exog_cols: List[str],
    accelerator: str,
    devices: int,
    h: int,
    input_size: int,
    max_steps: int,
    random_seed: int,
    logger: Any = False,
):
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast.models import GRU, TimeXer

    common = dict(
        h=h,
        input_size=input_size,
        loss=MSE(),
        valid_loss=MSE(),
        scaler_type="robust",
        batch_size=32,
        valid_batch_size=32,
        windows_batch_size=128,
        inference_windows_batch_size=1024,
        max_steps=max_steps,
        val_check_steps=1,
        early_stop_patience_steps=-1,
        step_size=1,
        learning_rate=2e-4,
        random_seed=random_seed,
        hist_exog_list=exog_cols,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=logger,
    )
    if model_name == "GRU":
        return GRU(
            **common,
            encoder_hidden_size=48,
            decoder_hidden_size=48,
            encoder_dropout=0.25,
        )
    if model_name == "TimeXer":
        timexer_kwargs = {**common, "learning_rate": 3e-4}
        return TimeXer(
            **timexer_kwargs,
            n_series=1,
            hidden_size=128,
            n_heads=4,
            d_ff=512,
            dropout=0.15,
        )
    raise ValueError(f"Unsupported model for historical exogenous CV: {model_name}")


def run_custom_itransformer(
    model_panel: pd.DataFrame,
    panel: pd.DataFrame,
    exog_cols: List[str],
    args: argparse.Namespace,
    output_root: Path,
    accelerator: str,
    cv_step_plan: Dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    set_torch_seed(args.seed)
    device = torch_device_for(accelerator)
    feature_cols = ["target_log_return", *exog_cols]
    train_base_length = max(
        len(model_panel) - args.horizon - ((args.n_windows - 1) * args.step_size),
        args.input_size + args.horizon + args.val_size,
    )
    train_base = model_panel.iloc[:train_base_length].copy()
    arr = train_base[feature_cols].to_numpy(dtype=np.float32)
    train_idx, val_idx = make_train_val_window_indices(
        n_rows=len(arr),
        input_size=args.input_size,
        horizon=args.horizon,
        val_size=args.val_size,
    )

    train_loader = DataLoader(
        OilWindowDataset(arr, args.input_size, args.horizon, train_idx),
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        OilWindowDataset(arr, args.input_size, args.horizon, val_idx),
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )

    model = TargetOnlyITransformer(
        seq_len=args.input_size,
        pred_len=args.horizon,
        n_vars=len(feature_cols),
        d_model=128,
        n_heads=4,
        e_layers=2,
        d_ff=256,
        dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    loss_rows: List[Dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.detach().cpu().item()))

        loss_rows.append(
            {
                "epoch": epoch,
                "step": epoch * cv_step_plan["estimated_steps_per_epoch"],
                "log_index": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                "valid_loss": float(np.mean(val_losses)) if val_losses else np.nan,
                "loss_history_source": "custom_itransformer",
            }
        )

    full_arr = model_panel[feature_cols].to_numpy(dtype=np.float32)
    records: List[Dict[str, Any]] = []
    last_cutoff_pos = len(model_panel) - args.horizon - 1
    cutoff_positions = range(
        last_cutoff_pos - ((args.n_windows - 1) * args.step_size),
        last_cutoff_pos + 1,
        args.step_size,
    )
    model.eval()
    with torch.no_grad():
        for cutoff_pos in cutoff_positions:
            x_start = cutoff_pos - args.input_size + 1
            x_end = cutoff_pos + 1
            if x_start < 0:
                continue
            x = torch.from_numpy(full_arr[x_start:x_end, :]).unsqueeze(0).to(device)
            pred_returns = model(x).detach().cpu().numpy().reshape(-1)
            cutoff_date = model_panel.index[cutoff_pos]
            base_price = float(panel.loc[cutoff_date, TARGET_COL])
            cumulative_return = 0.0
            for horizon_idx, pred_return in enumerate(pred_returns, start=1):
                target_pos = cutoff_pos + horizon_idx
                if target_pos >= len(model_panel):
                    continue
                ds = model_panel.index[target_pos]
                cumulative_return += float(pred_return)
                records.append(
                    {
                        "unique_id": UNIQUE_ID,
                        "ds": ds,
                        "cutoff": cutoff_date,
                        "pred_log_return": float(pred_return),
                        "actual_log_return": float(model_panel.iloc[target_pos]["target_log_return"]),
                        "actual_price": float(panel.loc[ds, TARGET_COL]),
                        "horizon_idx": horizon_idx,
                        "predicted_price": float(base_price * np.exp(cumulative_return)),
                    }
                )

    loss_dir = output_root / "CustomITransformer_loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    loss_history = pd.DataFrame(loss_rows)
    loss_history.to_csv(loss_dir / "loss_history.csv", index=False)
    predictions = pd.DataFrame(records)
    return predictions, loss_history, loss_dir


def estimate_epoch_equivalent_max_steps(
    pipeline_module: Any,
    train_length: int,
    input_size: int,
    horizon: int,
    max_epochs: int,
    max_steps_override: int | None,
) -> Dict[str, int]:
    training_cfg = {
        "batch_size": 32,
        "windows_batch_size": 128,
    }
    steps_per_epoch = pipeline_module.estimate_steps_per_epoch(
        train_length=train_length,
        input_size=input_size,
        horizon=horizon,
        n_series=1,
        training_cfg=training_cfg,
    )
    epoch_equivalent_max_steps = int(max_epochs) * int(steps_per_epoch)
    max_steps = (
        min(int(max_steps_override), epoch_equivalent_max_steps)
        if max_steps_override is not None
        else epoch_equivalent_max_steps
    )
    return {
        "requested_max_epochs": int(max_epochs),
        "estimated_steps_per_epoch": int(steps_per_epoch),
        "epoch_equivalent_max_steps": int(epoch_equivalent_max_steps),
        "max_steps": int(max_steps),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"MSE": mse, "MAE": mae, "MAPE": mape}


def find_prediction_column(cv_df: pd.DataFrame, model_name: str) -> str:
    if model_name in cv_df.columns:
        return model_name
    blocked = {"unique_id", "ds", "cutoff", "y"}
    candidates = [
        column for column in cv_df.columns if column not in blocked and pd.api.types.is_numeric_dtype(cv_df[column])
    ]
    if not candidates:
        raise ValueError(f"No prediction column found for {model_name}. Columns={list(cv_df.columns)}")
    return candidates[0]


def attach_price_predictions(cv_df: pd.DataFrame, panel: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    out = cv_df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["cutoff"] = pd.to_datetime(out["cutoff"])
    out["pred_log_return"] = pd.to_numeric(out[pred_col], errors="coerce")
    out["actual_log_return"] = pd.to_numeric(out["y"], errors="coerce")
    out["actual_price"] = out["ds"].map(panel[TARGET_COL])

    predicted_prices = []
    horizons = []
    for _, group in out.sort_values(["cutoff", "ds"]).groupby("cutoff", sort=False):
        cutoff = pd.Timestamp(group["cutoff"].iloc[0])
        if cutoff not in panel.index:
            predicted_prices.extend([np.nan] * len(group))
            horizons.extend([np.nan] * len(group))
            continue
        base_price = float(panel.loc[cutoff, TARGET_COL])
        cumulative_return = 0.0
        for horizon_idx, (_, row) in enumerate(group.iterrows(), start=1):
            cumulative_return += float(row["pred_log_return"])
            predicted_prices.append(base_price * np.exp(cumulative_return))
            horizons.append(horizon_idx)

    out["horizon_idx"] = horizons
    out["predicted_price"] = predicted_prices
    out = out.dropna(
        subset=["actual_log_return", "pred_log_return", "actual_price", "predicted_price", "horizon_idx"]
    ).copy()
    out["horizon_idx"] = out["horizon_idx"].astype(int)
    return out


def save_prediction_plot(all_predictions: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    actual = (
        all_predictions[["ds", "actual_price"]]
        .drop_duplicates("ds")
        .sort_values("ds")
    )
    ax.plot(actual["ds"], actual["actual_price"], color="black", linewidth=2.4, marker="o", label="Actual WTI")
    for model_name, group in all_predictions.groupby("model"):
        pred = group.groupby("ds", as_index=False)["predicted_price"].mean().sort_values("ds")
        ax.plot(pred["ds"], pred["predicted_price"], linewidth=1.8, marker="o", label=f"{model_name} mean forecast")
    ax.set_title("Weekly WTI fold-48 h=2 actual vs predicted price")
    ax.set_xlabel("Date")
    ax.set_ylabel("WTI price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_fold_forecast_paths_plot(
    all_predictions: pd.DataFrame,
    panel: pd.DataFrame,
    output_path: Path,
    lookback_weeks: int = 40,
) -> None:
    if all_predictions.empty:
        return

    model_names = list(all_predictions["model"].dropna().unique())
    n_models = max(len(model_names), 1)
    fig, axes = plt.subplots(n_models, 1, figsize=(13, 4.2 * n_models), squeeze=False, sharex=True)

    min_cutoff = pd.to_datetime(all_predictions["cutoff"]).min()
    max_forecast_date = pd.to_datetime(all_predictions["ds"]).max()
    plot_start = min_cutoff - pd.Timedelta(weeks=lookback_weeks)
    actual = (
        panel.loc[(panel.index >= plot_start) & (panel.index <= max_forecast_date), [TARGET_COL]]
        .reset_index()
        .rename(columns={"ds": "ds", TARGET_COL: "actual_price"})
    )

    for ax, model_name in zip(axes[:, 0], model_names):
        model_predictions = all_predictions[all_predictions["model"] == model_name].copy()
        ax.plot(
            actual["ds"],
            actual["actual_price"],
            color="black",
            linewidth=2.1,
            label="actual",
        )

        first_forecast = True
        for cutoff, group in model_predictions.sort_values(["cutoff", "ds"]).groupby("cutoff", sort=True):
            cutoff = pd.Timestamp(cutoff)
            if cutoff not in panel.index:
                continue
            path_dates = [cutoff, *pd.to_datetime(group["ds"]).tolist()]
            path_values = [float(panel.loc[cutoff, TARGET_COL]), *group["predicted_price"].astype(float).tolist()]
            ax.axvline(cutoff, color="tab:blue", linestyle=":", linewidth=0.7, alpha=0.22)
            ax.plot(
                path_dates,
                path_values,
                color="tab:blue",
                marker="o",
                linewidth=1.1,
                markersize=3.6,
                alpha=0.9,
                label=f"{model_name} forecast paths" if first_forecast else None,
            )
            first_forecast = False

        ax.set_title(f"{TARGET_COL} - {model_name} continual actual forecast {model_predictions['cutoff'].nunique()} folds")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper center", ncols=2)

    axes[-1, 0].set_xlabel("ds")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_loss_overview(loss_histories: Dict[str, pd.DataFrame], output_path: Path) -> None:
    n_models = max(len(loss_histories), 1)
    fig, axes = plt.subplots(n_models, 1, figsize=(11, 3.8 * n_models), squeeze=False)
    for ax, (model_name, history) in zip(axes[:, 0], loss_histories.items()):
        if "train_loss" in history.columns and history["train_loss"].notna().any():
            train = history[["epoch", "train_loss"]].dropna()
            base = train["train_loss"].iloc[0] if len(train) and train["train_loss"].iloc[0] != 0 else 1.0
            ax.plot(train["epoch"], train["train_loss"] / base, label="train_loss normalized", linewidth=1.5)
        if "valid_loss" in history.columns and history["valid_loss"].notna().any():
            valid = history[["epoch", "valid_loss"]].dropna()
            base = valid["valid_loss"].iloc[0] if len(valid) and valid["valid_loss"].iloc[0] != 0 else 1.0
            ax.plot(valid["epoch"], valid["valid_loss"] / base, label="valid_loss normalized", linewidth=1.8, marker="o")
        ax.set_title(f"{model_name} representative final-fit MSE loss")
        ax.set_xlabel("epoch/log step")
        ax.set_ylabel("normalized MSE loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    repo_root = args.repo_root.resolve()
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import newoil.pipeline as pipeline
    from newoil.pipeline import build_loss_csv_logger, load_csv_logger_loss_history, plot_loss_curves, save_loss_history

    pipeline.ensure_neuralforecast_runtime()
    NeuralForecast = pipeline.NeuralForecast

    accelerator, devices = detect_accelerator(args.devices)
    if accelerator == "gpu":
        torch.set_float32_matmul_precision(args.matmul_precision)
    panel, model_panel, exog_cols, metadata = weekly_panel(repo_root)
    nf_df = make_nf_df(model_panel, exog_cols)
    cv_train_length = max(
        len(nf_df) - args.val_size - args.horizon - ((args.n_windows - 1) * args.step_size),
        args.input_size + args.horizon,
    )
    loss_train_length = max(len(nf_df) - args.horizon, args.input_size + args.horizon)
    cv_step_plan = estimate_epoch_equivalent_max_steps(
        pipeline_module=pipeline,
        train_length=cv_train_length,
        input_size=args.input_size,
        horizon=args.horizon,
        max_epochs=args.max_epochs,
        max_steps_override=args.max_steps,
    )
    loss_step_plan = estimate_epoch_equivalent_max_steps(
        pipeline_module=pipeline,
        train_length=loss_train_length,
        input_size=args.input_size,
        horizon=args.horizon,
        max_epochs=args.max_epochs,
        max_steps_override=args.max_steps,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() / f"weekly_logret_h2_fold48_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        **metadata,
        "horizon": args.horizon,
        "n_windows": args.n_windows,
        "step_size": args.step_size,
        "val_size": args.val_size,
        "input_size": args.input_size,
        "max_epochs": args.max_epochs,
        "max_steps_override": args.max_steps,
        "cv_train_length_for_step_estimate": cv_train_length,
        "cv_step_plan": cv_step_plan,
        "loss_train_length_for_step_estimate": loss_train_length,
        "loss_step_plan": loss_step_plan,
        "models": args.models,
        "accelerator": accelerator,
        "devices": devices,
        "matmul_precision": args.matmul_precision if accelerator == "gpu" else "",
        "target_transform": "log-diff",
        "exog_transform": "none",
    }
    (output_root / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[TRAINING PLAN] "
        f"requested max_epochs={args.max_epochs}; "
        f"CV max_steps={cv_step_plan['max_steps']} "
        f"({cv_step_plan['estimated_steps_per_epoch']} steps/epoch); "
        f"loss-plot max_steps={loss_step_plan['max_steps']} "
        f"({loss_step_plan['estimated_steps_per_epoch']} steps/epoch)"
    )
    print(f"[DEVICE] accelerator={accelerator}, devices={devices}")

    all_predictions = []
    summary_rows = []
    loss_histories: Dict[str, pd.DataFrame] = {}

    for model_name in args.models:
        print(f"\n[CV] {model_name}")
        if model_name == "CustomITransformer":
            predictions, history, loss_dir = run_custom_itransformer(
                model_panel=model_panel,
                panel=panel,
                exog_cols=exog_cols,
                args=args,
                output_root=output_root,
                accelerator=accelerator,
                cv_step_plan=cv_step_plan,
            )
            predictions["model"] = model_name
            predictions_path = output_root / f"{model_name}_cv_predictions.csv"
            predictions.to_csv(predictions_path, index=False)
            all_predictions.append(predictions)
            loss_histories[model_name] = history
            plot_loss_curves(
                history,
                f"{model_name} target-log-return MSE loss",
                loss_dir / "loss_curve.png",
                x_column="epoch",
                x_label="Epoch",
                x_max=args.max_epochs,
            )
        else:
            model = build_model(
                model_name=model_name,
                exog_cols=exog_cols,
                accelerator=accelerator,
                devices=devices,
                h=args.horizon,
                input_size=args.input_size,
                max_steps=cv_step_plan["max_steps"],
                random_seed=args.seed,
                logger=False,
            )
            nf = NeuralForecast(models=[model], freq="W-MON")
            cv_df = nf.cross_validation(
                df=nf_df,
                val_size=args.val_size,
                n_windows=args.n_windows,
                step_size=args.step_size,
            )
            pred_col = find_prediction_column(cv_df, model_name)
            predictions = attach_price_predictions(cv_df, panel, pred_col)
            predictions["model"] = model_name
            predictions_path = output_root / f"{model_name}_cv_predictions.csv"
            predictions.to_csv(predictions_path, index=False)
            all_predictions.append(predictions)

        return_metrics = compute_metrics(predictions["actual_log_return"], predictions["pred_log_return"])
        price_metrics = compute_metrics(predictions["actual_price"], predictions["predicted_price"])
        summary_rows.append(
            {
                "model": model_name,
                "folds_requested": args.n_windows,
                "folds_evaluated": int(predictions["cutoff"].nunique()),
                "n_predictions": int(len(predictions)),
                "return_MSE": return_metrics["MSE"],
                "return_MAE": return_metrics["MAE"],
                "return_MAPE": return_metrics["MAPE"],
                "price_MSE": price_metrics["MSE"],
                "price_MAE": price_metrics["MAE"],
                "price_MAPE": price_metrics["MAPE"],
                "predictions_csv": str(predictions_path),
            }
        )

        if model_name == "CustomITransformer":
            continue

        print(f"[LOSS] {model_name} representative final fit")
        loss_dir = output_root / f"{model_name}_loss"
        loss_dir.mkdir(parents=True, exist_ok=True)
        loss_model = build_model(
            model_name=model_name,
            exog_cols=exog_cols,
            accelerator=accelerator,
            devices=devices,
            h=args.horizon,
            input_size=args.input_size,
            max_steps=loss_step_plan["max_steps"],
            random_seed=args.seed,
            logger=build_loss_csv_logger(loss_dir),
        )
        loss_nf = NeuralForecast(models=[loss_model], freq="W-MON")
        loss_nf.fit(df=nf_df.iloc[: -args.horizon].copy(), val_size=args.val_size)
        history = save_loss_history(loss_nf.models[0], loss_dir)
        if "loss_history_source" not in history.columns:
            csv_history = load_csv_logger_loss_history(loss_dir)
            if not csv_history.empty:
                history = csv_history
                history.to_csv(loss_dir / "loss_history.csv", index=False)
        loss_histories[model_name] = history
        plot_loss_curves(
            history,
            f"{model_name} final-fit MSE loss",
            loss_dir / "loss_curve.png",
            x_column="epoch" if "epoch" in history.columns else "step",
            x_label="Epoch/log step",
            x_max=args.max_epochs if "epoch" in history.columns else loss_step_plan["max_steps"],
        )

    predictions_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions_all.to_csv(output_root / "cv_predictions_all_models.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "fold48_metrics_summary.csv", index=False)

    prediction_plot = output_root / "actual_vs_predicted_price.png"
    save_prediction_plot(predictions_all, prediction_plot)
    fold_paths_plot = output_root / "fold_forecast_paths.png"
    save_fold_forecast_paths_plot(predictions_all, panel, fold_paths_plot)
    loss_plot = output_root / "loss_curves_normalized_overview.png"
    save_loss_overview(loss_histories, loss_plot)

    print(f"\nOutput root: {output_root}")
    print(f"Metrics summary: {output_root / 'fold48_metrics_summary.csv'}")
    print(f"Prediction graph: {prediction_plot}")
    print(f"Fold paths graph: {fold_paths_plot}")
    print(f"Loss graph: {loss_plot}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly WTI multivariate target-log-return h=2 fold-48 runner.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--n-windows", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--val-size", type=int, default=48)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap. Leave unset to run the epoch-equivalent step budget.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPU devices. Default is 1 to avoid DDP/NCCL failures on multi-GPU hosts.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="Float32 matmul precision for CUDA Tensor Cores.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
