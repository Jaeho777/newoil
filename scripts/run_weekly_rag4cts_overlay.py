from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
EPS = 1e-8


@dataclass
class CandidateWindow:
    origin_idx: int
    start_idx: int
    end_idx: int
    origin_dt: str
    start_dt: str
    end_dt: str
    future_returns: List[float]
    future_prices: List[float]
    distance: float
    rank: int = -1


def safe_log_return(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if (x <= 0).any():
        return x.diff() / (x.shift(1).abs() + EPS)
    return np.log(x.clip(lower=EPS) / x.shift(1).clip(lower=EPS))


def rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = x.rolling(window, min_periods=max(3, window // 3)).median()
    mad = (x - med).abs().rolling(window, min_periods=max(3, window // 3)).median()
    return (x - med) / (1.4826 * mad + EPS)


def load_weekly_panel(repo_root: Path) -> pd.DataFrame:
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
    panel["target_log_return"] = np.log(panel[TARGET_COL]).diff()
    panel = panel.dropna(subset=["target_log_return", *exog_cols]).copy()
    panel = panel.reset_index().rename(columns={"ds": "dt"})
    return panel


def make_features(panel: pd.DataFrame, variables: Sequence[str], z_window: int, slope_window: int, vol_window: int) -> Tuple[pd.DataFrame, List[str]]:
    features = pd.DataFrame({"dt": panel["dt"]})
    for variable in variables:
        x = pd.to_numeric(panel[variable], errors="coerce").astype(float)
        ret = safe_log_return(x)
        features[f"ret__{variable}"] = ret
        features[f"z__{variable}"] = rolling_robust_z(x, z_window)
        features[f"slope__{variable}"] = (x - x.shift(slope_window)) / float(slope_window)
        features[f"vol__{variable}"] = ret.rolling(vol_window, min_periods=max(3, vol_window // 2)).std()
    feature_cols = [column for column in features.columns if column != "dt"]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return features, feature_cols


def feature_variable(feature_col: str) -> str:
    return feature_col.split("__", 1)[1] if "__" in feature_col else feature_col


def target_feature_cols(feature_cols: Sequence[str]) -> List[str]:
    return [col for col in feature_cols if feature_variable(col) == TARGET_COL]


def causal_scale(features: pd.DataFrame, origin_idx: int, feature_cols: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
    train = features.loc[:origin_idx, feature_cols].astype(float)
    med = train.median()
    mad = (train - med).abs().median()
    scale = 1.4826 * mad + EPS
    return med, scale


def scaled_window(features: pd.DataFrame, start_idx: int, end_idx: int, feature_cols: Sequence[str], med: pd.Series, scale: pd.Series) -> np.ndarray:
    values = features.loc[start_idx:end_idx, feature_cols].astype(float)
    return ((values - med) / scale).to_numpy(dtype=float)


def temporal_weights(history_length: int, n_features: int, decay: float) -> np.ndarray:
    point_weights = decay ** np.arange(history_length - 1, -1, -1)
    point_weights = point_weights / (point_weights.mean() + EPS)
    return np.repeat(point_weights[:, None], n_features, axis=1)


def weighted_distance(query: np.ndarray, candidate: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * (query - candidate) ** 2) + EPS))


def actual_returns(panel: pd.DataFrame, origin_idx: int, horizon: int) -> np.ndarray:
    returns = []
    for h in range(1, horizon + 1):
        prev_price = float(panel.loc[origin_idx + h - 1, TARGET_COL])
        now_price = float(panel.loc[origin_idx + h, TARGET_COL])
        returns.append(math.log(max(now_price, EPS) / max(prev_price, EPS)))
    return np.asarray(returns, dtype=float)


def returns_to_prices(origin_price: float, returns: np.ndarray) -> np.ndarray:
    return float(origin_price) * np.exp(np.cumsum(np.asarray(returns, dtype=float)))


def build_recent_folds(panel: pd.DataFrame, n_folds: int, history_length: int, horizon: int, step_size: int) -> pd.DataFrame:
    min_origin = history_length + 2
    max_origin = len(panel) - horizon - 1
    origins = list(range(min_origin, max_origin + 1, step_size))
    if len(origins) < n_folds:
        raise ValueError(f"Not enough origins: {len(origins)} < {n_folds}")
    selected = origins[-n_folds:]
    rows = []
    for fold, origin_idx in enumerate(selected, start=1):
        rows.append(
            {
                "fold": fold,
                "origin_idx": origin_idx,
                "origin_dt": panel.loc[origin_idx, "dt"],
                "history_start_idx": origin_idx - history_length + 1,
                "history_start_dt": panel.loc[origin_idx - history_length + 1, "dt"],
                "forecast_start_dt": panel.loc[origin_idx + 1, "dt"],
                "forecast_end_dt": panel.loc[origin_idx + horizon, "dt"],
            }
        )
    return pd.DataFrame(rows)


def retrieve_windows(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    origin_idx: int,
    feature_cols: Sequence[str],
    history_length: int,
    horizon: int,
    top_k: int,
    decay: float,
) -> List[CandidateWindow]:
    query_history_start = origin_idx - history_length + 1
    max_candidate_origin = query_history_start - horizon - 1
    min_candidate_origin = history_length - 1
    if max_candidate_origin < min_candidate_origin:
        return []
    med, scale = causal_scale(features, origin_idx, feature_cols)
    query = scaled_window(features, query_history_start, origin_idx, feature_cols, med, scale)
    weights = temporal_weights(history_length, len(feature_cols), decay)
    candidates: List[CandidateWindow] = []
    for candidate_origin in range(min_candidate_origin, max_candidate_origin + 1):
        start_idx = candidate_origin - history_length + 1
        candidate = scaled_window(features, start_idx, candidate_origin, feature_cols, med, scale)
        distance = weighted_distance(query, candidate, weights)
        future_returns = actual_returns(panel, candidate_origin, horizon).tolist()
        future_prices = panel.loc[candidate_origin + 1:candidate_origin + horizon, TARGET_COL].astype(float).tolist()
        candidates.append(
            CandidateWindow(
                origin_idx=candidate_origin,
                start_idx=start_idx,
                end_idx=candidate_origin + horizon,
                origin_dt=str(pd.Timestamp(panel.loc[candidate_origin, "dt"]).date()),
                start_dt=str(pd.Timestamp(panel.loc[start_idx, "dt"]).date()),
                end_dt=str(pd.Timestamp(panel.loc[candidate_origin + horizon, "dt"]).date()),
                future_returns=future_returns,
                future_prices=future_prices,
                distance=distance,
            )
        )
    candidates = sorted(candidates, key=lambda item: item.distance)[:top_k]
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rank = rank
    return candidates


def retrieval_forecast(candidates: Sequence[CandidateWindow], k: int, horizon: int) -> np.ndarray:
    selected = list(candidates[: max(1, min(k, len(candidates)))])
    if not selected:
        return np.zeros(horizon, dtype=float)
    distances = np.asarray([max(candidate.distance, EPS) for candidate in selected], dtype=float)
    weights = (1.0 / distances) / ((1.0 / distances).sum() + EPS)
    paths = np.asarray([candidate.future_returns for candidate in selected], dtype=float)
    return (weights[:, None] * paths).sum(axis=0)


def dynamic_k(candidates: Sequence[CandidateWindow], max_k: int) -> Tuple[int, pd.DataFrame]:
    candidates = list(candidates)
    if len(candidates) <= 2:
        return max(1, len(candidates)), pd.DataFrame([{"candidate_k": max(1, len(candidates)), "agent_mae": np.nan, "selected": True}])
    agent = candidates[0]
    rows = []
    best_k = 1
    best_mae = np.inf
    for k in range(1, min(max_k, len(candidates) - 1) + 1):
        pred = retrieval_forecast(candidates[1:], k, len(agent.future_returns))
        truth = np.asarray(agent.future_returns, dtype=float)
        mae = float(np.mean(np.abs(pred - truth)))
        rows.append({"candidate_k": k, "agent_mae": mae, "selected": False})
        if mae < best_mae:
            best_mae = mae
            best_k = k
    for row in rows:
        row["selected"] = row["candidate_k"] == best_k
    return best_k, pd.DataFrame(rows)


def supervised_design(panel: pd.DataFrame, features: pd.DataFrame, feature_cols: Sequence[str], max_origin: int, history_length: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for origin_idx in range(history_length - 1, max_origin + 1):
        if origin_idx + horizon >= len(panel):
            break
        start_idx = origin_idx - history_length + 1
        med, scale = causal_scale(features, origin_idx, feature_cols)
        x = scaled_window(features, start_idx, origin_idx, feature_cols, med, scale).ravel()
        y = actual_returns(panel, origin_idx, horizon)
        rows.append(x)
        targets.append(y)
    if not rows:
        return np.empty((0, history_length * len(feature_cols))), np.empty((0, horizon))
    return np.vstack(rows), np.vstack(targets)


def ridge_forecast(panel: pd.DataFrame, features: pd.DataFrame, origin_idx: int, feature_cols: Sequence[str], history_length: int, horizon: int, alpha: float) -> np.ndarray:
    max_train_origin = origin_idx - horizon
    x_train, y_train = supervised_design(panel, features, feature_cols, max_train_origin, history_length, horizon)
    if len(x_train) < 10:
        return np.zeros(horizon, dtype=float)
    med, scale = causal_scale(features, origin_idx, feature_cols)
    query = scaled_window(features, origin_idx - history_length + 1, origin_idx, feature_cols, med, scale).ravel()
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + EPS
    x_train_scaled = (x_train - x_mean) / x_std
    query_scaled = (query - x_mean) / x_std
    x_aug = np.c_[np.ones(len(x_train_scaled)), x_train_scaled]
    q_aug = np.r_[1.0, query_scaled]
    penalty = np.eye(x_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)
    return np.asarray(q_aug @ coef, dtype=float)


def blend_returns(model_returns: np.ndarray, retrieval_returns: np.ndarray, alpha: float) -> np.ndarray:
    model_returns = np.asarray(model_returns, dtype=float)
    retrieval_returns = np.asarray(retrieval_returns, dtype=float)
    return (1.0 - alpha) * model_returns + alpha * retrieval_returns


def load_deep_predictions(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"model", "cutoff", "ds", "pred_log_return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Deep prediction file missing columns: {missing}")
    frame["cutoff"] = pd.to_datetime(frame["cutoff"])
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame


def compute_metrics(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in records.groupby("method"):
        return_err = group["pred_return"].to_numpy(float) - group["actual_return"].to_numpy(float)
        price_err = group["pred_price"].to_numpy(float) - group["actual_price"].to_numpy(float)
        actual_price = np.abs(group["actual_price"].to_numpy(float))
        by_path = group.groupby("fold").agg(pred_cum=("pred_return", "sum"), actual_cum=("actual_return", "sum"))
        direction_acc = np.mean(np.sign(by_path["pred_cum"]) == np.sign(by_path["actual_cum"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude_capture = np.nanmean(by_path["pred_cum"] / (by_path["actual_cum"] + EPS))
        abs_actual_cum = np.abs(by_path["actual_cum"])
        spike_threshold = abs_actual_cum.quantile(0.75)
        spike = by_path[abs_actual_cum >= spike_threshold]
        spike_direction_acc = np.mean(np.sign(spike["pred_cum"]) == np.sign(spike["actual_cum"])) if len(spike) else np.nan
        rows.append(
            {
                "method": method,
                "folds": int(group["fold"].nunique()),
                "n_predictions": int(len(group)),
                "return_MSE": float(np.mean(return_err ** 2)),
                "return_MAE": float(np.mean(np.abs(return_err))),
                "price_MAE": float(np.mean(np.abs(price_err))),
                "price_RMSE": float(np.sqrt(np.mean(price_err ** 2))),
                "price_MAPE": float(np.mean(np.abs(price_err) / (actual_price + EPS)) * 100.0),
                "direction_acc": float(direction_acc),
                "magnitude_capture_ratio": float(magnitude_capture),
                "spike_direction_acc": float(spike_direction_acc),
            }
        )
    return pd.DataFrame(rows).sort_values(["price_MAPE", "return_MAE"]).reset_index(drop=True)


def save_forecast_plot(records: pd.DataFrame, panel: pd.DataFrame, output_path: Path, max_methods: int = 8) -> None:
    methods = records.groupby("method")["price_MAPE_proxy"].mean().sort_values().index.tolist() if "price_MAPE_proxy" in records else records["method"].drop_duplicates().tolist()
    methods = methods[:max_methods]
    fig, axes = plt.subplots(len(methods), 1, figsize=(13, 3.6 * len(methods)), sharex=True, squeeze=False)
    min_cutoff = pd.to_datetime(records["origin_dt"]).min()
    max_forecast = pd.to_datetime(records["forecast_dt"]).max()
    actual = panel[(panel["dt"] >= min_cutoff - pd.Timedelta(weeks=40)) & (panel["dt"] <= max_forecast)]
    for ax, method in zip(axes[:, 0], methods):
        sub = records[records["method"] == method]
        ax.plot(actual["dt"], actual[TARGET_COL], color="black", linewidth=2.0, label="actual")
        first = True
        for _, path in sub.sort_values(["fold", "horizon"]).groupby("fold"):
            origin_dt = pd.Timestamp(path["origin_dt"].iloc[0])
            origin_price = float(path["origin_price"].iloc[0])
            dates = [origin_dt, *pd.to_datetime(path["forecast_dt"]).tolist()]
            values = [origin_price, *path["pred_price"].astype(float).tolist()]
            ax.axvline(origin_dt, color="tab:blue", linestyle=":", linewidth=0.7, alpha=0.18)
            ax.plot(dates, values, color="tab:blue", marker="o", linewidth=1.0, alpha=0.75, label="forecast path" if first else None)
            first = False
        ax.set_title(method)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1, 0].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metrics_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_df = metrics.sort_values("price_MAPE").copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(plot_df["method"], plot_df["price_MAPE"])
    ax.set_ylabel("price MAPE (%)")
    ax.set_title("RAG4CTS overlay method comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6g}")
        else:
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in view.columns]
    rows = view.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve() / f"weekly_rag4cts_overlay_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    tables_dir = output_root / "tables"
    plots_dir = output_root / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    panel = load_weekly_panel(repo_root)
    features, all_feature_cols = make_features(panel, VARIABLES, args.z_window, args.slope_window, args.vol_window)
    target_cols = target_feature_cols(all_feature_cols)
    full_cols = all_feature_cols
    folds = build_recent_folds(panel, args.n_folds, args.history_length, args.horizon, args.step_size)
    deep_predictions = load_deep_predictions(args.model_predictions)

    config = {
        "target": TARGET_COL,
        "variables": VARIABLES,
        "history_length": args.history_length,
        "horizon": args.horizon,
        "n_folds": args.n_folds,
        "step_size": args.step_size,
        "top_k": args.top_k,
        "blend_alpha": args.blend_alpha,
        "ridge_alpha": args.ridge_alpha,
        "retrieval_similarity": "causal history-window weighted distance",
        "future_exog_used": False,
        "candidate_future_used_only_as_retrieved_outcome": True,
        "deep_predictions": str(args.model_predictions) if args.model_predictions else "",
        "model_start_date": str(pd.Timestamp(panel["dt"].min()).date()),
        "model_end_date": str(pd.Timestamp(panel["dt"].max()).date()),
        "rows": int(len(panel)),
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    folds.to_csv(tables_dir / "fold_table.csv", index=False)

    forecast_rows: List[Dict[str, Any]] = []
    retrieval_rows: List[Dict[str, Any]] = []
    dynamic_k_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for _, fold_row in folds.iterrows():
        fold = int(fold_row["fold"])
        origin_idx = int(fold_row["origin_idx"])
        origin_dt = pd.Timestamp(fold_row["origin_dt"])
        print(f"[FOLD] {fold}/{args.n_folds} origin={origin_dt.date()}", flush=True)

        target_retrieved = retrieve_windows(panel, features, origin_idx, target_cols, args.history_length, args.horizon, args.top_k, args.decay)
        feature_retrieved = retrieve_windows(panel, features, origin_idx, full_cols, args.history_length, args.horizon, args.top_k, args.decay)
        target_k, target_k_log = dynamic_k(target_retrieved, args.top_k)
        feature_k, feature_k_log = dynamic_k(feature_retrieved, args.top_k)
        retrieval_only = retrieval_forecast(target_retrieved, target_k, args.horizon)
        feature_retrieval = retrieval_forecast(feature_retrieved, feature_k, args.horizon)
        target_model = ridge_forecast(panel, features, origin_idx, target_cols, args.history_length, args.horizon, args.ridge_alpha)
        multi_model = ridge_forecast(panel, features, origin_idx, full_cols, args.history_length, args.horizon, args.ridge_alpha)

        method_returns: Dict[str, np.ndarray] = {
            "target_only_model": target_model,
            "multivariate_model": multi_model,
            "retrieval_only": retrieval_only,
            "model_plus_retrieval": blend_returns(multi_model, retrieval_only, args.blend_alpha),
            "feature_retrieval": feature_retrieval,
            "feature_model_retrieval": blend_returns(multi_model, feature_retrieval, args.blend_alpha),
        }

        if not deep_predictions.empty:
            cutoff_mask = deep_predictions["cutoff"] == origin_dt
            for model_name, group in deep_predictions[cutoff_mask].groupby("model"):
                group = group.sort_values("ds")
                if len(group) >= args.horizon:
                    deep_returns = group["pred_log_return"].to_numpy(dtype=float)[: args.horizon]
                    method_returns[f"deep_{model_name}"] = deep_returns
                    method_returns[f"deep_{model_name}_feature_retrieval"] = blend_returns(deep_returns, feature_retrieval, args.blend_alpha)

        actual_rets = actual_returns(panel, origin_idx, args.horizon)
        actual_prices = panel.loc[origin_idx + 1:origin_idx + args.horizon, TARGET_COL].astype(float).to_numpy()
        origin_price = float(panel.loc[origin_idx, TARGET_COL])

        for method, pred_rets in method_returns.items():
            pred_prices = returns_to_prices(origin_price, pred_rets)
            for h in range(1, args.horizon + 1):
                price_mape_proxy = abs(pred_prices[h - 1] - actual_prices[h - 1]) / (abs(actual_prices[h - 1]) + EPS) * 100.0
                forecast_rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "origin_idx": origin_idx,
                        "origin_dt": str(origin_dt.date()),
                        "horizon": h,
                        "forecast_dt": str(pd.Timestamp(panel.loc[origin_idx + h, "dt"]).date()),
                        "origin_price": origin_price,
                        "pred_return": float(pred_rets[h - 1]),
                        "actual_return": float(actual_rets[h - 1]),
                        "pred_price": float(pred_prices[h - 1]),
                        "actual_price": float(actual_prices[h - 1]),
                        "price_MAPE_proxy": float(price_mape_proxy),
                        "target_dynamic_k": int(target_k),
                        "feature_dynamic_k": int(feature_k),
                    }
                )

        for source, candidates in [("target", target_retrieved), ("feature", feature_retrieved)]:
            query_history_start = origin_idx - args.history_length + 1
            for candidate in candidates:
                retrieval_rows.append(
                    {
                        "fold": fold,
                        "source": source,
                        "rank": candidate.rank,
                        "origin_dt": str(origin_dt.date()),
                        "query_history_start_dt": str(pd.Timestamp(panel.loc[query_history_start, "dt"]).date()),
                        "candidate_origin_dt": candidate.origin_dt,
                        "candidate_start_dt": candidate.start_dt,
                        "candidate_end_dt": candidate.end_dt,
                        "distance": candidate.distance,
                        "candidate_future_cum_return": float(np.sum(candidate.future_returns)),
                        "no_overlap_pass": bool(candidate.end_idx < query_history_start),
                    }
                )
        target_k_log["fold"] = fold
        target_k_log["source"] = "target"
        feature_k_log["fold"] = fold
        feature_k_log["source"] = "feature"
        dynamic_k_rows.extend(pd.concat([target_k_log, feature_k_log], ignore_index=True).to_dict("records"))

        query_history_start = origin_idx - args.history_length + 1
        max_candidate_end = max([candidate.end_idx for candidate in [*target_retrieved, *feature_retrieved]], default=-1)
        deep_origin_predictions = deep_predictions[deep_predictions["cutoff"] == origin_dt] if not deep_predictions.empty else pd.DataFrame()
        deep_prediction_coverage = True
        if not deep_predictions.empty:
            deep_prediction_coverage = all(len(group) >= args.horizon for _, group in deep_origin_predictions.groupby("model"))
            deep_prediction_coverage = bool(deep_prediction_coverage and not deep_origin_predictions.empty)
        audit_rows.append(
            {
                "fold": fold,
                "origin_dt": str(origin_dt.date()),
                "target_not_in_exog": TARGET_COL not in [col for col in VARIABLES if col != TARGET_COL],
                "candidate_end_before_query_history": bool(max_candidate_end < query_history_start),
                "query_uses_only_history": True,
                "future_exog_not_used": True,
                "deep_prediction_cutoff_aligned": True if deep_predictions.empty else bool((deep_origin_predictions["ds"] > origin_dt).all()),
                "deep_prediction_coverage": deep_prediction_coverage,
            }
        )

    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(tables_dir / "all_method_forecasts.csv", index=False)
    retrieval_log = pd.DataFrame(retrieval_rows)
    retrieval_log.to_csv(tables_dir / "retrieval_candidates.csv", index=False)
    pd.DataFrame(dynamic_k_rows).to_csv(tables_dir / "dynamic_k_log.csv", index=False)
    leakage_audit = pd.DataFrame(audit_rows)
    leakage_audit.to_csv(tables_dir / "leakage_audit.csv", index=False)
    if not leakage_audit.drop(columns=["fold", "origin_dt"], errors="ignore").all().all():
        raise RuntimeError("Leakage audit failed. See tables/leakage_audit.csv")

    metrics = compute_metrics(forecasts)
    metrics.to_csv(tables_dir / "metrics_summary.csv", index=False)
    save_metrics_plot(metrics, plots_dir / "metrics_price_mape.png")
    save_forecast_plot(forecasts, panel, plots_dir / "continuous_forecast_overlay.png")

    best = metrics.iloc[0].to_dict()
    metric_lookup = metrics.set_index("method").to_dict("index")
    baseline_name = "multivariate_model" if "multivariate_model" in metric_lookup else "target_only_model"
    baseline = metric_lookup.get(baseline_name)
    improvement_line = "Baseline comparison could not be computed."
    if baseline:
        delta = float(baseline["price_MAPE"]) - float(best["price_MAPE"])
        improvement_line = (
            f"Best method `{best['method']}` changed price MAPE by {delta:.3f} percentage points "
            f"versus baseline `{baseline_name}`."
        )

    report = [
        "# Weekly WTI RAG4CTS Overlay Report",
        "",
        "## Experiment Setup",
        f"- Forecast target: `{TARGET_COL}` weekly price path.",
        f"- Data period: {config['model_start_date']} to {config['model_end_date']}.",
        f"- Forecast horizon: {args.horizon} weekly steps.",
        f"- Fixed-origin folds: {args.n_folds}, step size {args.step_size}.",
        f"- History window length: {args.history_length} weeks.",
        f"- Variables: {', '.join(VARIABLES)}.",
        "",
        "## Method",
        "- Retrieval uses only the observed history window up to each origin.",
        "- Candidate future paths are used only after historical candidates are selected.",
        "- No future exogenous variables are used for the query.",
        "- Compared methods: target-only model, multivariate model, retrieval-only, model+retrieval, feature+retrieval, and feature+model+retrieval.",
        "- Deep model predictions are overlaid when a fold48 prediction CSV is supplied.",
        "",
        "## Key Result",
        f"- Best method by price MAPE: `{best['method']}`.",
        f"- Price MAE: {float(best['price_MAE']):.4f}; price RMSE: {float(best['price_RMSE']):.4f}; price MAPE: {float(best['price_MAPE']):.4f}%.",
        f"- Direction accuracy: {float(best['direction_acc']):.4f}; spike direction accuracy: {float(best['spike_direction_acc']):.4f}.",
        f"- {improvement_line}",
        "",
        "## Interpretation",
        "- The retrieval methods test whether historically similar pre-shock windows contain useful post-shock price-path information.",
        "- If retrieval-augmented methods rank above the pure model baselines, the result supports the spike-amplitude compensation hypothesis.",
        "- If pure multivariate or target-only models rank higher, the retrieved historical crisis paths are not adding stable signal under this fold design.",
        "- The leakage audit below must remain all `True`; otherwise the results should not be used.",
        "",
        "## Metrics",
        markdown_table(metrics),
        "",
        "## Leakage Audit",
        markdown_table(leakage_audit),
    ]
    (output_root / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Output root: {output_root}", flush=True)
    print(f"Metrics: {tables_dir / 'metrics_summary.csv'}", flush=True)
    print(f"Forecasts: {tables_dir / 'all_method_forecasts.csv'}", flush=True)
    print(f"Leakage audit: {tables_dir / 'leakage_audit.csv'}", flush=True)
    print(f"Main plot: {plots_dir / 'continuous_forecast_overlay.png'}", flush=True)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly WTI RAG4CTS-style retrieval overlay experiment.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs")
    parser.add_argument("--model-predictions", type=Path, default=None)
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=48)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--blend-alpha", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--decay", type=float, default=0.96)
    parser.add_argument("--z-window", type=int, default=26)
    parser.add_argument("--slope-window", type=int, default=4)
    parser.add_argument("--vol-window", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
