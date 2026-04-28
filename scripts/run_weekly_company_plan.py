from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import yaml

from newoil.pipeline import run_batch_from_config


DEFAULT_BATCH_CONFIGS = [
    "configs/batches/weekly_wti_h2_mse_report.yaml",
    "configs/batches/weekly_wti_h2_mse_scaled_regularized.yaml",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def detect_accelerator() -> Tuple[str, int]:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        return "gpu", -1 if device_count > 1 else 1
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def apply_runtime_overrides(batch_cfg: Dict[str, Any], accelerator: str, devices: int) -> Dict[str, Any]:
    cfg = deepcopy(batch_cfg)
    training_defaults = cfg.setdefault("training_defaults", {})
    trainer_kwargs = training_defaults.setdefault("trainer_kwargs", {})
    trainer_kwargs["accelerator"] = accelerator
    trainer_kwargs["devices"] = devices
    trainer_kwargs.setdefault("enable_checkpointing", False)
    trainer_kwargs.setdefault("enable_progress_bar", False)
    trainer_kwargs.setdefault("enable_model_summary", False)
    return cfg


def run_plan(repo_root: Path, batch_configs: List[Path], output_root: Path) -> Path:
    accelerator, devices = detect_accelerator()
    print(f"Detected accelerator: {accelerator} / devices={devices}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plan_root = output_root / f"weekly_company_plan_{timestamp}"
    plan_root.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    manifest_rows = []

    with TemporaryDirectory(prefix="weekly_company_plan_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for batch_path in batch_configs:
            source_path = batch_path if batch_path.is_absolute() else (repo_root / batch_path).resolve()
            batch_cfg = load_yaml(source_path)
            patched_cfg = apply_runtime_overrides(batch_cfg, accelerator=accelerator, devices=devices)
            patched_path = tmp_root / source_path.name
            dump_yaml(patched_path, patched_cfg)

            print(f"\n[RUN] {source_path.name}")
            result = run_batch_from_config(
                batch_config_path=patched_path,
                repo_root=repo_root,
                output_root=plan_root,
            )
            summary_df = result.summary_df.copy()
            summary_df["source_batch_config"] = str(source_path)
            summary_df["resolved_accelerator"] = accelerator
            summary_df["resolved_devices"] = devices
            combined_rows.append(summary_df)
            manifest_rows.append(
                {
                    "source_batch_config": str(source_path),
                    "patched_batch_config": str(patched_path),
                    "batch_dir": str(result.batch_dir),
                    "resolved_accelerator": accelerator,
                    "resolved_devices": devices,
                }
            )

    combined_df = pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    combined_csv = plan_root / "combined_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    pd.DataFrame(manifest_rows).to_csv(plan_root / "run_manifest.csv", index=False)

    meta = {
        "plan_root": str(plan_root),
        "combined_summary_csv": str(combined_csv),
        "resolved_accelerator": accelerator,
        "resolved_devices": devices,
        "batch_configs": [str(p if p.is_absolute() else (repo_root / p).resolve()) for p in batch_configs],
    }
    with (plan_root / "plan_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nPlan output root: {plan_root}")
    print(f"Combined summary: {combined_csv}")
    return plan_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the company-facing weekly two-run plan locally.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path.cwd() / "outputs")
    parser.add_argument(
        "--batch-config",
        action="append",
        dest="batch_configs",
        help="Batch config path to include. Pass multiple times. Defaults to the report + scaled_regularized pair.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    batch_configs = [Path(p) for p in args.batch_configs] if args.batch_configs else [Path(p) for p in DEFAULT_BATCH_CONFIGS]
    run_plan(repo_root=repo_root, batch_configs=batch_configs, output_root=args.output_root.resolve())


if __name__ == "__main__":
    main()
