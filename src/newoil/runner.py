from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_batch_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run overnight WTI experiment batch.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=False, default=None)
    args = parser.parse_args()

    result = run_batch_from_config(
        batch_config_path=args.batch_config,
        repo_root=args.repo_root,
        output_root=args.output_root,
    )
    print(f"Batch completed: {result.batch_dir}")
    print(f"Summary CSV: {result.batch_dir / 'summary.csv'}")
    print(f"HTML report: {result.report_html}")


if __name__ == "__main__":
    main()
