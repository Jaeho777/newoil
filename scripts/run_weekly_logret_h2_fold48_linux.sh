#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=""
OUTPUT_ROOT=""
INSTALL_TORCH="skip"
INSTALL_DEPS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --install-torch)
      INSTALL_TORCH="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS="true"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/run_weekly_logret_h2_fold48_linux.sh [options]

Options:
  --repo-root PATH       Repository root. Default: script parent directory.
  --output-root PATH     Output root. Default: <repo-root>/outputs.
  --install-torch MODE   skip, cpu, or cu128. Default: skip.
  --install-deps         Install/upgrade required Python packages.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$REPO_ROOT/outputs"
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[SETUP] Creating virtual environment at $REPO_ROOT/.venv"
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$REPO_ROOT/.venv"
  else
    python3 -m venv "$REPO_ROOT/.venv"
  fi
fi

"$VENV_PYTHON" -m pip install --upgrade pip

case "$INSTALL_TORCH" in
  cpu)
    echo "[SETUP] Installing CPU PyTorch"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ;;
  cu128)
    echo "[SETUP] Installing CUDA 12.8 PyTorch"
    "$VENV_PYTHON" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ;;
  skip)
    ;;
  *)
    echo "--install-torch must be one of: skip, cpu, cu128" >&2
    exit 2
    ;;
esac

if [[ "$INSTALL_DEPS" == "true" ]]; then
  echo "[SETUP] Installing project dependencies"
  "$VENV_PYTHON" -m pip install --upgrade \
    "numpy>=1.26,<2.2" \
    "pandas==2.2.2" \
    "protobuf>=4.25,<6" \
    "tensorboard>=2.18,<2.20" \
    "pyyaml==6.0.2" \
    "matplotlib>=3.8,<3.11" \
    "rich>=13,<15" \
    "utilsforecast" \
    "coreforecast" \
    "lightning-utilities>=0.11,<0.16" \
    "torchmetrics>=1.6,<1.9" \
    "pytorch-lightning>=2.4,<2.6" \
    "ray[tune]>=2.20,<3.0" \
    "neuralforecast==3.1.7"
fi

SANITY_CODE="import numpy, pandas, torch, pytorch_lightning, torchmetrics, ray, neuralforecast; print('import sanity ok')"
"$VENV_PYTHON" -c "$SANITY_CODE"

echo "[RUN] Weekly multivariate target-log-return h=2 fold48"
echo "[RUN] Repo root: $REPO_ROOT"
echo "[RUN] Output root: $OUTPUT_ROOT"
"$VENV_PYTHON" "$REPO_ROOT/scripts/run_weekly_logret_h2_fold48.py" --repo-root "$REPO_ROOT" --output-root "$OUTPUT_ROOT"
