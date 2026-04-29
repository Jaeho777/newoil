param(
    [string]$RepoRoot = "",
    [string]$OutputRoot = "",
    [ValidateSet("skip", "cpu", "cu128")]
    [string]$InstallTorch = "skip",
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
} else {
    $RepoRoot = Resolve-Path $RepoRoot
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "outputs"
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[SETUP] Creating virtual environment at $RepoRoot\.venv"
    try {
        py -3.11 -m venv (Join-Path $RepoRoot ".venv")
    } catch {
        python -m venv (Join-Path $RepoRoot ".venv")
    }
}

& $VenvPython -m pip install --upgrade pip

if ($InstallTorch -eq "cpu") {
    Write-Host "[SETUP] Installing CPU PyTorch"
    & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
} elseif ($InstallTorch -eq "cu128") {
    Write-Host "[SETUP] Installing CUDA 12.8 PyTorch"
    & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

if ($InstallDeps) {
    Write-Host "[SETUP] Installing project dependencies"
    $ProjectDeps = @(
        "numpy>=1.26,<2.2",
        "pandas==2.2.2",
        "protobuf>=4.25,<6",
        "tensorboard>=2.18,<2.20",
        "pyyaml==6.0.2",
        "matplotlib>=3.8,<3.11",
        "rich>=13,<15",
        "utilsforecast",
        "coreforecast",
        "lightning-utilities>=0.11,<0.16",
        "torchmetrics>=1.6,<1.9",
        "pytorch-lightning>=2.4,<2.6",
        "ray[tune]>=2.20,<3.0",
        "neuralforecast==3.1.7"
    )
    & $VenvPython -m pip install --upgrade @ProjectDeps
}

Write-Host "[RUN] Weekly multivariate target-log-return h=2 fold48"
Write-Host "[RUN] Repo root: $RepoRoot"
Write-Host "[RUN] Output root: $OutputRoot"
& $VenvPython (Join-Path $RepoRoot "scripts\run_weekly_logret_h2_fold48.py") --repo-root $RepoRoot --output-root $OutputRoot
