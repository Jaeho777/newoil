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
    & $VenvPython -m pip install neuralforecast pandas numpy matplotlib pyyaml rich
}

Write-Host "[RUN] Repo root: $RepoRoot"
Write-Host "[RUN] Output root: $OutputRoot"
& $VenvPython (Join-Path $RepoRoot "scripts\run_weekly_company_plan.py") --repo-root $RepoRoot --output-root $OutputRoot
