# One-shot setup for Windows (PowerShell 5.1+ or PowerShell 7+).
# Creates a virtual environment, installs dependencies, generates the
# dataset, and runs the analysis pipeline so the app is ready to launch.
#
# Usage:  .\setup.ps1
# If PowerShell blocks scripts, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$py = 'python'
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) {
  Write-Host "ERROR: 'python' not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
  exit 1
}

Write-Host "==> Creating virtual environment (.venv)" -ForegroundColor Cyan
& $py -m venv .venv

# Activate
& .venv\Scripts\Activate.ps1

Write-Host "==> Upgrading pip" -ForegroundColor Cyan
pip install --upgrade pip

Write-Host "==> Installing dependencies (requirements.txt)" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "==> Generating dataset (data\dataset.csv, data\dataset_cleaned.csv)" -ForegroundColor Cyan
python -m scripts.generate_dataset

Write-Host "==> Running analysis pipeline (charts + rule tables)" -ForegroundColor Cyan
python -m scripts.run_pipeline

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup complete." -ForegroundColor Green
Write-Host " Run the dashboard with:" -ForegroundColor Green
Write-Host "   .\run.ps1" -ForegroundColor Green
Write-Host "   (or)  .venv\Scripts\Activate.ps1; streamlit run app\dashboard.py" -ForegroundColor Green
Write-Host " The dashboard opens at http://localhost:8501" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
