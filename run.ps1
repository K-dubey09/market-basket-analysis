# Launch the Streamlit dashboard on Windows (PowerShell 5.1+ or 7+).
# Expects .\setup.ps1 to have been run once.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

if (-not (Test-Path .venv)) {
  Write-Host "ERROR: no .venv found. Run .\setup.ps1 first." -ForegroundColor Red
  exit 1
}
& .venv\Scripts\Activate.ps1

if (-not (Test-Path data\dataset.csv)) {
  Write-Host "==> dataset.csv missing — generating" -ForegroundColor Yellow
  python -m scripts.generate_dataset
}
if (-not (Test-Path results\charts\01_top_products.png)) {
  Write-Host "==> charts missing — running pipeline" -ForegroundColor Yellow
  python -m scripts.run_pipeline
}

Write-Host "==> Launching Streamlit dashboard at http://localhost:8501" -ForegroundColor Cyan
streamlit run app\dashboard.py --server.address=0.0.0.0 --server.port=8501
