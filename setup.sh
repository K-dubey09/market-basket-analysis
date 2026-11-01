#!/usr/bin/env bash
# One-shot setup for macOS and Linux.
# Creates a virtual environment, installs dependencies, generates the
# dataset, and runs the analysis pipeline so the app is ready to launch.
#
# Usage:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: '$PY' not found. Install Python 3.10+ from https://python.org" >&2
  exit 1
fi

echo "==> Creating virtual environment (.venv)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies (requirements.txt)"
pip install -r requirements.txt

echo "==> Generating dataset (data/dataset.csv, data/dataset_cleaned.csv)"
python -m scripts.generate_dataset

echo "==> Running analysis pipeline (charts + rule tables)"
python -m scripts.run_pipeline

cat <<EOF

============================================================
 Setup complete.
 Run the dashboard with:
   ./run.sh
   (or)  source .venv/bin/activate && streamlit run app/dashboard.py
 The dashboard opens at http://localhost:8501
============================================================
EOF
