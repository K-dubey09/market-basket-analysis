#!/usr/bin/env bash
# Launch the Streamlit dashboard on macOS and Linux.
# Expects ./setup.sh to have been run once.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "ERROR: no .venv found. Run ./setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f data/dataset.csv ]; then
  echo "==> dataset.csv missing — generating"
  python -m scripts.generate_dataset
fi
if [ ! -f results/charts/01_top_products.png ]; then
  echo "==> charts missing — running pipeline"
  python -m scripts.run_pipeline
fi

echo "==> Launching Streamlit dashboard at http://localhost:8501"
exec streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=8501
