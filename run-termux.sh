#!/data/data/com.termux/files/usr/bin/env bash
# Launch the Streamlit dashboard on Android (Termux).
# Expects bash setup-termux.sh to have been run once.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "ERROR: no .venv found. Run bash setup-termux.sh first." >&2
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
echo "==> Open it in Chrome with:  termux-open-url http://localhost:8501"
exec streamlit run app/dashboard.py --server.address=127.0.0.1 --server.port=8501
