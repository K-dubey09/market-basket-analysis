#!/data/data/com.termux/files/usr/bin/env bash
# Android setup via Termux (https://termux.dev).
#
# Install Termux from F-Droid (NOT Google Play — the Play Store build is
# outdated and broken). Open Termux and run this script from the project
# root after copying the project folder into Termux's home:
#
#   pkg install git -y
#   git clone <your-repo-url> ~/market_basket_analysis     # or copy the folder
#   cd ~/market_basket_analysis
#   bash setup-termux.sh
#
# HONEST LIMITS on Android:
#   - Streamlit and matplotlib work on Termux but build steps are slow.
#   - The dashboard runs as a *local* web server; open it in Chrome via
#     `termux-open-url http://localhost:8501` (or just visit the URL).
#   - This is NOT a native Android app — there is no APK. It is the same
#     Streamlit web app you run on desktop, served from the phone.
#   - For a smoother mobile experience, deploy the app once (see the
#     Deployment Guide PDF) and open the public URL on the phone instead.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Updating Termux packages"
pkg update -y

echo "==> Installing Python and build toolchain (clang, make, binutils)"
pkg install -y python clang make binutils termux-tools

echo "==> Creating virtual environment (.venv)"
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

echo "==> Installing dependencies (this can take several minutes on Android)"
# numpy / pandas / matplotlib wheels exist for Android-arm64; if any wheel
# build fails, install the Termux-packaged equivalents first:
#   pkg install python-numpy python-pillow
pip install -r requirements.txt

echo "==> Generating dataset"
python -m scripts.generate_dataset

echo "==> Running analysis pipeline"
python -m scripts.run_pipeline

cat <<EOF

============================================================
 Setup complete on Android.
 Run the dashboard with:
   bash run-termux.sh
 The dashboard opens at http://localhost:8501
 Open it in Chrome with:  termux-open-url http://localhost:8501
============================================================
EOF
