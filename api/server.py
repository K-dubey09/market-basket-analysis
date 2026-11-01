"""Flask REST API for the Market Basket Analysis service.

Run with:  python -m api.server            (then open http://localhost:5000)
Or:        flask --app api.server run --port 5000

Endpoints
---------
GET /api/health                       -> {"status": "ok", ...}
GET /api/stats                        -> dataset headline stats
GET /api/products/top?n=20            -> top products by purchase frequency
GET /api/categories                   -> category distribution
GET /api/rules?algo=apriori&limit=20  -> top association rules
GET /api/recommend?product=<name>&n=5 -> cross-sell recommendations
GET /api/anomalies                    -> anomaly breakdown by fault type
GET /                                 -> HTML landing page with links

The dataset and rule sets are loaded once at startup (cached in module
globals) so every request is a fast dictionary lookup. For a production
deployment you'd add authentication, rate limiting and a real persistence
layer; this is intentionally lean for a college+HCL demo.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, abort

# Make the project root importable when run via `python -m api.server`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.data_loader import (
    load_dataset, split_anomalies, add_basket_features,
    build_transaction_matrix, quality_report,
)
from src import apriori_miner, fpgrowth_miner, recommender, eda

app = Flask(__name__)

# --------------------------------------------------------------------------- load once
print("[api] loading dataset + mining rules (one-time)...")
_raw = load_dataset()
_clean, _anom = split_anomalies(_raw)
_clean = add_basket_features(_clean)
_matrix = build_transaction_matrix(_clean)
_apriori_rules = apriori_miner.run_apriori(_matrix)
_fpgrowth_rules = fpgrowth_miner.run_fpgrowth(_matrix)
_stats = quality_report(_raw)
print(f"[api] ready: {_stats['transactions']} baskets, "
      f"{len(_apriori_rules)} Apriori rules, {len(_fpgrowth_rules)} FP-Growth rules")


# --------------------------------------------------------------------------- helpers
def _rules_for(algo: str):
    if algo.lower() == "fpgrowth":
        return _fpgrowth_rules
    return _apriori_rules  # default


# --------------------------------------------------------------------------- routes
@app.route("/")
def index():
    """Tiny HTML landing page so visiting the root URL isn't a 404."""
    return """
    <html><head><title>Market Basket Analysis API</title>
    <style>body{font-family:sans-serif;max-width:760px;margin:40px auto;line-height:1.5;padding:0 16px}
    code{background:#f4f4f4;padding:2px 6px;border-radius:3px} li{margin:4px 0}</style></head>
    <body><h1>🛒 Market Basket Analysis API</h1>
    <p>A Flask service exposing association rules and cross-sell recommendations
    as JSON. Try:</p>
    <ul>
      <li><code>GET /api/health</code></li>
      <li><code>GET /api/stats</code></li>
      <li><code>GET /api/products/top?n=10</code></li>
      <li><code>GET /api/categories</code></li>
      <li><code>GET /api/rules?algo=apriori&amp;limit=20</code></li>
      <li><code>GET /api/recommend?product=ValueMart%20Whole%20Milk&amp;n=5</code></li>
      <li><code>GET /api/anomalies</code></li>
    </ul>
    <p>College project in collaboration with <b>HCL Technologies</b>.</p>
    </body></html>
    """


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "market-basket-analysis",
        "transactions": _stats["transactions"],
        "rules": len(_apriori_rules),
    })


@app.route("/api/stats")
def stats():
    return jsonify(_stats)


@app.route("/api/products/top")
def top_products():
    n = request.args.get("n", default=20, type=int)
    n = max(1, min(n, 200))
    return jsonify(eda.top_products(_clean, n=n).to_dict(orient="records"))


@app.route("/api/categories")
def categories():
    return jsonify(eda.category_distribution(_clean).to_dict(orient="records"))


@app.route("/api/rules")
def rules():
    algo = request.args.get("algo", default="apriori")
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 500))
    df = _rules_for(algo).head(limit)
    return jsonify({
        "algo": algo,
        "count": int(len(df)),
        "rules": df.to_dict(orient="records"),
    })


@app.route("/api/recommend")
def recommend():
    product = request.args.get("product")
    n = request.args.get("n", default=5, type=int)
    if not product:
        abort(400, description="missing required 'product' query parameter")
    n = max(1, min(n, 20))
    recs = recommender.recommend(_apriori_rules, product, top_n=n)
    if recs.empty:
        return jsonify({"product": product, "recommendations": [],
                        "message": "no rules match this product at current thresholds"})
    return jsonify({
        "product": product,
        "count": int(len(recs)),
        "recommendations": recs.to_dict(orient="records"),
    })


@app.route("/api/anomalies")
def anomalies():
    return jsonify(eda.anomaly_breakdown(_raw).to_dict(orient="records"))


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
