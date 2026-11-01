"""Central configuration for the Market Basket Analysis project.

All paths, thresholds and tunable hyper-parameters live here so the data
pipeline, the analysis scripts and the Streamlit app share one source of
truth. Edit a value once and every downstream module picks it up.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
# Project root = parent of this file (config.py sits at the repo top level).
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
RULES_DIR = RESULTS_DIR / "rules"
DATASET_PATH = DATA_DIR / "dataset.csv"

# Make sure output folders exist when this module is imported anywhere.
for _d in (CHARTS_DIR, RULES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- data schema
# Columns produced by the synthetic generator (see data/README.md).
COLUMNS = [
    "transaction_id", "customer_id", "product_name", "category",
    "quantity", "price", "timestamp", "is_anomaly",
]
ID_COL = "transaction_id"
CUSTOMER_COL = "customer_id"
PRODUCT_COL = "product_name"
CATEGORY_COL = "category"
QTY_COL = "quantity"
PRICE_COL = "price"
TIME_COL = "timestamp"
ANOMALY_COL = "is_anomaly"

# --------------------------------------------------------------------------- algorithm knobs
# Association-rule mining thresholds. Tuned for ~1,300 baskets so the rule
# sets stay informative without exploding in size.
MIN_SUPPORT = 0.005          # itemset must appear in >= ~6 baskets
MIN_CONFIDENCE = 0.20        # P(consequent | antecedent) >= 20%
MIN_LIFT = 1.0               # anything below 1 means negative correlation
MAX_ITEMSET_LEN = 4          # cap itemset cardinality for speed
TOP_N_RULES = 20             # how many rules to surface in reports/charts

# Recommender
RECOMMEND_TOP_N = 5          # products returned per query product
