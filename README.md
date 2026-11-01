# 🛒 Market Basket Analysis for Retail Upselling

An end-to-end **market-basket analysis** project that turns retail transaction
data into actionable cross-sell recommendations. Mined with **Apriori** and
**FP-Growth**, served through an interactive **Streamlit** dashboard, and
fully reproducible from a single pipeline command.

> College project in collaboration with **HCL** — Business Analytics / Python.

---

## ✨ Features

| Area | What you get |
|------|--------------|
| **Data pipeline** | Typed loader, anomaly split, transaction-matrix builder — one source of truth in `config.py` |
| **EDA** | 8 charts: top products, revenue, category mix, basket size, daily/hourly patterns, category co-occurrence heatmap, anomaly breakdown |
| **Association rules** | Apriori **and** FP-Growth side-by-side, sorted by lift, with support/confidence/lift/leverage/conviction |
| **Recommender** | "Customers who bought X also bought Y" engine ranked by confidence and lift |
| **Dashboard** | 6-page Streamlit app: Upload Data · Overview · EDA · Rules · Recommender · Data Quality |
| **Upload your data** | Bring your own XLSX/CSV on the Upload page — Append to or Replace the bundled dataset; rules and charts re-run live |
| **REST API** | Flask service exposing rules and recommendations as JSON — see "REST API" below |
| **Reproducibility** | One command (`python -m scripts.run_pipeline`) regenerates every chart and CSV |

---

## 📤 Upload your own data (XLSX or CSV)

The dashboard ships with a 5,000-row synthetic retail dataset, but you can
bring your own. Open the dashboard, go to the **📤 Upload Data** page, and:

1. Pick a **merge mode** — *Append to existing dataset* (add your rows to the
   bundled data) or *Replace existing dataset* (use only your rows).
2. Choose a `.csv` or `.xlsx` file. The only **required** columns are
   `transaction_id` and `product_name`; the rest are filled with sensible
   defaults if missing. Column-name matching is case-insensitive.
3. Preview the parsed rows, then click **Confirm — save and re-run analysis**.

The merged dataset is saved to `data/dataset_uploaded.csv` (so it survives a
Streamlit restart), and every downstream page — Overview, EDA, Rules,
Recommender, Data Quality — re-runs on the active data immediately. The
sidebar shows whether the active dataset is the bundled one or your upload.
A **Reset to the bundled dataset** button on the same page deletes the
uploaded file and reverts.

Expected schema (one row per basket line item):

| Column          | Required | Type      | Default if missing |
|-----------------|----------|-----------|--------------------|
| transaction_id  | ✓        | string    | —                  |
| customer_id     |          | string    | blank              |
| product_name    | ✓        | string    | —                  |
| category        |          | string    | "Uncategorised"    |
| quantity        |          | number    | 1                  |
| price           |          | number    | 0.0                |
| timestamp       | ✓        | datetime  | now                |
| is_anomaly      |          | int (0/1) | 0                  |

---

## 📁 Project layout

```
market_basket_analysis/
├── config.py                  # paths, schema, algorithm knobs
├── requirements.txt
├── README.md                  # this file
├── setup.sh / run.sh          # one-shot setup + launch (macOS / Linux)
├── setup.ps1 / run.ps1        # one-shot setup + launch (Windows)
├── setup-termux.sh / run-termux.sh  # Android via Termux
├── data/
│   ├── dataset.csv            # 5,000-row retail basket dataset
│   └── README.md              # column dictionary + generation notes
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # load, clean, basket features, one-hot matrix
│   ├── eda.py                 # EDA aggregation functions
│   ├── apriori_miner.py       # Apriori wrapper (mlxtend)
│   ├── fpgrowth_miner.py      # FP-Growth wrapper (mlxtend)
│   ├── recommender.py         # cross-sell recommender
│   └── visualizer.py          # matplotlib chart rendering
├── app/
│   └── dashboard.py           # Streamlit 5-page dashboard
├── api/
│   └── server.py             # Flask REST API (rules + recommendations as JSON)
├── notebooks/
│   └── analysis.ipynb         # narrative walk-through of the pipeline
├── scripts/
│   └── run_pipeline.py        # end-to-end batch pipeline
└── results/
    ├── charts/                # 12 PNG charts (auto-generated)
    └── rules/                 # rule + itemset CSVs (auto-generated)
```

---

## 🚀 Quick start

There is a one-shot setup script for each platform. Each one creates a
virtual environment, installs dependencies, generates the dataset, and runs
the pipeline so the app is ready to launch.

### macOS & Linux

```bash
./setup.sh          # one-shot setup (venv + deps + dataset + pipeline)
./run.sh            # launch the dashboard at http://localhost:8501
```

### Windows (PowerShell)

```powershell
.\setup.ps1         # one-shot setup (venv + deps + dataset + pipeline)
.\run.ps1           # launch the dashboard at http://localhost:8501
```
If PowerShell blocks scripts, run once:  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Android (via Termux)

Install Termux from **F-Droid** (not Google Play — the Play Store build is
outdated), copy the project folder into Termux's home, then:

```bash
bash setup-termux.sh   # one-shot setup (slower on Android)
bash run-termux.sh     # launch the dashboard
termux-open-url http://localhost:8501   # open it in Chrome
```

> **Honest note on Android:** this is the *same* Streamlit web app running as
> a local server on the phone — there is no APK. For a smoother mobile
> experience, deploy the app once (see the Deployment Guide PDF) and open
> the public URL on the phone instead.

### Manual / any platform

```bash
# 1. install deps
pip install -r requirements.txt

# 2. reproduce every chart and rule table
python -m scripts.run_pipeline

# 3. launch the interactive dashboard
streamlit run app/dashboard.py

# 4. (optional) launch the REST API for programmatic access
python -m api.server
```

The dashboard opens at <http://localhost:8501>; the API at <http://localhost:5000>.

**Prerequisite:** Python 3.10 or newer from <https://python.org>.

---

## 🔌 REST API (Flask)

`api/server.py` exposes the mined rules and recommendations as JSON so other
services (mobile app, BI tool, chatbot) can consume them without the dashboard.
The dataset and rule sets load once at startup, so every request is a fast
dictionary lookup.

| Method & endpoint | Returns |
|-------------------|---------|
| `GET /api/health` | Service status + transaction/rule counts |
| `GET /api/stats` | Dataset headline stats |
| `GET /api/products/top?n=20` | Top products by purchase frequency |
| `GET /api/categories` | Category distribution |
| `GET /api/rules?algo=apriori&limit=20` | Top association rules (`algo` = `apriori` or `fpgrowth`) |
| `GET /api/recommend?product=<name>&n=5` | Cross-sell recommendations for a product |
| `GET /api/anomalies` | Anomaly breakdown by fault type |

Example:

```bash
curl "http://localhost:5000/api/recommend?product=BrewMaster%20Beer&n=3"
# -> {"product":"BrewMaster Beer","count":2,
#     "recommendations":[{"product":"FreshCo Tortilla Chips","lift":23.66,...},
#                        {"product":"ValueMart Tortilla Chips","lift":10.60,...}]}
```

---

## 🔬 Methodology

### 1. Data preparation
`src/data_loader.py` reads the CSV, parses the `timestamp` column as a real
datetime, splits rows into a **clean** set and an **anomaly** set using the
generator's `is_anomaly` flag plus a defensive numeric check (negative/zero
quantity or price, blank category), and builds the one-hot transaction
matrix required by `mlxtend`.

### 2. Association-rule mining
We run **two** algorithms on the same matrix so we can compare them:

* **Apriori** — level-wise candidate generation; intuitive, easy to explain.
* **FP-Growth** — mines a Frequent-Pattern tree; usually faster on dense data.

Both produce rules scored by **support** (how often the combo appears),
**confidence** (P(consequent | antecedent)) and **lift** (how much more
likely the consequent is, vs. its base rate). We surface the top rules by
lift, because a high-confidence rule on a popular item can still be
uninformative — lift corrects for popularity.

### 3. Cross-sell recommender
`src/recommender.py` matches rules whose antecedent is a *subset* of the
query basket, deduplicates the consequents, and ranks by lift. This is the
"what should we suggest next?" engine behind the dashboard's Recommender tab.

### 4. Reporting
The batch pipeline writes:
* `results/charts/*.png` — every EDA and rule figure
* `results/rules/*.csv` — Apriori rules, FP-Growth rules, frequent itemsets,
  the algorithm comparison, and the master recommendation table

---

## ⚖️ Ethical note

Association rules describe **past** purchasing patterns; they do not predict
need. Recommendations are surfaced transparently (with their confidence and
lift) and must not be used to manipulate vulnerable segments. See the
report's *Limitations & Ethics* section.

---

## 📊 Key results (5,000-row dataset)

These numbers regenerate automatically when you run the pipeline:

* ~1,300 baskets, 493 unique products, 12 categories
* Apriori and FP-Growth produce **identical** rule sets on this dataset
  (as expected — they mine the same support counts, just faster)
* Top cross-sell pairs surface from the seeded affinities — e.g. bread +
  milk, pasta + pasta sauce, chips + cola
* 3.02% anomaly rate, evenly split across 7 fault types
