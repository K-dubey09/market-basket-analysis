"""Streamlit app — Market Basket Analysis dashboard (reactive edition).

Run with:  streamlit run app/dashboard.py

Six pages, all driven from one cached dataset + one cached rule set. Every
widget that changes the analysis (mining parameters in the sidebar, the top-N
slider on the Rules page, the chart-mode toggle on EDA, the seed product on the
Recommender, a fresh upload) re-runs the affected computation and re-renders
the charts live — nothing here is a static pre-rendered image except the rule
network fallback. Charts are Plotly, rendered with ``width="stretch"``
so they re-flow to the window width (mobile-friendly and truly responsive).

  0. Upload Data   — bring your own XLSX/CSV; replace or append to the dataset
  1. Overview      — KPIs + live category mix + live top-N products
  2. EDA           — eight interactive charts, all reactive to a top-N slider
  3. Association Rules — top-N bar, support-vs-confidence bubble, interactive network
  4. Recommender   — pick a product, live cross-sell bar + table
  5. Data Quality  — anomaly drill-down + cleaned-data preview
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure the project root (parent of app/) is importable when run via
# `streamlit run app/dashboard.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.data_loader import (
    load_dataset, split_anomalies, add_basket_features, build_transaction_matrix,
    quality_report,
)
from src import apriori_miner, fpgrowth_miner, recommender, eda
from src import interactive as iv


# Path used to persist user-uploaded / merged data so it survives restarts.
# The active dataset is this file if it exists, otherwise the bundled CSV.
UPLOADED_PATH = config.DATA_DIR / "dataset_uploaded.csv"
EXPECTED_COLS = list(config.COLUMNS)

# --------------------------------------------------------------------------- page config
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- helpers
def _active_dataset_path() -> Path:
    """Return the path of the dataset that should be loaded right now."""
    return UPLOADED_PATH if UPLOADED_PATH.exists() else Path(config.DATASET_PATH)


def _coerce_to_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce a user frame to the project schema.

    Returns ``(cleaned, errors)``. Tries to be forgiving: column-name matching
    is case-insensitive and trims whitespace; missing optional columns are
    filled with sensible defaults so a minimal upload still works.
    """
    errors: list[str] = []
    if df.empty:
        return df, ["The uploaded file is empty."]

    # Normalise column names: lower + strip
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for expected in EXPECTED_COLS:
        for c in df.columns:
            if c.lower() == expected.lower() and c != expected:
                rename[c] = expected
                break
    df = df.rename(columns=rename)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if config.ID_COL in missing or config.PRODUCT_COL in missing:
        errors.append(
            f"Required columns missing: {config.ID_COL} and/or "
            f"{config.PRODUCT_COL}. The file must at least contain a "
            f"transaction id and a product name."
        )
        return df, errors

    # Fill optional columns with defaults if absent
    defaults = {
        config.CUSTOMER_COL: "",
        config.CATEGORY_COL: "Uncategorised",
        config.QTY_COL: 1,
        config.PRICE_COL: 0.0,
        config.TIME_COL: pd.Timestamp.now(),
        config.ANOMALY_COL: 0,
    }
    for col, default in defaults.items():
        if col in missing:
            df[col] = default
        elif col == config.ANOMALY_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Type coercion
    df[config.QTY_COL] = pd.to_numeric(df[config.QTY_COL], errors="coerce")
    df[config.PRICE_COL] = pd.to_numeric(df[config.PRICE_COL], errors="coerce")
    df[config.TIME_COL] = pd.to_datetime(df[config.TIME_COL], errors="coerce")
    df[config.ID_COL] = df[config.ID_COL].astype(str).str.strip()
    df[config.PRODUCT_COL] = df[config.PRODUCT_COL].astype(str).str.strip()
    df[config.CUSTOMER_COL] = df[config.CUSTOMER_COL].astype(str).str.strip()
    df[config.CATEGORY_COL] = df[config.CATEGORY_COL].astype(str).str.strip()

    # Drop rows missing the essential columns
    before = len(df)
    df = df.dropna(subset=[config.ID_COL, config.PRODUCT_COL])
    df = df[df[config.ID_COL] != ""]
    df = df[df[config.PRODUCT_COL] != ""]
    dropped = before - len(df)
    if dropped:
        errors.append(f"{dropped} row(s) dropped — blank transaction id or product name.")

    # Keep only known columns, in canonical order
    df = df[EXPECTED_COLS].reset_index(drop=True)
    return df, errors


def _read_upload(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded file into a DataFrame, supporting .csv and .xlsx."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return pd.DataFrame(), ["Unsupported file type. Use .csv or .xlsx."]
    except Exception as exc:  # noqa: BLE001 — surface any parse error to the UI
        return pd.DataFrame(), [f"Could not parse the file: {exc}"]
    return _coerce_to_schema(df)


# --------------------------------------------------------------------------- caching
# A single int "data version" sentinel busts the cache whenever the active
# dataset changes (upload/append/replace/reset). All five mining parameters
# are passed as args to get_rules, so changing any of them re-runs mining —
# that is what makes the dashboard reactive to the sidebar sliders.
@st.cache_data(show_spinner="Loading dataset...")
def get_data(_version: int):
    raw = load_dataset(_active_dataset_path())
    clean, anomalies = split_anomalies(raw)
    clean = add_basket_features(clean)
    return raw, clean, anomalies


@st.cache_data(show_spinner="Mining association rules...")
def get_rules(_version: int, min_support, min_confidence, min_lift, max_len, algo):
    _, clean, _ = get_data(_version)
    matrix = build_transaction_matrix(clean)
    if algo == "Apriori":
        rules = apriori_miner.run_apriori(matrix, min_support, min_confidence, min_lift, max_len)
        itemsets = apriori_miner.frequent_itemsets_apriori(matrix, min_support, max_len)
    else:
        rules = fpgrowth_miner.run_fpgrowth(matrix, min_support, min_confidence, min_lift, max_len)
        itemsets = fpgrowth_miner.frequent_itemsets_fpgrowth(matrix, min_support, max_len)
    return rules, itemsets


# --------------------------------------------------------------------------- session state
if "data_version" not in st.session_state:
    st.session_state.data_version = 0
DATA_VERSION = st.session_state.data_version

raw, clean_df, anomalies = get_data(DATA_VERSION)
qr = quality_report(raw)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🛒 Market Basket Analysis")
    st.caption("Retail upselling · college–HCL collaboration project")
    st.markdown("---")
    st.markdown("### Dataset snapshot")
    src = "uploaded" if UPLOADED_PATH.exists() else "bundled"
    st.caption(f"Active dataset: **{src}** (`{_active_dataset_path().name}`)")
    s1, s2 = st.columns(2)
    s1.metric("Transactions", f"{qr['transactions']:,}")
    s2.metric("Products", qr["products"])
    s1.metric("Customers", f"{qr['customers']:,}")
    s2.metric("Anomalies", f"{qr['anomalies']} ({100*qr['anomalies']/qr['total_rows']:.1f}%)")
    st.caption(f"Date range: {qr['date_range'][0]} → {qr['date_range'][1]}")
    st.markdown("---")
    st.markdown("### Mining parameters")
    st.caption("Changing any slider re-mines the rules and re-renders every rule chart.")
    algo = st.radio("Algorithm", ["Apriori", "FP-Growth"], horizontal=True)
    min_support = st.slider("Min support", 0.001, 0.05, float(config.MIN_SUPPORT), 0.001, format="%.3f")
    min_confidence = st.slider("Min confidence", 0.05, 1.0, float(config.MIN_CONFIDENCE), 0.05)
    min_lift = st.slider("Min lift", 0.5, 5.0, float(config.MIN_LIFT), 0.1)
    max_len = st.slider("Max itemset length", 2, 5, int(config.MAX_ITEMSET_LEN))
    st.markdown("---")
    st.caption("Built with Streamlit · mlxtend · Plotly")

rules, itemsets = get_rules(DATA_VERSION, min_support, min_confidence, min_lift, max_len, algo)

# --------------------------------------------------------------------------- pages
PAGES = ["📤 Upload Data", "📊 Overview", "🔍 EDA", "🔗 Association Rules",
         "🎯 Recommender", "🧹 Data Quality"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")


# ----------------------------------------------------------------- Upload Data
if page.startswith("📤"):
    st.title("📤 Upload Your Own Data")
    st.markdown(
        "Load a custom retail-basket dataset (.csv or .xlsx) and either "
        "**replace** the bundled dataset entirely or **append** the new rows "
        "to it. The merged dataset is saved to "
        f"`data/{UPLOADED_PATH.name}` so it survives a Streamlit restart, and "
        "every downstream page re-runs on the active data immediately."
    )
    st.markdown("### Expected schema")
    st.markdown(
        "The file should have one row per basket line item. The two required "
        "columns are `transaction_id` and `product_name`; the rest are filled "
        "with sensible defaults if absent. Column-name matching is "
        "case-insensitive."
    )
    schema_df = pd.DataFrame({
        "Column": EXPECTED_COLS,
        "Required": ["✓", " ", "✓", " ", " ", " ", "✓", " "],
        "Type": ["string", "string", "string", "string", "number", "number",
                 "datetime", "int (0/1)"],
        "Default if missing": ["—", "blank", "—", "Uncategorised", "1", "0.0",
                                "now", "0"],
    })
    st.dataframe(schema_df, width="stretch", hide_index=True)

    st.markdown("### Upload file")
    mode = st.radio(
        "Merge mode",
        ["Append to existing dataset", "Replace existing dataset"],
        horizontal=True,
        help="Append adds the new rows to the current dataset; Replace uses "
             "only the uploaded rows.",
    )
    uploaded = st.file_uploader(
        "Choose a .csv or .xlsx file",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        new_df, errors = _read_upload(uploaded)
        if errors and new_df.empty:
            st.error("Could not load the file:")
            for e in errors:
                st.markdown(f"- {e}")
        else:
            st.success(f"Loaded **{len(new_df):,}** rows from `{uploaded.name}`.")
            if errors:
                st.warning("Some rows needed fixing:")
                for e in errors:
                    st.markdown(f"- {e}")
            st.markdown("#### Preview of the uploaded rows")
            st.dataframe(new_df.head(10), width="stretch", hide_index=True)

            if mode.startswith("Append"):
                base = load_dataset(_active_dataset_path())
                merged = pd.concat([base[EXPECTED_COLS], new_df], ignore_index=True)
                st.info(f"Append mode: {len(base):,} existing + "
                        f"{len(new_df):,} new = **{len(merged):,}** rows.")
            else:  # Replace
                merged = new_df
                st.info(f"Replace mode: dataset will be **{len(merged):,}** rows.")

            if st.button("Confirm — save and re-run analysis", type="primary"):
                UPLOADED_PATH.parent.mkdir(parents=True, exist_ok=True)
                merged.to_csv(UPLOADED_PATH, index=False)
                st.session_state.data_version += 1
                st.success(
                    f"Saved to data/{UPLOADED_PATH.name} "
                    f"({len(merged):,} rows). Re-running analysis now — "
                    "navigate to the other pages to see the new rules and charts."
                )
                st.rerun()

    st.markdown("---")
    if UPLOADED_PATH.exists():
        st.markdown("### Reset to the bundled dataset")
        st.caption(f"A user dataset is currently active: `{UPLOADED_PATH.name}`.")
        if st.button("Delete uploaded dataset and revert to bundled"):
            UPLOADED_PATH.unlink(missing_ok=True)
            st.session_state.data_version += 1
            st.success("Reverted to the bundled dataset.")
            st.rerun()


# ----------------------------------------------------------------- Overview
elif page.startswith("📊"):
    st.title("📊 Market Basket Analysis — Overview")
    st.markdown(
        "An end-to-end pipeline that turns retail basket data into "
        "actionable cross-sell recommendations, mined with Apriori and "
        "FP-Growth and served through this interactive dashboard."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total rows", f"{qr['total_rows']:,}")
    k2.metric("Transactions", f"{qr['transactions']:,}")
    k3.metric("Unique products", qr["products"])
    k4.metric("Unique customers", f"{qr['customers']:,}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg basket size", f"{clean_df.groupby(config.ID_COL)[config.PRODUCT_COL].nunique().mean():.2f}")
    k2.metric("Total revenue", f"${(clean_df[config.QTY_COL]*clean_df[config.PRICE_COL]).sum():,.0f}")
    k3.metric("Anomalies flagged", f"{qr['anomalies']}")
    k4.metric("Rules mined", f"{len(rules):,}")

    st.markdown("---")
    st.markdown("### Category mix")
    cat_mode = st.radio("Chart style", ["Bar", "Pie"], horizontal=True, key="overview_cat")
    st.plotly_chart(iv.fig_category_distribution(clean_df, kind=cat_mode.lower()),
                    width="stretch")

    st.markdown("### Top products")
    topn_overview = st.slider("Top N", 5, 30, 10, 5, key="overview_topn")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(iv.fig_top_products(clean_df, topn_overview),
                        width="stretch")
    with c2:
        st.plotly_chart(iv.fig_top_revenue(clean_df, topn_overview),
                        width="stretch")


# ----------------------------------------------------------------- EDA
elif page.startswith("🔍"):
    st.title("🔍 Exploratory Data Analysis")
    if UPLOADED_PATH.exists():
        st.caption("Charts below render live from your uploaded data.")
    else:
        st.caption("Charts render live from the active dataset — drag any slider to update.")
    topn_eda = st.slider("Top N (products & revenue charts)", 5, 30, 10, 5, key="eda_topn")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(iv.fig_top_products(clean_df, topn_eda), width="stretch")
    with c2:
        st.plotly_chart(iv.fig_top_revenue(clean_df, topn_eda), width="stretch")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(iv.fig_category_distribution(clean_df, kind="bar"), width="stretch")
    with c2:
        st.plotly_chart(iv.fig_basket_size(clean_df), width="stretch")

    st.plotly_chart(iv.fig_daily_transactions(clean_df), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(iv.fig_hourly_pattern(clean_df), width="stretch")
    with c2:
        st.plotly_chart(iv.fig_anomaly_breakdown(raw), width="stretch")

    st.plotly_chart(iv.fig_category_heatmap(clean_df), width="stretch")


# ----------------------------------------------------------------- Rules
elif page.startswith("🔗"):
    st.title("🔗 Association Rules")
    st.caption(f"Algorithm: **{algo}** · {len(rules):,} rules · "
               f"{len(itemsets):,} frequent itemsets (≥2 items)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg lift", f"{rules['lift'].mean():.2f}" if not rules.empty else "—")
    m2.metric("Max lift", f"{rules['lift'].max():.2f}" if not rules.empty else "—")
    m3.metric("Avg confidence", f"{rules['confidence'].mean():.2f}" if not rules.empty else "—")

    # Reactive controls: changing top_n re-renders the bar, scatter and
    # network live from the current `rules` DataFrame (which itself is
    # reactive to the sidebar mining parameters).
    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        top_n = st.slider("Top N rules to visualise", 5, 100, int(config.TOP_N_RULES), 5,
                          help="Re-renders the bar, scatter and network charts live.")
    with ctrl2:
        sort_by = st.selectbox("Sort rules by", ["lift", "confidence", "support"], key="rules_sort")

    if rules.empty:
        st.warning("No rules at the current thresholds — lower min-support / "
                   "min-confidence / min-lift in the sidebar.")
    else:
        rules_sorted = rules.sort_values(sort_by, ascending=False).reset_index(drop=True)
        st.markdown(f"### Top rules by {sort_by}")
        st.dataframe(
            rules_sorted.head(top_n)[["antecedents", "consequents", "support",
                                      "confidence", "lift"]]
            .style.background_gradient(cmap="viridis", subset=["lift", "confidence"]),
            width="stretch", hide_index=True,
        )

        st.markdown("### Top rules by lift (bar)")
        st.plotly_chart(iv.fig_top_rules_bar(rules, top_n), width="stretch")

        st.markdown("### Support vs Confidence (size & colour = lift)")
        st.plotly_chart(iv.fig_rules_scatter(rules, top_n), width="stretch")

        st.markdown("### Rule network (top N)")
        st.plotly_chart(iv.fig_rules_network(rules, top_n), width="stretch")

    st.markdown("### Download rules")
    csv = rules.to_csv(index=False).encode()
    st.download_button("Download rules CSV", csv, f"{algo.lower()}_rules.csv", "text/csv")


# ----------------------------------------------------------------- Recommender
elif page.startswith("🎯"):
    st.title("🎯 Cross-Sell Recommender")
    st.markdown("Pick a product in the basket — see what customers commonly add next, ranked by lift.")
    products = sorted(clean_df[config.PRODUCT_COL].unique())
    seed_default = 0
    selected = st.selectbox("Seed product", products, index=seed_default)
    basket = st.multiselect("Also in basket (optional)", products)
    rec_n = st.slider("Recommendations", 1, 20, int(config.RECOMMEND_TOP_N), key="rec_n")
    query = set([selected] + basket)
    recs = recommender.recommend(rules, query, top_n=rec_n)

    if recs.empty:
        st.warning("No rules match this combination at the current thresholds. "
                   "Try lowering min-support / min-confidence in the sidebar, "
                   "or pick a different seed product.")
    else:
        st.markdown(f"#### Top {len(recs)} recommendations for **{selected}**")
        st.plotly_chart(iv.fig_recommendations_bar(recs), width="stretch")
        st.dataframe(
            recs.style.background_gradient(cmap="viridis", subset=["lift", "confidence"]),
            width="stretch", hide_index=True,
        )
        st.markdown(
            "**Reading the table** — *confidence* is the probability a customer "
            "who bought the seed product also bought the recommendation; *lift* "
            "compares that to the recommendation's overall popularity "
            "(lift > 1 means a genuine positive association)."
        )


# ----------------------------------------------------------------- Data Quality
elif page.startswith("🧹"):
    st.title("🧹 Data Quality")
    st.markdown(f"**{len(anomalies)}** rows were flagged and excluded from mining "
                f"({100*len(anomalies)/max(len(raw),1):.2f}% of the raw file).")
    st.markdown("### Anomaly breakdown by fault type")
    st.plotly_chart(iv.fig_anomaly_breakdown(raw), width="stretch")
    st.dataframe(eda.anomaly_breakdown(raw), width="stretch", hide_index=True)
    st.markdown("### Anomalous rows (sample)")
    st.dataframe(anomalies.head(20), width="stretch", hide_index=True)
    st.markdown("### Cleaned data preview")
    st.dataframe(clean_df.head(20), width="stretch", hide_index=True)
