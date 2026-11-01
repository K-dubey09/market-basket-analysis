"""Data loading, cleaning and basket construction.

Responsibilities
----------------
* Read ``data/dataset.csv`` into a typed ``DataFrame``.
* Split rows into a *clean* frame and an *anomaly* frame, using the
  ``is_anomaly`` flag plus a defensive numeric sanity check.
* Build the transactional one-hot matrix required by ``mlxtend``:
  one row per ``transaction_id``, one column per product, 1/0 membership.
* Lightweight quality-report helper used by EDA and the Streamlit sidebar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV with correct dtypes.

    The generator writes ``timestamp`` as ``YYYY-MM-DD HH:MM:SS`` and stores
    ``quantity`` / ``price`` / ``is_anomaly`` as integers/floats, so we parse
    them here once and let pandas carry the types downstream.
    """
    path = Path(path or config.DATASET_PATH)
    df = pd.read_csv(
        path,
        parse_dates=[config.TIME_COL],
        dtype={
            config.ID_COL: str,
            config.CUSTOMER_COL: str,
            config.PRODUCT_COL: str,
            config.CATEGORY_COL: str,
            config.ANOMALY_COL: int,
        },
    )
    df[config.QTY_COL] = pd.to_numeric(df[config.QTY_COL], errors="coerce")
    df[config.PRICE_COL] = pd.to_numeric(df[config.PRICE_COL], errors="coerce")
    return df


def split_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(clean, anomalies)``.

    A row counts as anomalous if either the generator's flag is set *or* a
    value is structurally invalid (negative/zero quantity or price, blank
    category). The defensive check catches rows the flag might miss.
    """
    flag = df[config.ANOMALY_COL] == 1
    bad_qty = df[config.QTY_COL] <= 0
    bad_price = df[config.PRICE_COL] <= 0
    bad_cat = df[config.CATEGORY_COL].fillna("").str.strip() == ""
    anom_mask = flag | bad_qty | bad_price | bad_cat
    return df.loc[~anom_mask].copy(), df.loc[anom_mask].copy()


def add_basket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-basket convenience columns: basket_size, line_total, hour."""
    df = df.copy()
    df["line_total"] = df[config.QTY_COL] * df[config.PRICE_COL]
    df["basket_size"] = df.groupby(config.ID_COL)[config.PRODUCT_COL].transform("count")
    df["hour"] = df[config.TIME_COL].dt.hour
    df["date"] = df[config.TIME_COL].dt.date
    return df


def build_transaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot transaction matrix (rows=baskets, cols=products, 1/0)."""
    basket = (
        df.groupby([config.ID_COL, config.PRODUCT_COL])[config.QTY_COL]
        .sum().unstack(fill_value=0)
    )
    return (basket > 0).astype(int)


def quality_report(df: pd.DataFrame) -> dict:
    """Compact dict of headline stats for the app sidebar and the report."""
    return {
        "total_rows": int(len(df)),
        "transactions": int(df[config.ID_COL].nunique()),
        "customers": int(df[config.CUSTOMER_COL].nunique()),
        "products": int(df[config.PRODUCT_COL].nunique()),
        "categories": int(df[config.CATEGORY_COL].nunique()),
        "anomalies": int((df[config.ANOMALY_COL] == 1).sum()),
        "date_range": (
            str(df[config.TIME_COL].min().date()),
            str(df[config.TIME_COL].max().date()),
        ),
    }
