"""Exploratory Data Analysis helpers.

Each function returns a tidy ``DataFrame`` ready for plotting; the actual
figure rendering lives in ``visualizer.py`` so EDA stays testable and
side-effect free.
"""
from __future__ import annotations

import pandas as pd

import config


def top_products(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Most frequently purchased products (by line-item count)."""
    counts = df[config.PRODUCT_COL].value_counts().head(n).reset_index()
    counts.columns = ["product", "count"]
    return counts


def top_products_by_revenue(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top products by total revenue (quantity * price)."""
    g = (
        df.assign(revenue=df[config.QTY_COL] * df[config.PRICE_COL])
        .groupby(config.PRODUCT_COL)["revenue"].sum()
        .sort_values(ascending=False).head(n).reset_index()
    )
    g.columns = ["product", "revenue"]
    return g


def category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[config.CATEGORY_COL].value_counts().reset_index()
    counts.columns = ["category", "count"]
    return counts


def basket_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.groupby(config.ID_COL)[config.PRODUCT_COL].nunique()
    out = sizes.value_counts().sort_index().reset_index()
    out.columns = ["basket_size", "num_baskets"]
    return out


def daily_transactions(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby(df[config.TIME_COL].dt.date)[config.ID_COL].nunique()
    out = s.reset_index()
    out.columns = ["date", "transactions"]
    return out


def hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(df[config.TIME_COL].dt.hour)[config.ID_COL].nunique().reset_index()
    out.columns = ["hour", "transactions"]
    return out


def top_customers(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    g = df.groupby(config.CUSTOMER_COL).agg(
        baskets=(config.ID_COL, "nunique"),
        items=(config.PRODUCT_COL, "count"),
        spend=(config.PRICE_COL, lambda s: (s * df.loc[s.index, config.QTY_COL]).sum()),
    ).sort_values("spend", ascending=False).head(n).reset_index()
    return g


def category_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise category co-occurrence matrix across baskets."""
    cats = (
        df.groupby(config.ID_COL)[config.CATEGORY_COL]
        .apply(lambda s: list(set(s)))
    )
    uniq = sorted(df[config.CATEGORY_COL].dropna().unique())
    mat = pd.DataFrame(0, index=uniq, columns=uniq)
    for lst in cats:
        for i in range(len(lst)):
            for j in range(len(lst)):
                mat.loc[lst[i], lst[j]] += 1
    return mat


def anomaly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Classify anomaly rows by fault type for the EDA chart."""
    anom = df[df[config.ANOMALY_COL] == 1].copy()
    labels = []
    for _, r in anom.iterrows():
        q, p, c = r[config.QTY_COL], r[config.PRICE_COL], str(r[config.CATEGORY_COL])
        if pd.isna(q) or q == 0:
            labels.append("quantity=0")
        elif q < 0:
            labels.append("quantity<0")
        elif q >= 100:
            labels.append("quantity huge")
        elif pd.isna(p) or p == 0:
            labels.append("price=0")
        elif p < 0:
            labels.append("price<0")
        elif p >= 1000:
            labels.append("price huge")
        elif c.strip() == "":
            labels.append("missing category")
        else:
            labels.append("other")
    out = pd.Series(labels).value_counts().reset_index()
    out.columns = ["fault", "count"]
    return out
