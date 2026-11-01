"""Chart rendering for EDA and association-rule results.

All figures are saved as PNGs to ``results/charts`` and also returned so the
Streamlit app can show them live. We pin a CJK-safe matplotlib font fallback
so labels never render as boxes if a product name were ever non-ASCII.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx

import config

# --- font setup (the dataset is English, but be defensive for any future data)
for _f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="viridis")

PALETTE = sns.color_palette("viridis", n_colors=12)


# --------------------------------------------------------------------------- helpers
def _save(fig, name: str) -> Path:
    out = config.CHARTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- EDA charts
def chart_top_products(df: pd.DataFrame, n: int = 15) -> Path:
    from src.eda import top_products
    data = top_products(df, n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=data, y="product", x="count", ax=ax, color=PALETTE[3])
    ax.set_title(f"Top {n} Products by Purchase Frequency", fontsize=13, weight="bold")
    ax.set_xlabel("Number of line items")
    ax.set_ylabel("")
    for i, v in enumerate(data["count"]):
        ax.text(v + 1, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    return _save(fig, "01_top_products.png")


def chart_top_revenue(df: pd.DataFrame, n: int = 15) -> Path:
    from src.eda import top_products_by_revenue
    data = top_products_by_revenue(df, n)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=data, y="product", x="revenue", ax=ax, color=PALETTE[5])
    ax.set_title(f"Top {n} Products by Revenue", fontsize=13, weight="bold")
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("")
    for i, v in enumerate(data["revenue"]):
        ax.text(v + 5, i, f"${v:,.0f}", va="center", fontsize=8)
    plt.tight_layout()
    return _save(fig, "02_top_revenue.png")


def chart_category_distribution(df: pd.DataFrame) -> Path:
    from src.eda import category_distribution
    data = category_distribution(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, y="category", x="count", ax=ax,
                hue="category", palette="viridis", legend=False)
    ax.set_title("Transactions by Category", fontsize=13, weight="bold")
    ax.set_xlabel("Line items")
    ax.set_ylabel("")
    plt.tight_layout()
    return _save(fig, "03_category_distribution.png")


def chart_basket_size(df: pd.DataFrame) -> Path:
    from src.eda import basket_size_distribution
    data = basket_size_distribution(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=data, x="basket_size", y="num_baskets", ax=ax, color=PALETTE[2])
    ax.set_title("Basket Size Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Items per basket")
    ax.set_ylabel("Number of baskets")
    for i, v in enumerate(data["num_baskets"]):
        ax.text(i, v + 2, str(v), ha="center", fontsize=8)
    plt.tight_layout()
    return _save(fig, "04_basket_size.png")


def chart_daily_transactions(df: pd.DataFrame) -> Path:
    from src.eda import daily_transactions
    data = daily_transactions(df)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(data["date"], data["transactions"], color=PALETTE[4], linewidth=1.5)
    ax.fill_between(data["date"], data["transactions"], alpha=0.2, color=PALETTE[4])
    ax.set_title("Daily Transaction Volume", fontsize=13, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    plt.tight_layout()
    return _save(fig, "05_daily_transactions.png")


def chart_hourly_pattern(df: pd.DataFrame) -> Path:
    from src.eda import hourly_pattern
    data = hourly_pattern(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=data, x="hour", y="transactions", ax=ax, color=PALETTE[6])
    ax.set_title("Hourly Transaction Pattern", fontsize=13, weight="bold")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Transactions")
    plt.tight_layout()
    return _save(fig, "06_hourly_pattern.png")


def chart_category_heatmap(df: pd.DataFrame) -> Path:
    from src.eda import category_cooccurrence
    mat = category_cooccurrence(df)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", ax=ax, cbar_kws={"label": "co-baskets"})
    ax.set_title("Category Co-occurrence Heatmap", fontsize=13, weight="bold")
    plt.tight_layout()
    return _save(fig, "07_category_heatmap.png")


def chart_anomaly_breakdown(df: pd.DataFrame) -> Path:
    from src.eda import anomaly_breakdown
    data = anomaly_breakdown(df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=data, y="fault", x="count", ax=ax, color=PALETTE[8])
    ax.set_title("Anomaly Breakdown by Fault Type", fontsize=13, weight="bold")
    ax.set_xlabel("Count of anomalous rows")
    ax.set_ylabel("")
    for i, v in enumerate(data["count"]):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=8)
    plt.tight_layout()
    return _save(fig, "08_anomaly_breakdown.png")


# --------------------------------------------------------------------------- rule charts
def chart_top_rules(rules: pd.DataFrame, n: int = config.TOP_N_RULES) -> Path:
    return _save(fig_top_rules(rules, n), "09_top_rules_lift.png")


def fig_top_rules(rules: pd.DataFrame, n: int = config.TOP_N_RULES):
    """Return a matplotlib Figure of the top-n rules by lift (no save)."""
    top = rules.head(n).copy()
    top["rule"] = top["antecedents"] + " → " + top["consequents"]
    fig, ax = plt.subplots(figsize=(10, 6))
    if top.empty:
        ax.text(0.5, 0.5, "No rules to display", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.barplot(data=top, y="rule", x="lift", ax=ax, hue="rule",
                palette="magma", legend=False)
    ax.set_title(f"Top {n} Association Rules by Lift", fontsize=13, weight="bold")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def chart_scatter_support_confidence(rules: pd.DataFrame) -> Path:
    return _save(fig_scatter_support_confidence(rules), "10_scatter_support_confidence.png")


def fig_scatter_support_confidence(rules: pd.DataFrame):
    """Return a matplotlib Figure of support vs confidence (colour = lift)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if rules.empty:
        ax.text(0.5, 0.5, "No rules to display", ha="center", va="center")
        ax.axis("off")
        return fig
    sc = ax.scatter(
        rules["support"], rules["confidence"],
        c=rules["lift"], cmap="viridis", s=20, alpha=0.7, edgecolors="none"
    )
    fig.colorbar(sc, ax=ax, label="Lift")
    ax.set_title("Support vs Confidence (colour = Lift)", fontsize=13, weight="bold")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    fig.tight_layout()
    return fig


def chart_top_itemsets(itemsets: pd.DataFrame, n: int = 15) -> Path:
    top = itemsets.head(n).copy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=top, y="items", x="support", ax=ax, color=PALETTE[4])
    ax.set_title(f"Top {n} Frequent Itemsets (≥2 items)", fontsize=13, weight="bold")
    ax.set_xlabel("Support")
    ax.set_ylabel("")
    plt.tight_layout()
    return _save(fig, "11_top_itemsets.png")


def chart_rules_network(rules: pd.DataFrame, n: int = 25) -> Path:
    return _save(fig_rules_network(rules, n), "12_rules_network.png")


def fig_rules_network(rules: pd.DataFrame, n: int = 25):
    """Return a matplotlib Figure of the rule network (no save).

    Used by the Streamlit Rules page so the network re-renders live from the
    current ``rules`` DataFrame whenever the mining parameters change.
    """
    top = rules.head(n).copy()
    fig, ax = plt.subplots(figsize=(11, 8))
    if top.empty:
        ax.text(0.5, 0.5, "No rules to display", ha="center", va="center")
        ax.axis("off")
        return fig
    G = nx.DiGraph()
    for _, r in top.iterrows():
        ants = [a.strip() for a in r["antecedents"].split(",")]
        cons = [c.strip() for c in r["consequents"].split(",")]
        for a in ants:
            for c in cons:
                G.add_edge(a, c, weight=r["lift"], confidence=r["confidence"])
    pos = nx.spring_layout(G, k=1.2, seed=42)
    degs = dict(G.degree())
    sizes = [degs[node] * 350 + 200 for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color="#440154", alpha=0.85)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white")
    widths = [G[u][v]["weight"] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#21918c",
                           width=[0.5 + w * 0.2 for w in widths],
                           arrows=True, arrowsize=12, alpha=0.6)
    ax.set_title(f"Association-Rule Network (top {n} rules)", fontsize=13, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig
