"""Interactive Plotly chart builders for the Streamlit dashboard.

Every public function takes a tidy DataFrame (produced by ``src.eda`` or the
miners) and returns a ``plotly.graph_objects.Figure``. There is **no Streamlit
dependency** here, so every figure can be unit-tested in isolation; the
dashboard renders them with ``st.plotly_chart(fig, width="stretch")``
so they re-flow to the container width (truly responsive) and stay interactive
(hover, zoom, legend toggle).

These complement ``visualizer.py`` (matplotlib PNGs for the static report) —
same data, interactive medium.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

# Viridis colour scales / discrete palette mirror the matplotlib charts so the
# dashboard and the static report look like one product.
_SEQ = "Viridis"
_DISCRETE = px.colors.qualitative.Bold

# Shared layout baseline: minimal chrome, sensible height. Individual charts
# can override `height`/`title` via `_apply`.
_BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=44, b=10),
    font=dict(size=12),
    coloraxis=dict(colorbar=dict(thickness=10, len=0.85)),
)


def _apply(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """Apply the shared layout + a title to a figure and return it."""
    fig.update_layout(**_BASE_LAYOUT, title=title, height=height)
    return fig


# --------------------------------------------------------------------------- EDA
def fig_top_products(df: pd.DataFrame, n: int = 15) -> go.Figure:
    from src.eda import top_products
    d = top_products(df, n)
    fig = px.bar(d, y="product", x="count", orientation="h",
                 color="count", color_continuous_scale=_SEQ)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Line items: %{x}<extra></extra>")
    return _apply(fig, f"Top {n} Products by Frequency", 460)


def fig_top_revenue(df: pd.DataFrame, n: int = 15) -> go.Figure:
    from src.eda import top_products_by_revenue
    d = top_products_by_revenue(df, n)
    fig = px.bar(d, y="product", x="revenue", orientation="h",
                 color="revenue", color_continuous_scale=_SEQ)
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>")
    return _apply(fig, f"Top {n} Products by Revenue", 460)


def fig_category_distribution(df: pd.DataFrame, kind: str = "bar") -> go.Figure:
    from src.eda import category_distribution
    d = category_distribution(df)
    if kind == "pie":
        fig = px.pie(d, names="category", values="count",
                     color="category", color_discrete_sequence=_DISCRETE)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return _apply(fig, "Category Mix", 460)
    fig = px.bar(d, y="category", x="count", orientation="h",
                 color="category", color_discrete_sequence=_DISCRETE)
    fig.update_yaxes(autorange="reversed")
    return _apply(fig, "Transactions by Category", 460)


def fig_basket_size(df: pd.DataFrame) -> go.Figure:
    from src.eda import basket_size_distribution
    d = basket_size_distribution(df)
    fig = px.bar(d, x="basket_size", y="num_baskets",
                 color="num_baskets", color_continuous_scale=_SEQ)
    fig.update_traces(hovertemplate="Basket size %{x} items: %{y} baskets<extra></extra>")
    return _apply(fig, "Basket Size Distribution", 380)


def fig_daily_transactions(df: pd.DataFrame) -> go.Figure:
    from src.eda import daily_transactions
    d = daily_transactions(df)
    fig = px.area(d, x="date", y="transactions")
    fig.update_traces(line=dict(color="#3b528b"), hovertemplate="%{x}<br>%{y} transactions<extra></extra>")
    fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(
        buttons=[dict(count=7, label="1w", step="day"),
                 dict(count=1, label="1m", step="month"),
                 dict(step="all", label="all")]))
    return _apply(fig, "Daily Transaction Volume", 420)


def fig_hourly_pattern(df: pd.DataFrame) -> go.Figure:
    from src.eda import hourly_pattern
    d = hourly_pattern(df)
    fig = px.bar(d, x="hour", y="transactions",
                 color="transactions", color_continuous_scale=_SEQ)
    fig.update_traces(hovertemplate="Hour %{x}:00 → %{y} transactions<extra></extra>")
    return _apply(fig, "Hourly Transaction Pattern", 380)


def fig_category_heatmap(df: pd.DataFrame) -> go.Figure:
    from src.eda import category_cooccurrence
    mat = category_cooccurrence(df)
    fig = go.Figure(data=go.Heatmap(
        z=mat.values, x=mat.columns, y=mat.index, colorscale=_SEQ,
        hovertemplate="(%{y}, %{x}) co-baskets: %{z}<extra></extra>",
        colorbar=dict(thickness=10, len=0.85),
    ))
    return _apply(fig, "Category Co-occurrence Heatmap", 520)


def fig_anomaly_breakdown(df: pd.DataFrame) -> go.Figure:
    from src.eda import anomaly_breakdown
    d = anomaly_breakdown(df)
    fig = px.bar(d, y="fault", x="count", orientation="h",
                 color="count", color_continuous_scale=_SEQ)
    fig.update_yaxes(autorange="reversed")
    return _apply(fig, "Anomaly Breakdown by Fault Type", 380)


# --------------------------------------------------------------------------- Rules
def fig_top_rules_bar(rules: pd.DataFrame, n: int = 20) -> go.Figure:
    """Horizontal bar of the top-n rules by lift (interactive)."""
    top = rules.head(n).copy()
    if top.empty:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return _apply(fig, f"Top {n} Rules by Lift", 460)
    top["rule"] = top["antecedents"] + " → " + top["consequents"]
    fig = px.bar(top, y="rule", x="lift", orientation="h",
                 color="lift", color_continuous_scale="Magma",
                 custom_data=["confidence", "support"])
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(hovertemplate="<b>%{y}</b><br>lift=%{x:.2f} · conf=%{customdata[0]:.2f} · supp=%{customdata[1]:.3f}<extra></extra>")
    return _apply(fig, f"Top {n} Association Rules by Lift", 520)


def fig_rules_scatter(rules: pd.DataFrame, n: int = 100) -> go.Figure:
    """Support vs confidence bubble chart, colour & size = lift."""
    d = rules.head(n).copy()
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return _apply(fig, f"Top {n} Rules — Support vs Confidence", 460)
    d["rule"] = d["antecedents"] + " → " + d["consequents"]
    d["size"] = (d["lift"] - d["lift"].min() + 0.1) * 8 + 6
    fig = px.scatter(d, x="support", y="confidence", size="size", color="lift",
                     hover_name="rule", color_continuous_scale=_SEQ,
                     range_color=[d["lift"].min(), d["lift"].max()])
    fig.update_traces(hovertemplate="<b>%{hover_name}</b><br>support=%{x:.3f} · confidence=%{y:.3f}<extra></extra>")
    return _apply(fig, f"Top {n} Rules — Support vs Confidence", 480)


def fig_rules_network(rules: pd.DataFrame, n: int = 25) -> go.Figure:
    """Interactive rule network: antecedent → consequent edges, node size = degree."""
    import networkx as nx
    top = rules.head(n).copy()
    fig = go.Figure()
    if top.empty:
        fig.add_annotation(text="No rules to display", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16)
        return _apply(fig, f"Association-Rule Network (top {n})", 520)
    G = nx.DiGraph()
    for _, r in top.iterrows():
        ants = [a.strip() for a in r["antecedents"].split(",")]
        cons = [c.strip() for c in r["consequents"].split(",")]
        for a in ants:
            for c in cons:
                G.add_edge(a, c, weight=float(r["lift"]), conf=float(r["confidence"]))
    pos = nx.spring_layout(G, k=1.2, seed=42)
    degs = dict(G.degree())

    edge_x, edge_y = [], []
    for u, v in G.edges:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=1, color="#9aa0a6"), hoverinfo="none",
                            showlegend=False))
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes:
        node_x.append(pos[node][0]); node_y.append(pos[node][1])
        node_text.append(node); node_size.append(degs[node] * 7 + 12)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                            text=node_text, textposition="top center",
                            marker=dict(size=node_size, color="#440154", opacity=0.9,
                                        line=dict(width=1, color="white")),
                            hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    return _apply(fig, f"Association-Rule Network (top {n})", 540)


# --------------------------------------------------------------------------- Recommender
def fig_recommendations_bar(recs: pd.DataFrame) -> go.Figure:
    """Grouped bar of confidence + lift for each recommended product."""
    if recs.empty:
        fig = go.Figure()
        fig.add_annotation(text="No recommendations for this basket at the current thresholds",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font_size=14)
        return _apply(fig, "Recommendations", 380)
    d = recs.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Confidence", x=d["product"], y=d["confidence"],
                        marker_color="#21918c",
                        hovertemplate="<b>%{x}</b><br>confidence %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Lift", x=d["product"], y=d["lift"],
                        marker_color="#440154",
                        hovertemplate="<b>%{x}</b><br>lift %{y:.2f}<extra></extra>"))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.15))
    return _apply(fig, "Recommended Products — Confidence vs Lift", 380)
