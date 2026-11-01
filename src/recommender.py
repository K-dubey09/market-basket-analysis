"""Cross-sell recommender built on mined association rules.

Given a product (or a small basket of products), return the top-N products
most likely to be bought together, scored by rule confidence and lift. This
is the "what should we suggest next?" engine that powers the app's
recommender tab and the report's upselling section.

Two scoring modes:
* ``score="confidence"`` — rank by P(consequent | antecedent)
* ``score="lift"`` — rank by how much more likely the consequent is
  vs. its base rate (corrects for popularity)
"""
from __future__ import annotations

import pandas as pd

import config


def recommend(rules: pd.DataFrame, query_products, top_n: int = config.RECOMMEND_TOP_N,
              score: str = "lift") -> pd.DataFrame:
    """Return recommended products for ``query_products``.

    ``query_products`` may be a single product name (str) or an iterable.
    Rules whose antecedent is a *subset* of the query set are matched, so a
    multi-product basket returns recommendations consistent with all items.
    """
    if isinstance(query_products, str):
        query = {query_products}
    else:
        query = set(query_products)
    if rules.empty or not query:
        return pd.DataFrame(columns=["product", "confidence", "lift", "rule"])

    recs = []
    for _, r in rules.iterrows():
        ants = {a.strip() for a in r["antecedents"].split(",")}
        cons = {c.strip() for c in r["consequents"].split(",")}
        if ants and ants.issubset(query) and not (cons & query):
            for prod in cons:
                recs.append({
                    "product": prod,
                    "confidence": r["confidence"],
                    "lift": r["lift"],
                    "rule": f"{r['antecedents']} -> {r['consequents']}",
                })
    if not recs:
        return pd.DataFrame(columns=["product", "confidence", "lift", "rule"])
    out = pd.DataFrame(recs)
    # aggregate duplicate products across multiple matching rules
    out = (out.groupby("product")
              .agg(confidence=("confidence", "max"), lift=("lift", "max"),
                   rule=("rule", "first"))
              .reset_index())
    return out.sort_values(score, ascending=False).head(top_n).reset_index(drop=True)


def build_recommendation_table(rules: pd.DataFrame, top_products: list[str],
                               top_n: int = config.RECOMMEND_TOP_N) -> pd.DataFrame:
    """For every product in ``top_products`` produce its top-N partners.

    Used to generate the report's master upselling table — one row per
    (seed product, recommended product) pair with the supporting metrics.
    """
    rows = []
    for prod in top_products:
        recs = recommend(rules, prod, top_n=top_n)
        for _, r in recs.iterrows():
            rows.append({
                "seed_product": prod,
                "recommended": r["product"],
                "confidence": round(r["confidence"], 3),
                "lift": round(r["lift"], 2),
            })
    return pd.DataFrame(rows)
