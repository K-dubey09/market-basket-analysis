"""Apriori association-rule mining wrapper around mlxtend.

Apriori uses a level-wise, candidate-generation search: it counts 1-itemsets,
prunes by min support, builds 2-itemset candidates from the survivors, and so
on. The cost grows exponentially with itemset length, so we cap
``MAX_ITEMSET_LEN``. For our 5,000-row, ~1,300-basket dataset this finishes
in well under a second.
"""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

import config


def run_apriori(transaction_matrix: pd.DataFrame,
                min_support: float = config.MIN_SUPPORT,
                min_confidence: float = config.MIN_CONFIDENCE,
                min_lift: float = config.MIN_LIFT,
                max_len: int = config.MAX_ITEMSET_LEN) -> pd.DataFrame:
    """Return frequent itemsets + association rules sorted by lift.

    Parameters mirror the mlxtend API; defaults come from ``config`` so the
    app and the batch pipeline stay in sync. The matrix is cast to bool
    because mlxtend warns (and will eventually error) on non-bool frames.
    """
    tm = transaction_matrix.astype(bool)
    frequent = apriori(
        tm,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    if frequent.empty:
        return pd.DataFrame(columns=[
            "antecedents", "consequents", "support", "confidence",
            "lift", "leverage", "conviction",
        ])
    rules = association_rules(
        frequent, metric="confidence", min_threshold=min_confidence
    )
    rules = rules[rules["lift"] >= min_lift].copy()
    # Make frozenset columns readable (sorted comma-joined strings).
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def frequent_itemsets_apriori(transaction_matrix: pd.DataFrame,
                             min_support: float = config.MIN_SUPPORT,
                             max_len: int = config.MAX_ITEMSET_LEN) -> pd.DataFrame:
    """Frequent itemsets only (no rules) — used by the EDA 'top combos' chart."""
    freq = apriori(
        transaction_matrix.astype(bool), min_support=min_support, use_colnames=True, max_len=max_len
    )
    freq = freq[freq["itemsets"].apply(len) >= 2].copy()
    freq["items"] = freq["itemsets"].apply(lambda s: " + ".join(sorted(s)))
    return freq.sort_values("support", ascending=False).reset_index(drop=True)
