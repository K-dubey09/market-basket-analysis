"""FP-Growth association-rule mining wrapper around mlxtend.

FP-Growth builds a Frequent-Pattern tree and mines it without candidate
generation, so it is typically faster than Apriori on dense / wide
transaction sets. The public API mirrors ``apriori_miner`` so the two can be
swapped and benchmarked from the Streamlit app.
"""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

import config


def run_fpgrowth(transaction_matrix: pd.DataFrame,
                 min_support: float = config.MIN_SUPPORT,
                 min_confidence: float = config.MIN_CONFIDENCE,
                 min_lift: float = config.MIN_LIFT,
                 max_len: int = config.MAX_ITEMSET_LEN) -> pd.DataFrame:
    tm = transaction_matrix.astype(bool)
    frequent = fpgrowth(
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
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def frequent_itemsets_fpgrowth(transaction_matrix: pd.DataFrame,
                              min_support: float = config.MIN_SUPPORT,
                              max_len: int = config.MAX_ITEMSET_LEN) -> pd.DataFrame:
    freq = fpgrowth(
        transaction_matrix.astype(bool), min_support=min_support, use_colnames=True, max_len=max_len
    )
    freq = freq[freq["itemsets"].apply(len) >= 2].copy()
    freq["items"] = freq["itemsets"].apply(lambda s: " + ".join(sorted(s)))
    return freq.sort_values("support", ascending=False).reset_index(drop=True)
