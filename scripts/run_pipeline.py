"""End-to-end batch pipeline.

Run with:  python -m scripts.run_pipeline

It cleans the data, runs Apriori + FP-Growth, scores recommendations, renders
every EDA and rule chart to ``results/charts``, and writes the rule tables to
``results/rules``. This is the script the README tells graders / reviewers to
run to reproduce every artifact in the report.
"""
from __future__ import annotations

import time

import pandas as pd

import config
from src.data_loader import (
    load_dataset, split_anomalies, add_basket_features, build_transaction_matrix,
    quality_report,
)
from src import eda, apriori_miner, fpgrowth_miner, recommender, visualizer


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 60)
    print("MARKET BASKET ANALYSIS — PIPELINE")
    print("=" * 60)

    # 1. Load + clean ---------------------------------------------------------
    raw = load_dataset()
    report = quality_report(raw)
    print(f"\n[1] Loaded {report['total_rows']:,} rows | "
          f"{report['transactions']:,} baskets | {report['products']} products")
    clean, anomalies = split_anomalies(raw)
    print(f"    clean rows: {len(clean):,} | anomalies: {len(anomalies):,} "
          f"({100*len(anomalies)/len(raw):.2f}%)")

    clean = add_basket_features(clean)

    # 2. EDA charts -----------------------------------------------------------
    print("\n[2] Rendering EDA charts...")
    visualizer.chart_top_products(clean)
    visualizer.chart_top_revenue(clean)
    visualizer.chart_category_distribution(clean)
    visualizer.chart_basket_size(clean)
    visualizer.chart_daily_transactions(clean)
    visualizer.chart_hourly_pattern(clean)
    visualizer.chart_category_heatmap(clean)
    visualizer.chart_anomaly_breakdown(raw)
    print("    8 EDA charts saved to results/charts/")

    # 3. Transaction matrix ---------------------------------------------------
    print("\n[3] Building transaction matrix...")
    matrix = build_transaction_matrix(clean)
    print(f"    shape: {matrix.shape} (baskets x products)")

    # 4. Apriori --------------------------------------------------------------
    print("\n[4] Mining Apriori rules...")
    ap_rules = apriori_miner.run_apriori(matrix)
    ap_itemsets = apriori_miner.frequent_itemsets_apriori(matrix)
    print(f"    rules: {len(ap_rules)} | frequent itemsets (>=2): {len(ap_itemsets)}")
    ap_rules.to_csv(config.RULES_DIR / "apriori_rules.csv", index=False)
    ap_itemsets.to_csv(config.RULES_DIR / "apriori_itemsets.csv", index=False)

    # 5. FP-Growth ------------------------------------------------------------
    print("\n[5] Mining FP-Growth rules...")
    fp_rules = fpgrowth_miner.run_fpgrowth(matrix)
    fp_itemsets = fpgrowth_miner.frequent_itemsets_fpgrowth(matrix)
    print(f"    rules: {len(fp_rules)} | frequent itemsets (>=2): {len(fp_itemsets)}")
    fp_rules.to_csv(config.RULES_DIR / "fpgrowth_rules.csv", index=False)
    fp_itemsets.to_csv(config.RULES_DIR / "fpgrowth_itemsets.csv", index=False)

    # 6. Algorithm comparison -------------------------------------------------
    print("\n[6] Comparing algorithms...")
    compare = pd.DataFrame({
        "metric": ["rules_found", "itemsets_found", "avg_lift", "max_lift", "avg_confidence"],
        "apriori": [
            len(ap_rules), len(ap_itemsets),
            ap_rules["lift"].mean() if not ap_rules.empty else 0,
            ap_rules["lift"].max() if not ap_rules.empty else 0,
            ap_rules["confidence"].mean() if not ap_rules.empty else 0,
        ],
        "fpgrowth": [
            len(fp_rules), len(fp_itemsets),
            fp_rules["lift"].mean() if not fp_rules.empty else 0,
            fp_rules["lift"].max() if not fp_rules.empty else 0,
            fp_rules["confidence"].mean() if not fp_rules.empty else 0,
        ],
    })
    compare.to_csv(config.RULES_DIR / "algorithm_comparison.csv", index=False)
    print(compare.to_string(index=False))

    # 7. Rule charts ----------------------------------------------------------
    print("\n[7] Rendering rule charts...")
    visualizer.chart_top_rules(ap_rules)
    visualizer.chart_scatter_support_confidence(ap_rules)
    visualizer.chart_top_itemsets(ap_itemsets)
    visualizer.chart_rules_network(ap_rules)
    print("    4 rule charts saved to results/charts/")

    # 8. Recommender ----------------------------------------------------------
    print("\n[8] Building recommender table...")
    seeds = eda.top_products(clean, n=15)["product"].tolist()
    rec_table = recommender.build_recommendation_table(ap_rules, seeds, top_n=3)
    rec_table.to_csv(config.RULES_DIR / "recommendations.csv", index=False)
    print(f"    {len(rec_table)} (seed, recommendation) pairs saved")

    # 9. Summary --------------------------------------------------------------
    print("\n[9] Done.")
    print(f"    elapsed: {time.perf_counter() - t0:.2f}s")
    print(f"    charts : {config.CHARTS_DIR}")
    print(f"    rules  : {config.RULES_DIR}")


if __name__ == "__main__":
    main()
