# `data/dataset.csv`

A 5,000-row synthetic retail basket dataset generated for market-basket
analysis. Reproducible from `scripts/generate_dataset.py` (`random.seed(42)`).

## Columns

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | `TXN######` — shared across all line items in one basket |
| `customer_id` | string | `CUST####` — 800 customers, ~150 loyal (weighted 3×) |
| `product_name` | string | Branded SKU, e.g. `FreshCo White Bread` (500 products) |
| `category` | string | One of 12 categories (Bakery, Dairy, Produce, ...) |
| `quantity` | int | Units purchased (1–5 typical; anomalies: 0, negative, or 999) |
| `price` | float | Unit price in USD (category-specific range; anomalies: 0, negative, or 9999.99) |
| `timestamp` | datetime | Checkout time `YYYY-MM-DD HH:MM:SS`, sequential, store hours 08:00–22:00 |
| `is_anomaly` | int (0/1) | 1 = corrupted row (excluded from mining) |

## Why affinities are realistic

The generator seeds **24 affinity clusters** (breakfast, Italian dinner,
BBQ, baby care, sandwich, salad, pizza night, taco night, cleaning, pet
bundles, ...). ~70% of baskets are themed around one cluster, the rest are
random. This produces co-occurrence patterns a real Apriori/FP-Growth model
can discover, instead of pure-noise data.

## Anomaly rate

~3% of rows are deliberately corrupted and flagged `is_anomaly = 1`, evenly
split across seven fault types so the data-quality page has something to
drill into.
