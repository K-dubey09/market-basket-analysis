#!/usr/bin/env python3
"""Generate a 5,000-row retail market-basket dataset with seeded product affinities.

Output columns:
    transaction_id, customer_id, product_name, category,
    quantity, price, timestamp, is_anomaly

Design notes
------------
* Catalog: exactly 500 products across 12 categories. Each product is a
  branded variant (e.g. "FreshCo White Bread") so multiple SKUs share an
  underlying base product ("White Bread").
* Affinities: 20+ hand-seeded "theme" clusters of base products that tend to
  be bought together (breakfast, Italian dinner, BBQ, baby care, ...). ~70%
  of baskets are themed, the rest are random -- this produces co-occurrence
  patterns an Apriori / FP-Growth model can discover.
* Anomalies: ~3% of rows are corrupted (zero/negative quantity or price,
  huge values, missing category) and flagged with is_anomaly=1.
* Timestamps: one checkout time per basket, strictly increasing, formatted
  YYYY-MM-DD HH:MM:SS, within store hours (08:00-22:00).
* Reproducible: random.seed(42).
"""
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta

random.seed(42)

OUT_PATH = "/home/z/my-project/download/dataset.csv"
TOTAL_ROWS = 5000

# ---------------------------------------------------------------- 1. Catalog
# (target_count, base_products, brands, unit_price_range)
category_specs = {
    "Bakery":        (45, ["White Bread", "Whole Wheat Bread", "Sourdough", "Bagel", "Croissant",
                           "Muffin", "Donut", "Baguette", "Pita Bread", "Tortilla Wrap",
                           "Cinnamon Roll", "Brioche Bun", "Dinner Roll", "Ciabatta", "Naan"],
                          ["FreshCo", "ValueMart", "GoldenOven"], (1.5, 6.0)),
    "Dairy":         (45, ["Whole Milk", "Skim Milk", "Soy Milk", "Almond Milk", "Cheddar Cheese",
                           "Mozzarella", "Parmesan", "Greek Yogurt", "Butter", "Cream Cheese",
                           "Sour Cream", "Heavy Cream", "Eggs", "Cottage Cheese", "String Cheese"],
                          ["DailyFarm", "ValueMart", "PureLife"], (1.2, 8.0)),
    "Produce":       (50, ["Bananas", "Apples", "Oranges", "Tomatoes", "Avocados", "Spinach",
                           "Lettuce", "Carrots", "Onions", "Potatoes", "Strawberries",
                           "Blueberries", "Grapes", "Cucumber", "Broccoli", "Bell Pepper",
                           "Lemon", "Lime", "Garlic", "Ginger", "Mushrooms", "Corn",
                           "Peaches", "Pears", "Cherries"],
                          ["OrganicGarden", "ValueMart"], (0.5, 7.0)),
    "Beverages":     (50, ["Cola", "Diet Cola", "Lemon Soda", "Orange Juice", "Apple Juice",
                           "Cranberry Juice", "Bottled Water", "Sparkling Water", "Coffee Beans",
                           "Ground Coffee", "Tea Bags", "Green Tea", "Energy Drink",
                           "Sports Drink", "Beer", "Lager", "Red Wine", "White Wine",
                           "Lemonade", "Iced Tea"],
                          ["BrewMaster", "ValueMart", "PureLife"], (1.0, 25.0)),
    "Snacks":        (45, ["Potato Chips", "Tortilla Chips", "Pretzels", "Popcorn", "Chocolate Bar",
                           "Cookies", "Crackers", "Granola Bar", "Mixed Nuts", "Dried Fruit",
                           "Fruit Snacks", "Trail Mix", "Beef Jerky", "Rice Cakes", "Seeds"],
                          ["SweetTooth", "ValueMart", "FreshCo"], (1.0, 8.0)),
    "Meat":          (40, ["Chicken Breast", "Ground Beef", "Bacon", "Sausage", "Salmon",
                           "Shrimp", "Ham", "Turkey Slices", "Pork Chops", "Hot Dogs",
                           "Steak", "Ground Turkey", "Chicken Wings", "Lamb Chops", "Tilapia"],
                          ["PrimeCut", "ValueMart", "DailyFarm"], (4.0, 22.0)),
    "Frozen":        (40, ["Frozen Pizza", "Ice Cream", "Frozen Vegetables", "Frozen Berries",
                           "Frozen Fries", "Frozen Waffles", "Frozen Dumplings", "Fish Sticks",
                           "Frozen Chicken", "Frozen Burrito", "Frozen Lasagna", "Frozen Yogurt",
                           "Frozen Pie", "Frozen Meals"],
                          ["FreshCo", "ValueMart", "GoldenOven"], (2.0, 12.0)),
    "Pantry":        (55, ["Pasta", "Rice", "Flour", "Sugar", "Olive Oil", "Vegetable Oil",
                           "Salt", "Black Pepper", "Ketchup", "Mustard", "Mayo", "Soy Sauce",
                           "Vinegar", "Canned Beans", "Canned Tomatoes", "Peanut Butter",
                           "Jelly", "Cereal", "Oatmeal", "Pancake Mix", "Honey", "Maple Syrup",
                           "Pasta Sauce", "Salsa", "Broth"],
                          ["ValueMart", "PureLife", "NatureBest"], (0.8, 10.0)),
    "Household":     (35, ["Paper Towels", "Toilet Paper", "Dish Soap", "Laundry Detergent",
                           "Trash Bags", "Aluminum Foil", "Plastic Wrap", "Cleaning Spray",
                           "Sponges", "Bleach", "Glass Cleaner", "Disinfectant Wipes",
                           "Dishwasher Tabs", "Fabric Softener", "Air Freshener"],
                          ["CleanCare", "ValueMart", "HappyHome"], (1.5, 15.0)),
    "Personal Care": (35, ["Toothpaste", "Toothbrush", "Mouthwash", "Dental Floss", "Shampoo",
                           "Conditioner", "Body Wash", "Soap Bar", "Deodorant", "Hand Lotion",
                           "Razor", "Cotton Swabs", "Hand Sanitizer", "Tissue Pack", "Sunscreen"],
                          ["CleanCare", "PureLife", "ValueMart"], (1.5, 12.0)),
    "Baby":          (30, ["Diapers", "Baby Wipes", "Baby Formula", "Baby Lotion", "Baby Shampoo",
                           "Baby Food Jar", "Baby Cereal", "Baby Snacks", "Pacifier", "Bottle"],
                          ["LittleOne", "ValueMart", "PureLife"], (2.0, 30.0)),
    "Pets":          (30, ["Dog Food", "Cat Food", "Cat Litter", "Dog Treats", "Cat Treats",
                           "Bird Seed", "Fish Flakes", "Hamster Food", "Pet Shampoo", "Pet Toy"],
                          ["PawPal", "ValueMart", "NatureBest"], (3.0, 25.0)),
}

catalog = []
base_to_variants = defaultdict(list)  # base product name -> list of product dicts
for cat, (target, bases, brands, price_range) in category_specs.items():
    combos = [f"{brand} {base}" for base in bases for brand in brands]
    random.shuffle(combos)
    for name in combos[:target]:
        base = name
        for brand in brands:
            if name.startswith(brand + " "):
                base = name[len(brand) + 1:]
                break
        product = {
            "name": name,
            "category": cat,
            "base_price": round(random.uniform(*price_range), 2),
            "base": base,
        }
        catalog.append(product)
        base_to_variants[base].append(product)

assert len(catalog) == 500, f"catalog size = {len(catalog)}"

# ---------------------------------------------------------------- 2. Affinities
# Each cluster lists base-product names that tend to be bought together.
affinity_clusters = [
    ["Cereal", "Whole Milk", "Bananas", "Eggs", "White Bread", "Butter",
     "Coffee Beans", "Orange Juice", "Strawberries"],                       # breakfast
    ["Pasta", "Pasta Sauce", "Parmesan", "Canned Tomatoes", "Garlic",
     "Onions", "Red Wine", "Olive Oil", "Ground Beef"],                    # Italian dinner
    ["Potato Chips", "Cola", "Popcorn", "Chocolate Bar",
     "Tortilla Chips", "Lemon Soda"],                                       # movie night
    ["Ground Beef", "Hot Dogs", "White Bread", "Brioche Bun", "Cola",
     "Beer", "Potato Chips", "Corn"],                                       # BBQ
    ["Diapers", "Baby Wipes", "Baby Formula", "Baby Lotion", "Baby Shampoo"],  # baby care
    ["White Bread", "Ham", "Turkey Slices", "Cheddar Cheese", "Mayo",
     "Lettuce", "Tomatoes"],                                                # sandwich
    ["Coffee Beans", "Sugar", "Whole Milk", "Cookies", "Donut", "Muffin"],  # coffee break
    ["Lettuce", "Tomatoes", "Cucumber", "Olive Oil", "Vinegar",
     "Carrots", "Avocados"],                                                # salad
    ["Frozen Pizza", "Cola", "Beer", "Mozzarella", "Tortilla Chips"],       # pizza night
    ["Laundry Detergent", "Cleaning Spray", "Paper Towels", "Sponges",
     "Trash Bags", "Bleach"],                                               # cleaning
    ["Tortilla Wrap", "Ground Beef", "Avocados", "Cheddar Cheese",
     "Salsa", "Lettuce", "Tomatoes"],                                       # taco night
    ["Dog Food", "Dog Treats", "Pet Toy", "Pet Shampoo"],                   # dog bundle
    ["Cat Food", "Cat Litter", "Cat Treats"],                               # cat bundle
    ["Toothpaste", "Toothbrush", "Mouthwash", "Soap Bar",
     "Shampoo", "Conditioner"],                                             # personal care
    ["Tea Bags", "Sugar", "Honey", "Cookies", "Lemon"],                     # tea time
    ["Pancake Mix", "Maple Syrup", "Butter", "Strawberries",
     "Whole Milk", "Eggs"],                                                 # pancake breakfast
    ["Ground Beef", "Brioche Bun", "Cheddar Cheese", "Ketchup",
     "Mustard", "Potato Chips", "Cola"],                                    # burger night
    ["Broth", "Canned Tomatoes", "Carrots", "Onions", "Garlic", "Salt"],    # soup
    ["Flour", "Sugar", "Butter", "Eggs", "Whole Milk"],                     # baking
    ["Greek Yogurt", "Strawberries", "Blueberries", "Honey", "Granola Bar"],  # yogurt parfait
    ["Sausage", "Eggs", "White Bread", "Orange Juice", "Coffee Beans"],     # brunch
    ["Salmon", "Lemon", "Butter", "Garlic", "Olive Oil"],                   # fish dinner
    ["Ice Cream", "Chocolate Bar", "Cookies", "Strawberries"],              # dessert
    ["Tortilla Chips", "Salsa", "Beer", "Lemon"],                           # nachos + drinks
]
valid_bases = set(base_to_variants.keys())
affinity_clusters = [
    [b for b in cluster if b in valid_bases]
    for cluster in affinity_clusters
]
affinity_clusters = [c for c in affinity_clusters if len(c) >= 2]

# ---------------------------------------------------------------- 3. Customers
NUM_CUSTOMERS = 800
NUM_LOYAL = 150
customers = [f"CUST{i:04d}" for i in range(1, NUM_CUSTOMERS + 1)]
cust_weights = [3 if i < NUM_LOYAL else 1 for i in range(NUM_CUSTOMERS)]

# ---------------------------------------------------------------- 4. Timestamps
clock = datetime(2024, 1, 1, 8, 0, 0)


def next_timestamp():
    """Advance the store clock to the next checkout (08:00-22:00 daily)."""
    global clock
    gap = random.randint(15, 100)  # minutes since previous basket
    clock = clock + timedelta(minutes=gap)
    if clock.hour >= 22:
        clock = (clock + timedelta(days=1)).replace(hour=8, minute=0, second=0)
    return clock


# ---------------------------------------------------------------- 5. Baskets
SIZE_OPTS = [1, 2, 3, 4, 5, 6, 7, 8]
SIZE_WTS = [10, 20, 22, 16, 12, 9, 6, 5]


def basket_products(size):
    """Return a list of distinct product dicts for one basket of ~size items."""
    if random.random() < 0.7 and size >= 2:
        cluster = random.choice(affinity_clusters)
        n = min(size, len(cluster))
        chosen_bases = random.sample(cluster, n)
        products = [random.choice(base_to_variants[b]) for b in chosen_bases]
        if size > n:
            pool = [p for p in catalog if p not in products]
            products += random.sample(pool, min(size - n, len(pool)))
        # noise: occasionally add one unrelated product
        if random.random() < 0.3:
            pool = [p for p in catalog if p not in products]
            if pool:
                products.append(random.choice(pool))
    else:
        products = random.sample(catalog, min(size, len(catalog)))
    return products


rows = []
basket_idx = 0
while len(rows) < TOTAL_ROWS:
    remaining = TOTAL_ROWS - len(rows)
    size = random.choices(SIZE_OPTS, weights=SIZE_WTS)[0]
    size = min(size, remaining)
    basket_idx += 1
    txn_id = f"TXN{basket_idx:06d}"
    cust = random.choices(customers, weights=cust_weights)[0]
    ts = next_timestamp()
    products = basket_products(size)
    # guard against overflowing TOTAL_ROWS on the final basket
    if len(rows) + len(products) > TOTAL_ROWS:
        products = products[:TOTAL_ROWS - len(rows)]
    if not products:
        continue
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    for p in products:
        qty = random.choices([1, 2, 3, 4, 5], weights=[55, 25, 10, 6, 4])[0]
        price = round(p["base_price"] * random.uniform(0.95, 1.05), 2)
        rows.append({
            "transaction_id": txn_id,
            "customer_id": cust,
            "product_name": p["name"],
            "category": p["category"],
            "quantity": qty,
            "price": price,
            "timestamp": ts_str,
            "is_anomaly": 0,
        })

assert len(rows) == TOTAL_ROWS

# ---------------------------------------------------------------- 6. Anomalies
anom_count = int(TOTAL_ROWS * 0.03) + random.randint(-5, 5)
anom_indices = random.sample(range(TOTAL_ROWS), anom_count)
anom_types = ["qty_zero", "qty_neg", "price_zero", "price_neg",
              "price_huge", "qty_huge", "missing_cat"]
for i, idx in enumerate(anom_indices):
    r = rows[idx]
    t = anom_types[i % len(anom_types)]
    if t == "qty_zero":
        r["quantity"] = 0
    elif t == "qty_neg":
        r["quantity"] = -random.randint(1, 3)
    elif t == "price_zero":
        r["price"] = 0.00
    elif t == "price_neg":
        r["price"] = round(-random.uniform(1, 20), 2)
    elif t == "price_huge":
        r["price"] = 9999.99
    elif t == "qty_huge":
        r["quantity"] = 999
    elif t == "missing_cat":
        r["category"] = ""
    r["is_anomaly"] = 1

# ---------------------------------------------------------------- 7. Write CSV
cols = ["transaction_id", "customer_id", "product_name", "category",
        "quantity", "price", "timestamp", "is_anomaly"]
with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# ---------------------------------------------------------------- 8. Summary
n_anom = sum(r["is_anomaly"] for r in rows)
print(f"Rows written       : {len(rows)}")
print(f"Baskets            : {basket_idx}")
print(f"Unique products    : {len(set(r['product_name'] for r in rows))}")
print(f"Unique customers   : {len(set(r['customer_id'] for r in rows))}")
print(f"Anomaly rows       : {n_anom} ({100 * n_anom / len(rows):.2f}%)")
print(f"Date range         : {min(r['timestamp'] for r in rows)} -> "
      f"{max(r['timestamp'] for r in rows)}")
print("\nCategory distribution (rows):")
for k, v in sorted(Counter(r["category"] for r in rows).items(),
                   key=lambda x: -x[1]):
    print(f"  {k:<14} {v}")

# timestamp monotonic check (by basket, since basket shares one timestamp)
seen_ts = []
for r in rows:
    if not seen_ts or seen_ts[-1] != r["timestamp"]:
        seen_ts.append(r["timestamp"])
mono = all(seen_ts[i] <= seen_ts[i + 1] for i in range(len(seen_ts) - 1))
print(f"\nTimestamps monotonic (basket order): {mono}")

# top co-occurring pairs -- evidence the seeded affinities surfaced
by_txn = defaultdict(list)
for r in rows:
    by_txn[r["transaction_id"]].append(r["product_name"])
pair_counter = Counter()
for prods in by_txn.values():
    uniq = sorted(set(prods))
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            pair_counter[(uniq[i], uniq[j])] += 1
print("\nTop 15 co-occurring product pairs (seeded-affinity evidence):")
for (a, b), c in pair_counter.most_common(15):
    print(f"  {c:4d}  {a}  +  {b}")
