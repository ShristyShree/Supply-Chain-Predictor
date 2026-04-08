import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("../data/Supply_Chain_Shipment_Data.csv")

# ----------------------------
# DATE CONVERSION
# ----------------------------
date_cols = [
    "pq first sent to client date",
    "po sent to vendor date",
    "scheduled delivery date",
    "delivered to client date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# ----------------------------
# NUMERIC CLEANING
# ----------------------------
numeric_cols = [
    "freight cost (usd)",
    "line item insurance (usd)",
    "line item quantity"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# ----------------------------
# SORT BY TIME (IMPORTANT)
# ----------------------------
df = df.sort_values("scheduled delivery date")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------

# 1. Delay
df["avg_delay_days"] = (
    df["delivered to client date"] - df["scheduled delivery date"]
).dt.days

df["avg_delay_days"] = df["avg_delay_days"].clip(lower=0)

# 2. On-time flag
df["on_time"] = (df["avg_delay_days"] == 0).astype(int)

# ----------------------------
# TIME-BASED FEATURES
# ----------------------------
df["delivery_month"] = df["scheduled delivery date"].dt.month
df["delivery_day"] = df["scheduled delivery date"].dt.day
df["delivery_weekday"] = df["scheduled delivery date"].dt.weekday

df["is_weekend"] = df["delivery_weekday"].isin([5, 6]).astype(int)
df["month_end"] = (df["delivery_day"] > 25).astype(int)

# ----------------------------
# NORMALIZATION (IMPORTANT)
# ----------------------------
scaler = MinMaxScaler()
df[["freight cost (usd)", "line item insurance (usd)"]] = scaler.fit_transform(
    df[["freight cost (usd)", "line item insurance (usd)"]]
)

# ----------------------------
# NO DATA LEAKAGE FEATURES
# ----------------------------

# Supplier fragility (past avg delay)
df["supplier_fragility"] = df.groupby("vendor")["avg_delay_days"] \
    .expanding().mean().shift().reset_index(level=0, drop=True)

# On-time rate (past)
df["on_time_rate"] = df.groupby("vendor")["on_time"] \
    .expanding().mean().shift().reset_index(level=0, drop=True)

# Disruptions (past count)
df["disruptions_12m"] = df.groupby("vendor")["avg_delay_days"] \
    .expanding().apply(lambda x: (x > 2).sum()).shift().reset_index(level=0, drop=True)

# Fill initial NaNs (first rows per vendor)
df[["supplier_fragility", "on_time_rate", "disruptions_12m"]] = df[
    ["supplier_fragility", "on_time_rate", "disruptions_12m"]
].fillna(0)

# ----------------------------
# OTHER FEATURES
# ----------------------------

df["inventory_days_remaining"] = df["line item quantity"] / 10

df["demand_surge_score"] = (
    df["line item quantity"] / df["line item quantity"].max()
)

df["demand_supply_gap"] = (
    df["line item quantity"] - df["line item quantity"].mean()
)

# Chaos score (normalized inputs)
df["chaos_score"] = (
    0.4 * df["avg_delay_days"] +
    0.3 * df["freight cost (usd)"] +
    0.3 * df["line item insurance (usd)"]
)

df["chaos_score"] = df["chaos_score"] / df["chaos_score"].max()

# ----------------------------
# TARGET VARIABLE
# ----------------------------

risk = (
    0.4 * df["avg_delay_days"] +
    0.2 * df["chaos_score"] +
    0.2 * df["supplier_fragility"] +
    0.2 * df["demand_surge_score"]
)

threshold = risk.quantile(0.75)

df["will_disrupt_in_next_7_days"] = (risk > threshold).astype(int)

# ----------------------------
# FINAL DATASET
# ----------------------------

final_df = df[[
    "avg_delay_days",
    "disruptions_12m",
    "on_time_rate",
    "inventory_days_remaining",
    "chaos_score",
    "demand_surge_score",
    "supplier_fragility",
    "demand_supply_gap",
    "delivery_month",
    "delivery_weekday",
    "is_weekend",
    "month_end",
    "will_disrupt_in_next_7_days"
]]

# Save
final_df.to_csv("../data/final_dataset.csv", index=False)

print("✅ Final dataset created!")
print(final_df.head())
print("\nClass Distribution:\n", final_df["will_disrupt_in_next_7_days"].value_counts())