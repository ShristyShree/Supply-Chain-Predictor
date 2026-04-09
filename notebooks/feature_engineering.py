import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../data/Supply_Chain_Shipment_Data.csv")

date_cols = [
    "pq first sent to client date",
    "po sent to vendor date",
    "scheduled delivery date",
    "delivered to client date"
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

numeric_cols = [
    "freight cost (usd)",
    "line item insurance (usd)",
    "line item quantity"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()
df = df.sort_values("scheduled delivery date")

df["delay_days"] = (
    df["delivered to client date"] - df["scheduled delivery date"]
).dt.days
df["delay_days"] = df["delay_days"].clip(lower=0)

df["on_time"] = (df["delay_days"] == 0).astype(int)

df["delivery_month"] = df["scheduled delivery date"].dt.month
df["delivery_day"] = df["scheduled delivery date"].dt.day
df["delivery_weekday"] = df["scheduled delivery date"].dt.weekday
df["is_weekend"] = df["delivery_weekday"].isin([5, 6]).astype(int)
df["month_end"] = (df["delivery_day"] > 25).astype(int)

scaler = MinMaxScaler()
df[["freight cost (usd)", "line item insurance (usd)"]] = scaler.fit_transform(
    df[["freight cost (usd)", "line item insurance (usd)"]]
)

df["delay_lag_1"] = df["delay_days"].shift(1)
df["delay_lag_7"] = df["delay_days"].shift(7)

df["supplier_fragility"] = df.groupby("vendor")["delay_days"] \
    .expanding().apply(lambda x: (x > 2).mean()) \
    .shift().reset_index(level=0, drop=True)

df["on_time_rate"] = df.groupby("vendor")["on_time"] \
    .expanding().mean().shift().reset_index(level=0, drop=True)

df["disruptions_12m"] = df.groupby("vendor")["delay_days"] \
    .expanding().apply(lambda x: (x > 2).sum()) \
    .shift().reset_index(level=0, drop=True)

df["inventory_days_remaining"] = df["line item quantity"] / 10
df["demand_surge_score"] = df["line item quantity"] / df["line item quantity"].max()
df["demand_supply_gap"] = df["line item quantity"] - df["line item quantity"].mean()

df["chaos_score"] = (
    0.5 * df["freight cost (usd)"] +
    0.5 * df["line item insurance (usd)"]
)

df["delay_trend"] = df["delay_days"].rolling(5).mean()
df["delay_change"] = df["delay_days"].diff()
df["vendor_delay_trend"] = df.groupby("vendor")["delay_days"].transform(lambda x: x.rolling(5).mean())
df["on_time_trend"] = df["on_time_rate"].diff()
df["chaos_trend"] = df["chaos_score"].diff()

df["delivery_urgency"] = (
    (df["scheduled delivery date"] - df["po sent to vendor date"]).dt.days
).clip(lower=0)

df["vendor_risk_score"] = df.groupby("vendor")["delay_days"] \
    .transform(lambda x: x.expanding().mean()).shift()

df["orders_last_7"] = df.groupby("vendor")["delay_days"].transform(lambda x: x.rolling(7).count())
df["demand_volatility"] = df["line item quantity"].rolling(5).std()
df["delay_std"] = df["delay_days"].rolling(5).std()

df["high_risk_flag"] = (
    (df["supplier_fragility"] > 0.3) &
    (df["demand_surge_score"] > 0.5)
).astype(int)

df["inventory_risk"] = (df["inventory_days_remaining"] < 50).astype(int)
df["chaos_flag"] = (df["chaos_score"] > 0.7).astype(int)

df["target"] = (
    (df["delay_days"] > 3) |
    (df["high_risk_flag"] == 1) |
    (df["chaos_flag"] == 1) |
    (df["inventory_risk"] == 1)
).astype(int)

df["target"] = df["target"].shift(-7)

df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

final_df = df[[
    "delay_lag_1","delay_lag_7","disruptions_12m","on_time_rate",
    "inventory_days_remaining","chaos_score","demand_surge_score",
    "supplier_fragility","demand_supply_gap","delivery_month",
    "delivery_weekday","is_weekend","month_end","delay_trend",
    "delay_change","vendor_delay_trend","on_time_trend","chaos_trend",
    "delivery_urgency","vendor_risk_score","orders_last_7",
    "demand_volatility","delay_std","high_risk_flag",
    "inventory_risk","chaos_flag","target"
]]

final_df.to_csv("../data/final_dataset.csv", index=False)

print("✅ FINAL DATASET READY")
print(final_df["target"].value_counts())