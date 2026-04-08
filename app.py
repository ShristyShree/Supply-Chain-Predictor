import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("data/final_dataset.csv")

X = df.drop("will_disrupt_in_next_7_days", axis=1)
y = df["will_disrupt_in_next_7_days"]

# Train model
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
model.fit(X, y)

# ----------------------------
# UI
# ----------------------------
st.title("🚚 Supply Chain Disruption Predictor")

st.write("Enter shipment details:")

# Inputs (with keys)
avg_delay_days = st.number_input("Avg Delay Days", 0.0, key="delay")
supplier_fragility = st.number_input("Supplier Fragility", 0.0, key="fragility")
on_time_rate = st.number_input("On Time Rate", 0.0, key="on_time")
chaos_score = st.number_input("Chaos Score", 0.0, key="chaos")

# ----------------------------
# CREATE INPUT DATA
# ----------------------------
input_dict = {
    "avg_delay_days": avg_delay_days,
    "disruptions_12m": 0,
    "on_time_rate": on_time_rate,
    "inventory_days_remaining": 0,
    "chaos_score": chaos_score,
    "demand_surge_score": 0,
    "supplier_fragility": supplier_fragility,
    "demand_supply_gap": 0,
    "delivery_month": 1,
    "delivery_weekday": 1,
    "is_weekend": 0,
    "month_end": 0
}

input_data = pd.DataFrame([input_dict])

# Match training columns
input_data = input_data.reindex(columns=X.columns)

# ----------------------------
# PREDICT BUTTON (WITH KEY)
# ----------------------------
if st.button("Predict", key="predict_btn"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"🚨 Disruption Risk! Probability: {prob:.2f}")
    else:
        st.success(f"✅ No Disruption. Probability: {prob:.2f}")

    # ----------------------------
    # SHAP EXPLANATION (ALL INSIDE)
    # ----------------------------
    st.subheader("🔍 Why this prediction?")
    st.write("Red → increases risk | Blue → decreases risk")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
    plt.close(fig)