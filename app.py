import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Supply Chain AI", layout="wide")

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
# RECOMMENDATION ENGINE
# ----------------------------
def get_recommendations(input_data, prob):
    recs = []

    delay = input_data["avg_delay_days"].values[0]
    fragility = input_data["supplier_fragility"].values[0]
    chaos = input_data["chaos_score"].values[0]
    on_time = input_data["on_time_rate"].values[0]

    if prob > 0.7:
        recs.append("High disruption risk detected — take immediate action")

    if delay > 3:
        recs.append("Reduce delays by optimizing logistics routes")

    if fragility > 0.7:
        recs.append("Consider switching to a more reliable supplier")

    if chaos > 0.6:
        recs.append("Monitor external risks like weather and congestion")

    if on_time < 0.5:
        recs.append("Improve delivery scheduling and planning")

    return recs

# ----------------------------
# UI HEADER
# ----------------------------
st.title("🚚 Supply Chain Disruption Predictor")
st.markdown("### 🔍 Predict shipment risks with explainable AI")

st.divider()

# ----------------------------
# INPUT SECTION (SLIDERS)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    avg_delay_days = st.slider("Avg Delay Days", 0.0, 10.0, 2.0)
    supplier_fragility = st.slider("Supplier Fragility", 0.0, 1.0, 0.5)

with col2:
    on_time_rate = st.slider("On Time Rate", 0.0, 1.0, 0.5)
    chaos_score = st.slider("Chaos Score", 0.0, 1.0, 0.5)

st.divider()

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
input_data = input_data.reindex(columns=X.columns)

# ----------------------------
# CENTERED BUTTON
# ----------------------------
center = st.columns([2, 1, 2])
with center[1]:
    predict_btn = st.button("🚀 Predict", use_container_width=True)

# ----------------------------
# PREDICTION
# ----------------------------
if predict_btn:
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()

    # ----------------------------
    # CONFIDENCE DISPLAY
    # ----------------------------
    st.subheader("📊 Prediction Confidence")
    st.metric("Confidence Score", f"{prob:.2f}")
    st.progress(float(prob))

    # ----------------------------
    # RISK LEVEL
    # ----------------------------
    if prob > 0.8:
        st.error("🔴 HIGH RISK")
    elif prob > 0.5:
        st.warning("🟠 MEDIUM RISK")
    else:
        st.success("🟢 LOW RISK")

    # ----------------------------
    # RESULT
    # ----------------------------
    if pred == 1:
        st.error(f"🚨 Disruption Risk! Probability: {prob:.2f}")
    else:
        st.success(f"✅ No Disruption. Probability: {prob:.2f}")

    # ----------------------------
    # INPUT SUMMARY
    # ----------------------------
    st.subheader("📋 Input Summary")
    st.write(input_data)

    # ----------------------------
    # RECOMMENDATIONS
    # ----------------------------
    st.subheader("💡 Recommendations")

    recs = get_recommendations(input_data, prob)

    for r in recs:
        st.write("➡️", r)

    # ----------------------------
    # SHAP EXPLANATION
    # ----------------------------
    st.subheader("🔍 Why this prediction?")
    st.caption("🔴 Red = increases risk | 🔵 Blue = decreases risk")

    st.warning(" Model predictions depend on training data patterns.")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Adjust inputs and click Predict ")