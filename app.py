"""
app.py — Supply Chain Disruption Intelligence System (enhanced)
===============================================================
Enhancements over previous version:
  1. UI: better contrast, clear color hierarchy, larger metric fonts, improved cards
  2. Driver view: ORS real route, folium map, distance/duration, congestion estimate
  3. Real-time feature mapping: weather→chaos, congestion→delay, news→fragility
  4. Dynamic SHAP-driven recommendations
  5. Fully dynamic AI narrative (no hard-coded feature names)

Run: streamlit run app.py
"""
from __future__ import annotations
import os, sys, warnings
from datetime import datetime

import joblib, matplotlib, matplotlib.pyplot as plt
import numpy as np, pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils.signals import (
    MODEL_FEATURE_COLUMNS, CITY_COORDS,
    build_feature_vector, compute_congestion, compute_financial_impact,
    fetch_news_risk, fetch_weather, fetch_ors_route, generate_narrative,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="SupplyChain AI", page_icon="🔗",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base layout ── */
section[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #21262d; }
.block-container { padding-top:1.2rem; padding-bottom:2rem; }

/* ── KPI cards — larger, clearer ── */
.kpi-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:18px 20px; text-align:center;
    transition: border-color .2s;
}
.kpi-card:hover { border-color:#58a6ff; }
.kpi-label { font-size:11px; font-weight:600; color:#8b949e;
             text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
.kpi-value { font-size:32px; font-weight:800; color:#f0f6fc; line-height:1.1; }
.kpi-delta-bad  { font-size:13px; color:#f85149; margin-top:4px; font-weight:500; }
.kpi-delta-good { font-size:13px; color:#3fb950; margin-top:4px; font-weight:500; }
.kpi-delta-neu  { font-size:13px; color:#8b949e; margin-top:4px; }

/* ── Risk verdict badge ── */
.badge-high   { display:inline-block; background:#3d1a1a; color:#ff7b72;
                padding:8px 20px; border-radius:8px; font-weight:700;
                font-size:17px; border:1px solid #6e1c1c; letter-spacing:.04em; }
.badge-medium { display:inline-block; background:#3d2e0a; color:#e3b341;
                padding:8px 20px; border-radius:8px; font-weight:700;
                font-size:17px; border:1px solid #6e4f0f; }
.badge-low    { display:inline-block; background:#0d2d1a; color:#3fb950;
                padding:8px 20px; border-radius:8px; font-weight:700;
                font-size:17px; border:1px solid #196c2e; }

/* ── Narrative box ── */
.narrative {
    background:#161b22; border-left:4px solid #388bfd;
    border-radius:0 10px 10px 0; padding:16px 20px;
    font-size:15px; line-height:1.8; color:#e6edf3; margin:14px 0;
}

/* ── News card ── */
.news-item {
    background:#161b22; border-radius:8px;
    padding:11px 15px; margin-bottom:9px;
    font-size:13.5px; color:#c9d1d9; line-height:1.55;
    border:1px solid #21262d;
}
.news-neg { border-left:4px solid #f85149; }
.news-pos { border-left:4px solid #3fb950; }
.news-neu { border-left:4px solid #484f58; }

/* ── Section header ── */
.section-header {
    font-size:15px; font-weight:700; color:#f0f6fc;
    padding-bottom:7px; border-bottom:1px solid #21262d;
    margin:22px 0 14px; letter-spacing:.01em;
}

/* ── Rec card ── */
.rec-card {
    background:#161b22; border-radius:8px;
    padding:11px 15px; margin-bottom:8px;
    font-size:13.5px; color:#c9d1d9; line-height:1.55;
    border:1px solid #21262d;
}
.rec-critical { border-left:4px solid #f85149; }
.rec-high     { border-left:4px solid #ff9800; }
.rec-medium   { border-left:4px solid #e3b341; }
.rec-info     { border-left:4px solid #3fb950; }

/* ── Signal pill ── */
.sig-pill {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:12px; font-weight:600; margin:2px;
}
.sig-red    { background:#3d1a1a; color:#f85149; border:1px solid #6e1c1c; }
.sig-yellow { background:#3d2e0a; color:#e3b341; border:1px solid #6e4f0f; }
.sig-green  { background:#0d2d1a; color:#3fb950; border:1px solid #196c2e; }
.sig-blue   { background:#0d1f3c; color:#58a6ff; border:1px solid #1a4480; }

/* ── Route info card ── */
.route-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:14px 18px; margin-bottom:12px;
}
.route-stat { font-size:22px; font-weight:700; color:#f0f6fc; }
.route-label { font-size:11px; color:#8b949e; text-transform:uppercase;
               font-weight:600; letter-spacing:.06em; }

/* ── Inputs ── */
div[data-testid="stNumberInput"] input { background:#161b22 !important; color:#f0f6fc !important; }
div[data-testid="stTextInput"]   input { background:#161b22 !important; color:#f0f6fc !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    border:1px solid #30363d; background:#161b22; color:#c9d1d9;
    border-radius:8px; font-size:13px; font-weight:500; transition:.2s;
}
div[data-testid="stButton"] > button:hover {
    background:#21262d; border-color:#58a6ff; color:#f0f6fc; }

/* ── Primary button ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#1f6feb,#388bfd);
    border:none; color:#fff; font-weight:700; font-size:15px;
    border-radius:10px; padding:10px 0;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#388bfd,#58a6ff); }

/* ── Metric widget override ── */
[data-testid="metric-container"] { background:#161b22; border:1px solid #21262d;
                                    border-radius:10px; padding:12px 16px; }
[data-testid="stMetricValue"]    { font-size:24px !important; font-weight:700; color:#f0f6fc; }
[data-testid="stMetricLabel"]    { font-size:12px !important; color:#8b949e; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Data & models ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv("data/final_dataset.csv")

@st.cache_resource(show_spinner=False)
def load_ensemble():
    paths = {"xgb":"model/xgb_model.pkl","rf":"model/rf_model.pkl","lr":"model/lr_model.pkl"}
    missing = [k for k,p in paths.items() if not os.path.exists(p)]
    if missing:
        st.error(f"❌ Model files not found: {missing}. Run `python model/train_model.py` first.")
        st.stop()
    return joblib.load(paths["xgb"]), joblib.load(paths["rf"]), joblib.load(paths["lr"])

@st.cache_resource(show_spinner=False)
def load_shap_explainer(_xgb):
    return shap.TreeExplainer(_xgb)

df_full        = load_dataset()
xgb, rf, lr   = load_ensemble()
shap_explainer = load_shap_explainer(xgb)
FEATURE_COLS   = MODEL_FEATURE_COLUMNS


# ── Prediction engine ──────────────────────────────────────────────────────

def run_ensemble(fvec: dict) -> tuple[float, float, float, float]:
    row   = pd.DataFrame([fvec])[FEATURE_COLS]
    p_xgb = float(xgb.predict_proba(row)[0, 1])
    p_rf  = float(rf.predict_proba(row)[0, 1])
    p_lr  = float(lr.predict_proba(row)[0, 1])
    ens   = round((p_xgb + p_rf + p_lr) / 3, 4)
    return round(p_xgb,4), round(p_rf,4), round(p_lr,4), ens

def run_shap(fvec: dict) -> tuple[np.ndarray, list[tuple[str, float]]]:
    row  = pd.DataFrame([fvec])[FEATURE_COLS]
    vals = shap_explainer.shap_values(row)[0]
    pairs = sorted(zip(FEATURE_COLS, vals), key=lambda x: abs(x[1]), reverse=True)
    return vals, list(pairs)


# ── Plotly helpers ─────────────────────────────────────────────────────────

_DL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(color="#c9d1d9", size=12), margin=dict(l=0,r=0,t=30,b=0))

def risk_gauge(prob: float, title: str = "Disruption Risk") -> go.Figure:
    pct   = round(prob * 100, 1)
    color = "#f85149" if prob > 0.7 else "#e3b341" if prob > 0.5 else "#3fb950"
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={"text": title, "font": {"size": 14, "color": "#8b949e"}},
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis":  {"range":[0,100], "tickcolor":"#484f58", "tickwidth":1,
                      "tickfont":{"size":11,"color":"#8b949e"}},
            "bar":   {"color": color, "thickness": 0.28},
            "bgcolor":"#0d1117", "borderwidth":0,
            "steps": [{"range":[0,50],"color":"#0d2d1a"},
                      {"range":[50,75],"color":"#3d2e0a"},
                      {"range":[75,100],"color":"#3d1a1a"}],
            "threshold":{"line":{"color":"#f0f6fc","width":2},"thickness":0.8,"value":70},
        }))
    fig.update_layout(height=250, **_DL)
    return fig

def shap_bar(pairs: list[tuple[str,float]], role: str, n: int = 8) -> go.Figure:
    top    = pairs[:n]
    labels = [p[0].replace("_"," ").title() for p in top]
    vals   = [p[1] for p in top]
    colors = ["#f85149" if v > 0 else "#388bfd" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        textfont=dict(size=11, color="#c9d1d9")))
    fig.update_layout(
        title=dict(text=f"Feature contributions — {role} view",
                   font=dict(size=13, color="#f0f6fc")),
        xaxis=dict(title="SHAP value", zeroline=True, zerolinecolor="#30363d",
                   zerolinewidth=1.5),
        yaxis=dict(autorange="reversed"),
        height=max(280, n*36), **_DL)
    return fig

def model_score_bar(p_xgb, p_rf, p_lr, ens) -> go.Figure:
    models = ["XGBoost","Random Forest","Logistic Reg.","Ensemble"]
    probs  = [p_xgb, p_rf, p_lr, ens]
    ens_c  = "#f85149" if ens > 0.7 else "#e3b341" if ens > 0.5 else "#3fb950"
    colors = ["#6e40c9","#8957e5","#a371f7", ens_c]
    fig = go.Figure(go.Bar(
        x=models, y=probs, marker_color=colors,
        text=[f"{p:.3f}" for p in probs], textposition="outside",
        textfont=dict(size=13, color="#f0f6fc")))
    fig.update_layout(yaxis=dict(range=[0,1.15], tickformat=".0%"), height=230, **_DL)
    return fig

def demand_forecast_bar(surge: float) -> go.Figure:
    weeks = ["Week 1","Week 2","Week 3","Week 4"]
    base  = 1200
    units = [int(base*1.0), int(base*(1+surge*0.8)),
             int(base*(1+surge*1.6)), int(base*(1+surge*0.9))]
    colors = ["#3fb950",
              "#e3b341" if surge > 0.3 else "#3fb950",
              "#f85149" if surge > 0.5 else "#e3b341",
              "#3fb950"]
    fig = go.Figure(go.Bar(x=weeks, y=units, marker_color=colors,
                           text=units, textposition="outside",
                           textfont=dict(size=12, color="#f0f6fc")))
    fig.update_layout(yaxis_title="Units", height=230, **_DL)
    return fig

def news_mini_gauge(risk: float) -> go.Figure:
    c = "#f85149" if risk > 0.6 else "#e3b341" if risk > 0.4 else "#3fb950"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(risk*100,1),
        title={"text":"News Risk Score","font":{"size":12,"color":"#8b949e"}},
        number={"suffix":"%","font":{"size":26,"color":c}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":c},
               "steps":[{"range":[0,40],"color":"#0d2d1a"},
                        {"range":[40,65],"color":"#3d2e0a"},
                        {"range":[65,100],"color":"#3d1a1a"}]}))
    fig.update_layout(height=190, **_DL)
    return fig


# ── Folium map builder (Driver) ────────────────────────────────────────────

def build_folium_map(route, weather_sev: float, cong_idx: float):
    """Build a folium map from a RouteSignal. Returns folium.Map."""
    import folium

    geom = route["geometry"]   # [[lon,lat], ...]
    wpts = route["waypoints"]

    # Centre on midpoint of route
    mid_lat = (wpts[0]["lat"] + wpts[-1]["lat"]) / 2
    mid_lon = (wpts[0]["lon"] + wpts[-1]["lon"]) / 2
    zoom    = 8 if route["distance_km"] > 100 else 10

    risk = (cong_idx + weather_sev) / 2
    tile = "CartoDB dark_matter"
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=zoom,
                   tiles=tile, control_scale=True)

    # Route polyline — colour by risk
    route_color = "#f85149" if risk > 0.6 else "#e3b341" if risk > 0.35 else "#3fb950"
    if len(geom) >= 2:
        coords = [[c[1], c[0]] for c in geom]   # folium wants [lat,lon]
        folium.PolyLine(coords, color=route_color, weight=5,
                        opacity=0.85, tooltip="Route").add_to(m)

    # Waypoint markers
    icon_colors = {"origin":"green","dest.":"red","dest":"red"}
    for i, wp in enumerate(wpts):
        clr = "green" if i == 0 else "red"
        folium.Marker(
            location=[wp["lat"], wp["lon"]],
            popup=folium.Popup(wp["name"], max_width=200),
            tooltip=wp["name"],
            icon=folium.Icon(color=clr, icon="truck" if i==0 else "flag",
                             prefix="fa"),
        ).add_to(m)

    # Weather impact circle at origin
    if weather_sev > 0.3:
        sev_color = "#f85149" if weather_sev > 0.6 else "#e3b341"
        folium.Circle(
            location=[wpts[0]["lat"], wpts[0]["lon"]],
            radius=15_000 * weather_sev,
            color=sev_color, fill=True, fill_opacity=0.15,
            tooltip=f"Weather impact zone (severity {weather_sev:.0%})",
        ).add_to(m)

    # Congestion zone at midpoint
    if cong_idx > 0.4 and len(geom) > 2:
        mid = geom[len(geom)//2]
        cog_color = "#f85149" if cong_idx > 0.65 else "#e3b341"
        folium.Circle(
            location=[mid[1], mid[0]],
            radius=10_000 * cong_idx,
            color=cog_color, fill=True, fill_opacity=0.18,
            tooltip=f"Congestion zone (index {cong_idx:.0%})",
        ).add_to(m)

    return m


# ── KPI card helper ────────────────────────────────────────────────────────

def kpi(col, label: str, value: str, delta: str = "", delta_bad: bool = True):
    cls = "kpi-delta-bad" if delta_bad else "kpi-delta-good" if not delta_bad else "kpi-delta-neu"
    if delta == "":
        cls = "kpi-delta-neu"
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='{cls}'>{delta}</div>"
        f"</div>", unsafe_allow_html=True)


# ── Dynamic SHAP recommendations ──────────────────────────────────────────

def build_recommendations(
    role: str, prob: float, fvec: dict,
    shap_pairs: list[tuple[str, float]],
    weather_sig: dict, congestion_sig: dict,
) -> list[tuple[str, str, str]]:   # [(icon, urgency, text)]

    recs: list[tuple[str, str, str]] = []

    if prob > 0.7:
        recs.append(("🚨", "Critical",
                     "Ensemble probability exceeds 70% — activate contingency protocol immediately."))

    # SHAP-driven: rank positive contributors and give role-appropriate advice
    pos_drivers = [(f, v) for f, v in shap_pairs if v > 0]

    _advice: dict[str, dict[str, tuple[str, str, str]]] = {
        # feature → {role: (icon, urgency, text)}
        "avg_delay_days": {
            "Manager": ("🚚","High", f"Avg delay {fvec['avg_delay_days']:.1f}d — audit top 5 delayed vendors and escalate SLAs."),
            "Driver":  ("⏱️","High", f"Route delay {fvec['avg_delay_days']:.1f}d — notify recipient and recalculate ETA."),
            "Supplier":("📋","Medium","Outbound delays detected — review dispatch schedule and carrier agreements."),
        },
        "supplier_fragility": {
            "Manager": ("🏭","High",  f"Fragility={fvec['supplier_fragility']:.3f} — qualify at least 2 backup suppliers now."),
            "Driver":  ("🏭","Info",  "Supplier reliability flagged — cargo may require extra handling care."),
            "Supplier":("🏭","Medium","Your fragility score is elevated — review capacity utilisation and defect rates."),
        },
        "chaos_score": {
            "Manager": ("🌧️","Medium","Chaos Score elevated — cross-check weather feeds and port congestion dashboards."),
            "Driver":  ("🌧️","High",  "Combined weather/congestion chaos is high — consider delaying departure."),
            "Supplier":("🌧️","Medium","External chaos signals elevated — build buffer into lead time commitments."),
        },
        "on_time_rate": {
            "Manager": ("📅","Medium",f"On-time rate {fvec['on_time_rate']*100:.0f}% — review scheduling and carrier KPIs."),
            "Driver":  ("📅","Medium","Your on-time rate is below target — prioritise this delivery."),
            "Supplier":("📅","Medium","Low on-time rate detected — investigate root causes in production or dispatch."),
        },
        "disruptions_12m": {
            "Manager": ("📊","Medium",f"Disruptions last 12m: {fvec['disruptions_12m']:.0f} — this supplier needs a corrective action plan."),
            "Driver":  ("📊","Info",  f"Route has had {fvec['disruptions_12m']:.0f} past disruptions — stay alert."),
            "Supplier":("📊","Medium",f"{fvec['disruptions_12m']:.0f} disruptions in 12m — identify systemic causes."),
        },
        "inventory_days_remaining": {
            "Manager": ("📦","High",  f"Inventory at {fvec['inventory_days_remaining']:.0f}d — trigger safety-stock reorder immediately."),
            "Driver":  ("📦","Info",  "Inventory at destination may be critically low — verify receiving capacity."),
            "Supplier":("📦","Critical",f"Only {fvec['inventory_days_remaining']:.0f} days of raw material — emergency procurement required."),
        },
        "demand_surge_score": {
            "Manager": ("📈","Medium","Demand surge detected — coordinate with suppliers to prevent stockout."),
            "Driver":  ("📈","Info",  "Demand surge at destination — expect extended unloading times."),
            "Supplier":("📈","High",  f"Surge score {fvec['demand_surge_score']:.2f} — alert sub-suppliers and scale production."),
        },
        "demand_supply_gap": {
            "Manager": ("⚖️","Medium","Demand-supply gap is significant — review order fulfilment capacity."),
            "Driver":  ("⚖️","Info",  "Demand-supply imbalance noted at destination."),
            "Supplier":("⚖️","High",  f"Gap of {fvec['demand_supply_gap']:,.0f} units — plan capacity expansion or sub-contracting."),
        },
    }

    seen = set()
    for feat, _ in pos_drivers:
        if feat in _advice and feat not in seen:
            role_advice = _advice[feat].get(role)
            if role_advice:
                recs.append(role_advice)
                seen.add(feat)
        if len(recs) >= 6:
            break

    # Role-specific live-signal recommendations
    if role == "Driver":
        if weather_sig["severity"] > 0.5:
            recs.append(("🌧️","High",
                         f"Severe {weather_sig['condition']} — reduce speed, check cargo integrity, update dispatch."))
        if congestion_sig["index"] > 0.65:
            recs.append(("🚦","High","Severe congestion on route — switch to alternate corridor immediately."))
        recs.append(("📍","Info", congestion_sig["description"]))

    if not recs:
        recs.append(("✅","Info","All indicators within normal range — continue routine monitoring."))

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
# ── GLOBAL SIDEBAR STYLING ──
st.markdown("""
<style>

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

/* General text (safe — not breaking inputs) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #E5E7EB !important;
}

/* Headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Selectbox (fix visibility) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 8px;
}

/* Selected value */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown menu */
section[data-testid="stSidebar"] div[data-baseweb="select"] ul {
    background-color: #111827 !important;
}

/* Dropdown items */
section[data-testid="stSidebar"] div[data-baseweb="select"] li {
    color: white !important;
}

/* Text inputs */
section[data-testid="stSidebar"] input {
    background-color: #1f2937 !important;
    color: white !important;
    border-radius: 6px;
}

/* Placeholder text */
section[data-testid="stSidebar"] input::placeholder {
    color: #9CA3AF !important;
}

/* Buttons */
section[data-testid="stSidebar"] button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<span style='color:#FFFFFF;font-size:24px'> 🔗 SupplyChain AI</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#FFFFFF;font-size:13px'>Disruption Intelligence Platform</span>",
                unsafe_allow_html=True)
    st.divider()

    role: str = st.selectbox("Your role", ["Manager","Driver","Supplier"],
                             help="Dashboard adapts inputs, model weighting, and outputs to your role.")
    st.divider()
    st.markdown("""
<span style='color:#FFFFFF; font-size:24px; font-weight:600;'>
Supplier / Route City
</span>
""", unsafe_allow_html=True)
    
    city: str = st.selectbox("City",
                             ["Chennai","Mumbai","Delhi","Bangalore","Kolkata","Hyderabad","Pune","Surat"])
    st.divider()
    st.markdown("""
<span style='color:#FFFFFF;font-size:24px; font-weight:600;'>
🔑 API Keys <span style='font-style:italic; font-size:16px; color:#9CA3AF;'>(leave blank for demo mode)</span>
</span>
""", unsafe_allow_html=True)
    owm_key  = st.text_input("OpenWeatherMap", type="password", placeholder="paste key…")
    news_key = st.text_input("NewsAPI",        type="password", placeholder="paste key…")
    ors_key  = st.text_input("OpenRouteService (Driver)", type="password", placeholder="paste key…")
    st.divider()

    @st.cache_data(ttl=300, show_spinner=False)
    def _signals(city, owm_key, news_key):
        return fetch_weather(city, owm_key), compute_congestion(city), fetch_news_risk(news_key)

    weather_sig, congestion_sig, news_sig = _signals(city, owm_key, news_key)

    # ── Sidebar live tiles ──
    lbw = "🟢 Live" if weather_sig["live"] else "🔵 Demo"
    st.markdown(f"**Weather — {city}** {lbw}")
    wc1, wc2 = st.columns(2)
    wc1.metric("Condition", weather_sig["condition"])
    wc2.metric("Temp", f"{weather_sig['temp_c']}°C")
    sp  = round(weather_sig["severity"]*100)
    sc  = "🔴" if sp>60 else "🟡" if sp>30 else "🟢"
    st.caption(f"Severity {sc} {sp}%  ·  Wind {weather_sig['wind_kmh']} km/h  ·  Humidity {weather_sig['humidity']}%")

    st.markdown(f"**Traffic — {city}**")
    cc = "🔴" if congestion_sig["index"]>0.65 else "🟡" if congestion_sig["index"]>0.40 else "🟢"
    st.caption(f"{cc} {congestion_sig['level']} ({round(congestion_sig['index']*100)}%)  ·  {congestion_sig['description']}")

    lbn = "🟢 Live" if news_sig["live"] else "🔵 Demo"
    st.markdown(f"**News Risk** {lbn}")
    nr = news_sig["risk_score"]
    nc = "🔴" if nr>0.6 else "🟡" if nr>0.4 else "🟢"
    st.metric("Score", f"{nc} {nr}")

    st.divider()
    if st.button("🔄 Refresh signals", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

role_icon  = {"Manager":"📊","Driver":"🚛","Supplier":"🏭"}[role]
role_color = {"Manager":"#6e40c9","Driver":"#3fb950","Supplier":"#e3b341"}[role]

st.markdown(
    f"""
    <div style='padding-top:8px; overflow:visible;'>
        <h2 style='
            margin:0;
            font-size:45px;
            line-height:1.4;
        '>
             Supply Chain Disruption Intelligence
            <span style='color:{role_color}'>— {role} View</span>
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<p style='color:#666;font-size:20px;'>Ensemble ML · SHAP Explainability · Real-Time Signals · Role-Aware Predictions</p>", unsafe_allow_html=True)
st.divider()

# ── KPI strip ──────────────────────────────────────────────────────────────

k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1,"Active Suppliers","142","↑ 4 critical",  delta_bad=True)
kpi(k2,"Avg Chaos Score", "0.67","↑ +0.12 WoW",  delta_bad=True)
kpi(k3,"Est. Risk (INR)", "₹18L","30-day window", delta_bad=True)
kpi(k4,"On-Time Rate",    "81%", "↓ −3% vs target",delta_bad=True)
nr_d = delta_bad=news_sig["risk_score"] > 0.5
kpi(k5,"News Risk",f"{news_sig['risk_score']}","live signal", delta_bad=nr_d)
st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)


# ── Scenario simulator ─────────────────────────────────────────────────────

st.markdown(
    "<div style='color:#000000; font-weight:600;' class='section-header'>🔮 What-If Scenario Simulator</div>",
    unsafe_allow_html=True
)
sc1,sc2,sc3,sc4,sc5 = st.columns(5)
if "scenario" not in st.session_state: st.session_state.scenario = None

SCENARIOS = {
    " Cyclone":         dict(avg_delay=9.0, fragility=0.85,chaos=0.95,on_time=0.10,inventory=200.0, disruptions=8, surge=0.9,gap=50000.0),
    " Supplier Fail":   dict(avg_delay=10.0,fragility=1.00,chaos=0.90,on_time=0.05,inventory=150.0, disruptions=10,surge=0.8,gap=80000.0),
    " Fuel +30%":        dict(avg_delay=4.0, fragility=0.60,chaos=0.75,on_time=0.40,inventory=900.0, disruptions=3, surge=0.4,gap=15000.0),
    " Port Congestion":  dict(avg_delay=6.0, fragility=0.50,chaos=0.70,on_time=0.30,inventory=600.0, disruptions=5, surge=0.5,gap=20000.0),
    " All Clear":        dict(avg_delay=0.0, fragility=0.01,chaos=0.00,on_time=1.00,inventory=5000.0,disruptions=0, surge=0.02,gap=0.0),
}
for col,(label,vals) in zip([sc1,sc2,sc3,sc4,sc5], SCENARIOS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.scenario = vals

scen = st.session_state.scenario
if scen:
    st.info("⚡ Scenario active — inputs pre-filled. Adjust if needed, then click **Run Prediction**.")
st.divider()

def _d(key, fallback):
    return scen[key] if scen and key in scen else fallback


# ─────────────────────────────────────────────────────────────────────────────
# ROLE INPUT PANELS
# ─────────────────────────────────────────────────────────────────────────────

route_sig = None   # populated in Driver block

if role == "Manager":
    st.markdown(
    "<div class='section-header' style='color:black;'> Operational Inputs</div>",
    unsafe_allow_html=True
)

    # ── Real-time signal mapping display ──
    smap_c1, smap_c2, smap_c3 = st.columns(3)
    smap_c1.markdown(
        f"<div class='rec-card rec-info'>"
        f"<b>🌧️ Weather → chaos_score</b><br>"
        f"Severity <b>{round(weather_sig['severity']*100)}%</b> adds "
        f"<b>+{round(weather_sig['severity']*0.25,3)}</b> to chaos</div>",
        unsafe_allow_html=True)
    smap_c2.markdown(
        f"<div class='rec-card rec-info'>"
        f"<b>🚦 Congestion → chaos_score</b><br>"
        f"Index <b>{round(congestion_sig['index']*100)}%</b> blends into "
        f"chaos at 35% weight</div>",
        unsafe_allow_html=True)
    news_frag_bump = round(news_sig["risk_score"] * 0.02, 4)
    smap_c3.markdown(
        f"<div class='rec-card rec-info'>"
        f"<b>📰 News → supplier_fragility</b><br>"
        f"News risk <b>{news_sig['risk_score']}</b> adds "
        f"<b>+{news_frag_bump}</b> to fragility</div>",
        unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        avg_delay   = st.number_input("Avg Delay Days",   min_value=0.0,max_value=30.0,value=_d("avg_delay",2.0),step=0.5)
        disruptions = st.number_input("Disruptions (12m)",min_value=0,  max_value=63,  value=int(_d("disruptions",5)),step=1)
        on_time     = st.slider("On-Time Rate", 0.0,1.0, _d("on_time",0.85), step=0.01)
    with col_r:
        fragility    = st.slider("Supplier Fragility",0.0,0.142,_d("fragility",0.05),step=0.005,format="%.3f")
        chaos_input  = st.slider("Chaos Score (raw)",  0.0,1.0, _d("chaos",0.01),   step=0.01)
        avg_order_val= st.number_input("Avg Order Value (₹)",min_value=50_000,max_value=10_000_000,value=500_000,step=50_000)

    weather_sev    = weather_sig["severity"]
    congestion_idx = congestion_sig["index"]
    inventory_days = 2658.0
    demand_surge   = 0.052
    demand_gap     = 0.0


elif role == "Driver":
    st.markdown("<div class='section-header'>🗺️ Route Planning & Delivery Conditions</div>",
                unsafe_allow_html=True)

    # ── ORS route inputs ──
    ri_c1, ri_c2, ri_c3 = st.columns([2,2,1])
    with ri_c1:
        origin_input = st.text_input("Origin (city or address)", value=city,
                                     placeholder="e.g. Chennai or Anna Nagar, Chennai")
    with ri_c2:
        dest_input   = st.text_input("Destination (city or address)", value="Mumbai",
                                     placeholder="e.g. Mumbai or Dharavi, Mumbai")
    with ri_c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        fetch_route  = st.button("🔍 Get Route", use_container_width=True)

    # Session-state route persistence
    if "route_sig" not in st.session_state:
        st.session_state.route_sig = None

    if fetch_route:
        with st.spinner("Fetching route from OpenRouteService…"):
            st.session_state.route_sig = fetch_ors_route(
                origin_input, dest_input, ors_key, congestion_sig["index"])

    route_sig = st.session_state.route_sig

    # ── Live signal tiles ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**📡 Real-Time Signals** *(auto-fetched — fed directly into model)*")
        weather_sev    = weather_sig["severity"]
        congestion_idx = congestion_sig["index"]
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("Weather Severity",  f"{round(weather_sev*100)}%",   weather_sig["condition"])
        mc2.metric("Congestion Index",  f"{round(congestion_idx*100)}%", congestion_sig["level"])
        mc3.metric("Wind Speed",        f"{weather_sig['wind_kmh']} km/h")

        # Real-time mapping pills
        st.markdown(
            f"<span class='sig-pill {'sig-red' if weather_sev>0.5 else 'sig-yellow' if weather_sev>0.2 else 'sig-green'}'>"
            f"Weather→chaos: +{round(weather_sev*0.5,3)}</span>"
            f"<span class='sig-pill {'sig-red' if congestion_idx>0.65 else 'sig-yellow' if congestion_idx>0.4 else 'sig-green'}'>"
            f"Congestion→delay: +{round(congestion_idx*3.0,2)}d</span>",
            unsafe_allow_html=True)

        if route_sig:
            st.markdown("**📏 Route Info**")
            rs1,rs2,rs3 = st.columns(3)
            live_badge = "🟢" if route_sig["live"] else "🔵"
            rs1.metric("Distance", f"{route_sig['distance_km']} km", f"{live_badge} {'Live ORS' if route_sig['live'] else 'Estimate'}")
            rs2.metric("Duration", f"{route_sig['duration_min']:.0f} min")
            rs3.metric("Route Congestion", f"{round(route_sig['congestion_est']*100)}%")
            if route_sig["error"]:
                st.caption(f"ℹ️ {route_sig['error']}")

    with col_r:
        st.markdown("**🚛 Driver Manual Inputs**")
        avg_delay   = st.number_input("Current Route Delay (days)",min_value=0.0,max_value=15.0,
                                       value=float(route_sig["delay_days_est"]) if route_sig else _d("avg_delay",1.0),
                                       step=0.5)
        disruptions = st.number_input("Past Route Disruptions",min_value=0,max_value=20,
                                       value=int(_d("disruptions",2)),step=1)
        on_time     = st.slider("Your On-Time Rate",0.0,1.0,_d("on_time",0.80),step=0.01)
        fragility   = 0.059
        chaos_input = 0.014
        inventory_days = 2658.0; demand_surge=0.052; demand_gap=0.0; avg_order_val=500_000

    # ── Folium map ──
    st.markdown("<div class='section-header'>🗺️ Route Risk Map</div>", unsafe_allow_html=True)

    if route_sig:
        try:
            from streamlit_folium import st_folium
            fmap = build_folium_map(route_sig, weather_sig["severity"], congestion_sig["index"])
            st_folium(fmap, use_container_width=True, height=380, returned_objects=[])
        except ImportError:
            st.info("Install streamlit-folium for interactive map: `pip install streamlit-folium folium`")
            # Plotly fallback
            lat0 = (route_sig["waypoints"][0]["lat"] + route_sig["waypoints"][-1]["lat"]) / 2
            lon0 = (route_sig["waypoints"][0]["lon"] + route_sig["waypoints"][-1]["lon"]) / 2
            risk = (congestion_idx + weather_sev) / 2
            zc   = "red" if risk>0.6 else "orange" if risk>0.35 else "green"
            lats = [c[1] for c in route_sig["geometry"]]
            lons = [c[0] for c in route_sig["geometry"]]
            fig  = go.Figure()
            fig.add_trace(go.Scattermapbox(lat=lats,lon=lons,mode="lines",
                                           line=dict(width=4,color=zc),name="Route"))
            fig.add_trace(go.Scattermapbox(
                lat=[w["lat"] for w in route_sig["waypoints"]],
                lon=[w["lon"] for w in route_sig["waypoints"]],
                mode="markers+text",
                marker=dict(size=[16,16],color=["#3fb950","#f85149"]),
                text=[w["name"] for w in route_sig["waypoints"]],
                textposition="top right"))
            fig.update_layout(mapbox=dict(style="carto-darkmatter",center=dict(lat=lat0,lon=lon0),zoom=6),
                              height=360, **_DL)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    else:
        # No route yet — show city-centred placeholder map
        lat0, lon0 = CITY_COORDS.get(city, (13.0827, 80.2707))
        try:
            from streamlit_folium import st_folium
            import folium
            fm = folium.Map(location=[lat0,lon0], zoom_start=10, tiles="CartoDB dark_matter")
            folium.Marker([lat0,lon0], tooltip=f"{city} (origin)",
                          icon=folium.Icon(color="green",icon="truck",prefix="fa")).add_to(fm)
            st_folium(fm, use_container_width=True, height=340, returned_objects=[])
        except ImportError:
            st.info("Enter origin & destination above, then click **Get Route** to see the interactive map.")

    # ── Route alerts ──
    st.markdown("<div class='section-header'>⚠️ Live Route Alerts</div>", unsafe_allow_html=True)
    al1, al2 = st.columns(2)
    with al1:
        if weather_sig["severity"] > 0.5:
            st.error(f"🌧️ Severe {weather_sig['condition']} — reduce speed, check cargo integrity")
        elif weather_sig["severity"] > 0.2:
            st.warning(f"🌤️ Moderate {weather_sig['condition']} — proceed with caution")
        else:
            st.success(f"☀️ {weather_sig['condition']} — normal conditions")
    with al2:
        if congestion_sig["index"] > 0.65:
            st.error(f"🚦 {congestion_sig['level']} congestion — reroute via alternate corridor")
        elif congestion_sig["index"] > 0.40:
            st.warning(f"🚦 {congestion_sig['level']} congestion — expect delays on main route")
        else:
            st.success(f"🚦 {congestion_sig['level']} traffic — clear to proceed")


elif role == "Supplier":
    st.markdown("<div class='section-header'>🏭 Supplier Health Inputs</div>", unsafe_allow_html=True)

    # Signal mapping display
    sm1,sm2 = st.columns(2)
    sm1.markdown(
        f"<div class='rec-card rec-info'>"
        f"<b>📰 News Risk → supplier_fragility</b><br>"
        f"Adds <b>+{round(news_sig['risk_score']*0.02,4)}</b> to your fragility input</div>",
        unsafe_allow_html=True)
    sm2.markdown(
        f"<div class='rec-card rec-info'>"
        f"<b>🌧️ Weather → chaos component</b><br>"
        f"Severity {round(weather_sig['severity']*100)}% contributes to chaos score</div>",
        unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fragility    = st.slider("Your Fragility Score",   0.0,0.142,_d("fragility",0.05),step=0.005,format="%.3f")
        disruptions  = st.number_input("Your Disruptions (12m)",min_value=0,max_value=63,value=int(_d("disruptions",3)),step=1)
        avg_delay    = st.number_input("Your Avg Delay (days)",min_value=0.0,max_value=30.0,value=_d("avg_delay",1.5),step=0.5)
    with col_r:
        demand_surge  = st.slider("Demand Surge Score",    0.0,1.0,_d("surge",0.05),step=0.01)
        inventory_days= st.number_input("Raw Material (days stock)",min_value=0.0,max_value=51500.0,value=_d("inventory",1000.0),step=50.0)
        demand_gap    = st.number_input("Demand-Supply Gap (units)",min_value=-50000.0,max_value=500000.0,value=_d("gap",0.0),step=1000.0)

    on_time=0.922; chaos_input=0.014
    weather_sev=weather_sig["severity"]; congestion_idx=congestion_sig["index"]; avg_order_val=500_000

    st.markdown("<div class='section-header'>📈 Demand Forecast — Next 4 Weeks</div>", unsafe_allow_html=True)
    st.plotly_chart(demand_forecast_bar(demand_surge), use_container_width=True, config={"displayModeBar":False})
    cap = 2000; w3 = int(1200*(1+demand_surge*1.6))
    if w3 > cap:
        st.warning(f"⚠️ Week 3 forecast ({w3:,} units) exceeds capacity ({cap:,}). Activate sub-supplier or notify buyers early.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────

_, pred_col, _ = st.columns([3,2,3])
with pred_col:
    predict = st.button("🚀 Run Prediction", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

if predict:

    # Build role-specific feature vector with full real-time signal mapping
    fvec = build_feature_vector(
        role,
        avg_delay_days          = avg_delay,
        on_time_rate            = on_time,
        supplier_fragility      = fragility,
        disruptions_12m         = float(disruptions),
        chaos_score             = chaos_input,
        inventory_days_remaining= inventory_days,
        demand_surge_score      = demand_surge,
        demand_supply_gap       = demand_gap,
        weather_severity        = weather_sev,
        congestion_index        = congestion_idx,
        news_risk_score         = news_sig["risk_score"],
        route_congestion        = route_sig["congestion_est"] if route_sig else -1.0,
        route_delay_days        = route_sig["delay_days_est"]  if route_sig else -1.0,
    )

    p_xgb, p_rf, p_lr, prob = run_ensemble(fvec)
    shap_vals, shap_pairs    = run_shap(fvec)
    narrative                = generate_narrative(role, prob, fvec, shap_pairs)

    st.divider()

    risk_label = "HIGH RISK"   if prob>0.7 else "MODERATE RISK" if prob>0.5 else "LOW RISK"
    badge_cls  = "badge-high"  if prob>0.7 else "badge-medium"  if prob>0.5 else "badge-low"

    # ── Gauge + model scores ──
    st.markdown(
    "<div class='section-header' style='color:black;'> Ensemble Prediction Results</div>",
    unsafe_allow_html=True
)
    res_l, res_r = st.columns([1,1])

    with res_l:
        st.plotly_chart(risk_gauge(prob, f"{role} Disruption Risk"),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"<div style='text-align:center;margin-top:-8px'>"
                    f"<span class='{badge_cls}'>{risk_label}</span></div>",
                    unsafe_allow_html=True)

    with res_r:
        st.markdown("**🧠 Individual Model Scores**")
        st.plotly_chart(model_score_bar(p_xgb,p_rf,p_lr,prob),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown("**💬 AI Narrative**")
        st.markdown(f"<div class='narrative'>{narrative}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Financial impact ──
    st.markdown("""
<style>

/* Metric label (Delay Cost, etc.) */
[data-testid="stMetricLabel"] {
    color: #374151 !important;  /* dark grey */
    font-weight: 500;
}

/* Metric value (₹36,000 etc.) */
[data-testid="stMetricValue"] {
    color: black !important;
    font-weight: 700;
}

/* Delta (30-day horizon) */
[data-testid="stMetricDelta"] {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
    if role in ("Manager","Supplier"):
        st.markdown(
    "<div class='section-header' style='color:black;'>Financial Impact Estimate (30-day)</div>",
    unsafe_allow_html=True
)
        impact = compute_financial_impact(
            prob, avg_delay, inventory_days,
            avg_order_val if role=="Manager" else 500_000)
        fi1,fi2,fi3,fi4 = st.columns(4)
        fi1.metric("Delay Cost",     f"₹{impact['delay_cost_inr']:,}")
        fi2.metric("Stockout Risk",  f"₹{impact['stockout_cost_inr']:,}")
        fi3.metric("Order at Risk",  f"₹{impact['order_risk_inr']:,}")
        fi4.metric("Total Exposure", f"₹{impact['total_inr']:,}",
                   delta="30-day horizon", delta_color="inverse")
        st.divider()

    # ── SHAP explainability ──
    st.markdown(
    f"<div class='section-header' style='color:black;'> SHAP Explainability — {role} View</div>",
    unsafe_allow_html=True
)
    xp_l, xp_r = st.columns([1,1])

    with xp_l:
        role_features = {
            "Manager":  ["avg_delay_days","supplier_fragility","on_time_rate","disruptions_12m","chaos_score"],
            "Driver":   ["chaos_score","avg_delay_days","on_time_rate","disruptions_12m","delivery_weekday"],
            "Supplier": ["supplier_fragility","demand_surge_score","inventory_days_remaining","demand_supply_gap","disruptions_12m"],
        }
        role_pairs = [(f,v) for f,v in shap_pairs if f in role_features[role]]
        others     = [(f,v) for f,v in shap_pairs if f not in role_features[role]]
        combined   = (role_pairs + others)[:8]
        st.plotly_chart(shap_bar(combined, role), use_container_width=True, config={"displayModeBar":False})
        st.caption("🔴 Red = increases risk &nbsp;·&nbsp; 🔵 Blue = decreases risk")

    with xp_r:
        st.markdown("**Waterfall — single prediction breakdown**")
        row      = pd.DataFrame([fvec])[FEATURE_COLS]
        expl_obj = shap_explainer(row)
        fig_wf, ax_wf = plt.subplots(figsize=(6,4.5))
        fig_wf.patch.set_facecolor("#0d1117")
        ax_wf.set_facecolor("#0d1117")
        shap.plots.waterfall(expl_obj[0], max_display=8, show=False)
        plt.rcParams.update({"text.color":"#c9d1d9","axes.labelcolor":"#c9d1d9",
                              "xtick.color":"#8b949e","ytick.color":"#8b949e"})
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

    st.divider()

    # ── Dynamic SHAP-driven recommendations ──
    st.markdown(
    "<div class='section-header' style='color:black; font-weight:600; font-size:20px;'> Recommendations</div>",
    unsafe_allow_html=True
)
    recs = build_recommendations(role, prob, fvec, shap_pairs, weather_sig, congestion_sig)
    urgency_cls = {"Critical":"rec-critical","High":"rec-high","Medium":"rec-medium","Info":"rec-info"}
    urgency_dot = {"Critical":"🔴","High":"🟠","Medium":"🟡","Info":"🟢"}

    for icon, urgency, text in recs:
        dot = urgency_dot.get(urgency,"⚪")
        st.markdown(
            f"<div class='rec-card {urgency_cls.get(urgency,'rec-info')}'>"
            f"{dot} <b>{urgency}</b> &nbsp; {icon} &nbsp; {text}</div>",
            unsafe_allow_html=True)

    st.divider()

    # ── News feed ──
    st.markdown("<div class='section-header'>📰 Live News Risk Feed</div>", unsafe_allow_html=True)
    nl, nr_col = st.columns([2,1])
    with nl:
        for item in news_sig["headlines"]:
            sc  = "news-neg" if item["score"]<-0.1 else "news-pos" if item["score"]>0.1 else "news-neu"
            ico = "🔴" if item["score"]<-0.1 else "🟢" if item["score"]>0.1 else "⚪"
            st.markdown(
                f"<div class='news-item {sc}'>{ico} {item['title']}"
                f"<span style='float:right;color:#484f58;font-size:11px'>{item['score']:+.2f}</span></div>",
                unsafe_allow_html=True)
    with nr_col:
        st.plotly_chart(news_mini_gauge(news_sig["risk_score"]),
                        use_container_width=True, config={"displayModeBar":False})

    st.divider()

    # ── Debug expander ──
    with st.expander("🔧 Feature vector fed to model (debug)", expanded=False):
        st.json(fvec)
        st.caption(f"XGB={p_xgb:.4f}  RF={p_rf:.4f}  LR={p_lr:.4f}  → Ensemble={prob:.4f}")
        if route_sig:
            st.caption(f"Route: {route_sig['distance_km']} km  ·  {route_sig['duration_min']:.0f} min  "
                       f"·  cong={route_sig['congestion_est']}  ·  live={route_sig['live']}")

else:
    st.markdown(
        "<div style='text-align:center;padding:50px;color:#484f58;font-size:15px'>"
        "⬆️ Configure inputs above and click <b>Run Prediction</b></div>",
        unsafe_allow_html=True)
