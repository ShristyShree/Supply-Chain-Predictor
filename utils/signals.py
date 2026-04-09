"""
utils/signals.py
----------------
Real-world signal fetchers for the Supply Chain AI platform.
Each function returns a typed dict so callers never touch raw API responses.

API keys are injected at call time (from Streamlit secrets or env vars).
All functions have a structured fallback when keys are absent or APIs fail,
so the app never breaks in demo / offline mode.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from typing import TypedDict

import requests


# ─────────────────────────────────────────────────────────────────────────────
# TYPE CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class WeatherSignal(TypedDict):
    condition: str          # e.g. "Rain"
    severity: float         # 0.0 – 1.0
    temp_c: float
    humidity: int
    wind_kmh: float
    city: str
    live: bool              # True = fetched from API, False = fallback


class CongestionSignal(TypedDict):
    index: float            # 0.0 – 1.0
    level: str              # "Low" | "Moderate" | "High" | "Severe"
    peak_hours: bool
    description: str


class NewsSignal(TypedDict):
    risk_score: float       # 0.0 – 1.0
    headlines: list[dict]   # [{"title": str, "score": float}]
    live: bool


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────────────────────────────────────

# Maps OWM condition → disruption severity (domain knowledge)
_SEVERITY_MAP: dict[str, float] = {
    "Clear": 0.0, "Clouds": 0.1, "Mist": 0.2, "Haze": 0.2,
    "Smoke": 0.3, "Drizzle": 0.35, "Fog": 0.4, "Dust": 0.45,
    "Rain": 0.55, "Snow": 0.65, "Squall": 0.70,
    "Thunderstorm": 0.85, "Tornado": 1.0,
}

# Realistic fallbacks per major Indian logistics cities
_CITY_FALLBACKS: dict[str, WeatherSignal] = {
    "Chennai":   {"condition": "Rain",        "severity": 0.55, "temp_c": 32.0, "humidity": 85, "wind_kmh": 18.0, "city": "Chennai",   "live": False},
    "Mumbai":    {"condition": "Clouds",       "severity": 0.10, "temp_c": 30.0, "humidity": 78, "wind_kmh": 14.0, "city": "Mumbai",    "live": False},
    "Delhi":     {"condition": "Haze",         "severity": 0.20, "temp_c": 28.0, "humidity": 60, "wind_kmh": 10.0, "city": "Delhi",     "live": False},
    "Bangalore": {"condition": "Clear",        "severity": 0.00, "temp_c": 25.0, "humidity": 55, "wind_kmh": 8.0,  "city": "Bangalore", "live": False},
    "Kolkata":   {"condition": "Thunderstorm", "severity": 0.85, "temp_c": 31.0, "humidity": 90, "wind_kmh": 25.0, "city": "Kolkata",   "live": False},
    "Hyderabad": {"condition": "Clouds",       "severity": 0.10, "temp_c": 29.0, "humidity": 65, "wind_kmh": 12.0, "city": "Hyderabad", "live": False},
    "Pune":      {"condition": "Clear",        "severity": 0.00, "temp_c": 26.0, "humidity": 50, "wind_kmh": 9.0,  "city": "Pune",      "live": False},
    "Surat":     {"condition": "Clouds",       "severity": 0.10, "temp_c": 31.0, "humidity": 72, "wind_kmh": 15.0, "city": "Surat",     "live": False},
}


def fetch_weather(city: str, api_key: str = "") -> WeatherSignal:
    """
    Fetch live weather from OpenWeatherMap.
    Falls back to city-specific realistic defaults when key is absent/API fails.
    """
    fallback = _CITY_FALLBACKS.get(city, _CITY_FALLBACKS["Chennai"])

    if not api_key:
        return fallback

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city},IN&appid={api_key}&units=metric&lang=en"
        )
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        condition = data["weather"][0]["main"]
        severity  = _SEVERITY_MAP.get(condition, 0.25)

        # Rain intensity boost
        rain_1h = data.get("rain", {}).get("1h", 0.0)
        if rain_1h > 20:
            severity = min(1.0, severity + 0.15)

        wind_ms  = data["wind"]["speed"]  # m/s
        wind_kmh = round(wind_ms * 3.6, 1)
        # Strong wind boost
        if wind_kmh > 60:
            severity = min(1.0, severity + 0.10)

        return WeatherSignal(
            condition=condition,
            severity=round(severity, 3),
            temp_c=round(data["main"]["temp"], 1),
            humidity=data["main"]["humidity"],
            wind_kmh=wind_kmh,
            city=city,
            live=True,
        )

    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# CONGESTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_congestion(city: str, hour: int | None = None) -> CongestionSignal:
    """
    Time-of-day congestion model calibrated to Indian metro patterns.
    hour=None uses current local hour. Replace with a real traffic API
    (Google Maps Distance Matrix, TomTom, HERE) by swapping the internals.
    """
    if hour is None:
        hour = datetime.now().hour

    # Morning peak: 8–10, Evening peak: 17–20, Night low: 23–5
    base_pattern = {
        0: 0.05, 1: 0.04, 2: 0.04, 3: 0.04, 4: 0.05, 5: 0.08,
        6: 0.25, 7: 0.55, 8: 0.85, 9: 0.90, 10: 0.70, 11: 0.55,
        12: 0.50, 13: 0.48, 14: 0.45, 15: 0.52, 16: 0.65, 17: 0.88,
        18: 0.92, 19: 0.85, 20: 0.65, 21: 0.40, 22: 0.20, 23: 0.10,
    }

    # City multipliers (metros have higher baseline congestion)
    city_mult: dict[str, float] = {
        "Mumbai": 1.20, "Delhi": 1.15, "Chennai": 1.05,
        "Bangalore": 1.18, "Kolkata": 1.10, "Hyderabad": 1.08,
        "Pune": 1.00, "Surat": 0.92,
    }

    raw      = base_pattern.get(hour, 0.5)
    mult     = city_mult.get(city, 1.0)
    index    = round(min(1.0, raw * mult), 3)
    peak     = index > 0.65
    is_weekend = datetime.now().weekday() >= 5
    if is_weekend:
        index = round(index * 0.55, 3)  # weekends ~45% lower congestion

    if index < 0.25:
        level, desc = "Low",      "Roads clear — minimal delay expected"
    elif index < 0.50:
        level, desc = "Moderate", "Normal traffic — minor slowdowns possible"
    elif index < 0.75:
        level, desc = "High",     "Heavy congestion — expect significant delays"
    else:
        level, desc = "Severe",   "Gridlock conditions — reroute immediately"

    return CongestionSignal(
        index=index,
        level=level,
        peak_hours=peak,
        description=desc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_HEADLINES: list[dict] = [
    {"title": "Cyclone warning issued for Bay of Bengal coastline", "score": -0.82},
    {"title": "Chennai port faces congestion amid surge in imports", "score": -0.55},
    {"title": "Fuel prices hiked by ₹2/litre — logistics cost impact expected", "score": -0.48},
    {"title": "New trade corridor opens between India and UAE", "score": 0.62},
    {"title": "Freight demand up 12% — capacity crunch likely next quarter", "score": -0.35},
]


def fetch_news_risk(api_key: str = "", query: str = "supply chain India disruption") -> NewsSignal:
    """
    Fetch news headlines and compute sentiment-based risk score.
    Requires: pip install vaderSentiment newsapi-python
    Falls back to curated realistic headlines when key is absent.
    """
    if not api_key:
        avg   = sum(h["score"] for h in _FALLBACK_HEADLINES) / len(_FALLBACK_HEADLINES)
        score = round((1.0 - avg) / 2.0, 3)
        return NewsSignal(risk_score=score, headlines=_FALLBACK_HEADLINES, live=False)

    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}&language=en&pageSize=10&sortBy=publishedAt&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            sia = SentimentIntensityAnalyzer()
            headlines = [
                {"title": a["title"], "score": round(sia.polarity_scores(a["title"])["compound"], 3)}
                for a in articles
                if a.get("title") and a["title"] != "[Removed]"
            ]
        except ImportError:
            headlines = [{"title": a["title"], "score": 0.0} for a in articles if a.get("title")]

        if not headlines:
            raise ValueError("No usable headlines returned")

        avg   = sum(h["score"] for h in headlines) / len(headlines)
        score = round((1.0 - avg) / 2.0, 3)
        return NewsSignal(risk_score=score, headlines=headlines[:6], live=True)

    except Exception:
        avg   = sum(h["score"] for h in _FALLBACK_HEADLINES) / len(_FALLBACK_HEADLINES)
        score = round((1.0 - avg) / 2.0, 3)
        return NewsSignal(risk_score=score, headlines=_FALLBACK_HEADLINES, live=False)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER  — role-aware input construction
# ─────────────────────────────────────────────────────────────────────────────

# These are the 12 features the model was trained on (order matters for SHAP)
MODEL_FEATURE_COLUMNS = [
    "avg_delay_days", "disruptions_12m", "on_time_rate",
    "inventory_days_remaining", "chaos_score", "demand_surge_score",
    "supplier_fragility", "demand_supply_gap",
    "delivery_month", "delivery_weekday", "is_weekend", "month_end",
]


def build_feature_vector(
    role: str,
    *,
    # Manager / shared
    avg_delay_days: float = 2.0,
    on_time_rate: float = 0.85,
    supplier_fragility: float = 0.05,
    disruptions_12m: float = 5.0,
    chaos_score: float = 0.01,
    # Supplier
    inventory_days_remaining: float = 1000.0,
    demand_surge_score: float = 0.05,
    demand_supply_gap: float = 0.0,
    # Driver (real-time signals)
    weather_severity: float = 0.0,
    congestion_index: float = 0.3,
) -> dict[str, float]:
    """
    Constructs the model's 12-feature input vector with role-specific
    signal weighting. The same ensemble model is used for all roles;
    we modify INPUTS not the model.

    Role weighting logic:
      - Driver:   chaos_score dominated by weather + congestion
      - Manager:  chaos_score dominated by delay + supplier signals
      - Supplier: chaos_score dominated by demand + inventory signals
    """
    now = datetime.now()

    if role == "Driver":
        # Weather and congestion are first-class signals
        adjusted_chaos = round(
            0.50 * weather_severity +
            0.30 * congestion_index +
            0.20 * min(1.0, avg_delay_days / 10.0),
            4,
        )
        # Driver doesn't manage inventory — use population mean
        inv_days = 2658.0
        d_surge  = 0.052
        d_gap    = 0.0
        # Congestion inflates effective delay
        effective_delay = round(avg_delay_days + congestion_index * 3.0, 2)

    elif role == "Supplier":
        # Demand and inventory dominate
        adjusted_chaos = round(
            0.40 * demand_surge_score +
            0.30 * supplier_fragility +
            0.30 * max(0.0, 1.0 - inventory_days_remaining / 30.0),
            4,
        )
        inv_days       = inventory_days_remaining
        d_surge        = demand_surge_score
        d_gap          = demand_supply_gap
        effective_delay = avg_delay_days

    else:  # Manager
        adjusted_chaos = round(
            0.40 * min(1.0, avg_delay_days / 10.0) +
            0.35 * supplier_fragility +
            0.25 * chaos_score,
            4,
        )
        inv_days       = 2658.0
        d_surge        = 0.052
        d_gap          = 0.0
        effective_delay = avg_delay_days

    return {
        "avg_delay_days":          round(effective_delay, 2),
        "disruptions_12m":         round(disruptions_12m, 2),
        "on_time_rate":            round(on_time_rate, 4),
        "inventory_days_remaining": round(inv_days, 2),
        "chaos_score":             adjusted_chaos,
        "demand_surge_score":      round(d_surge, 4),
        "supplier_fragility":      round(supplier_fragility, 4),
        "demand_supply_gap":       round(d_gap, 2),
        "delivery_month":          float(now.month),
        "delivery_weekday":        float(now.weekday()),
        "is_weekend":              float(now.weekday() >= 5),
        "month_end":               float(now.day > 25),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-READABLE EXPLANATION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(
    role: str,
    prob: float,
    feature_vector: dict,
    shap_top: list[tuple[str, float]],
) -> str:
    """
    Generates a plain-English explanation of the prediction.
    shap_top: [(feature_name, shap_value), ...] sorted by abs importance desc.
    """
    level = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.5 else "LOW"
    pct   = round(prob * 100, 1)

    # Top 2 drivers
    drivers = [f for f, v in shap_top[:3] if v > 0]

    _labels = {
        "avg_delay_days":           "high average delivery delay",
        "chaos_score":              "elevated chaos score (weather + congestion)",
        "supplier_fragility":       "fragile supplier profile",
        "on_time_rate":             "poor on-time delivery rate",
        "disruptions_12m":          "frequent past disruptions",
        "inventory_days_remaining": "critically low inventory",
        "demand_surge_score":       "demand surge pressure",
        "demand_supply_gap":        "demand-supply mismatch",
        "weather_severity":         "severe weather conditions",
        "congestion_index":         "heavy traffic congestion",
    }

    cause_text = " and ".join(_labels.get(d, d) for d in drivers[:2])

    if role == "Driver":
        if prob > 0.7:
            return (
                f"⚠️ Route risk is {level} ({pct}%). "
                f"Primary drivers: {cause_text}. "
                "Recommend switching to alternate route immediately and reducing speed in affected zones."
            )
        elif prob > 0.5:
            return (
                f"🟡 Route risk is {level} ({pct}%). "
                f"{cause_text.capitalize()} detected. "
                "Proceed with caution — monitor conditions every 30 minutes."
            )
        else:
            return f"🟢 Route conditions are {level} risk ({pct}%). Clear to proceed on planned route."

    elif role == "Supplier":
        if prob > 0.7:
            return (
                f"🔴 Disruption risk is {level} ({pct}%). "
                f"Caused by {cause_text}. "
                "Immediate action required: notify buyers, trigger safety stock protocol."
            )
        elif prob > 0.5:
            return (
                f"🟡 Disruption risk is {level} ({pct}%). "
                f"{cause_text.capitalize()} warrants close monitoring. "
                "Prepare contingency supply plan for next 7 days."
            )
        else:
            return f"🟢 Operations appear stable ({pct}% risk). Continue current production schedule."

    else:  # Manager
        if prob > 0.7:
            return (
                f"🔴 Supply chain disruption risk is {level} ({pct}%). "
                f"Key risk factors: {cause_text}. "
                "Escalate to operations team and activate supplier contingency protocol."
            )
        elif prob > 0.5:
            return (
                f"🟡 Moderate disruption risk detected ({pct}%). "
                f"{cause_text.capitalize()} are the primary concerns. "
                "Review supplier SLAs and increase inventory buffer."
            )
        else:
            return f"🟢 Supply chain health is stable ({pct}% disruption risk). No immediate action required."


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def compute_financial_impact(
    prob: float,
    avg_delay_days: float,
    inventory_days_remaining: float,
    avg_order_value_inr: float = 500_000,
) -> dict[str, int]:
    """
    Estimates 30-day financial exposure in INR.
    Methodology: delay cost + stockout cost + probabilistic order loss.
    """
    delay_cost    = int(avg_delay_days * 18_000)                           # ₹18k / delay day
    stockout_days = max(0.0, 7.0 - inventory_days_remaining)
    stockout_cost = int(stockout_days * 30_000)                            # ₹30k / stockout day
    order_risk    = int(prob * avg_order_value_inr)
    total         = delay_cost + stockout_cost + order_risk
    return {
        "delay_cost_inr":    delay_cost,
        "stockout_cost_inr": stockout_cost,
        "order_risk_inr":    order_risk,
        "total_inr":         total,
    }
