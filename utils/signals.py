"""
utils/signals.py  (enhanced)
-----------------------------
Real-world signal fetchers for the Supply Chain AI platform.
New in this version:
  - fetch_ors_route()  — OpenRouteService directions + distance/duration
  - build_feature_vector() — news_risk maps to supplier_fragility, ORS overrides delay
  - generate_narrative()   — fully SHAP-driven, no hard-coded feature assumptions
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import TypedDict
import requests


# ── Type contracts ─────────────────────────────────────────────────────────

class WeatherSignal(TypedDict):
    condition: str
    severity: float
    temp_c: float
    humidity: int
    wind_kmh: float
    city: str
    live: bool

class CongestionSignal(TypedDict):
    index: float
    level: str
    peak_hours: bool
    description: str

class NewsSignal(TypedDict):
    risk_score: float
    headlines: list[dict]
    live: bool

class RouteSignal(TypedDict):
    distance_km: float
    duration_min: float
    geometry: list[list[float]]   # [[lon, lat], ...]
    waypoints: list[dict]
    congestion_est: float
    delay_days_est: float
    live: bool
    error: str


# ── Weather ────────────────────────────────────────────────────────────────

_SEVERITY_MAP: dict[str, float] = {
    "Clear":0.0,"Clouds":0.1,"Mist":0.2,"Haze":0.2,"Smoke":0.3,
    "Drizzle":0.35,"Fog":0.4,"Dust":0.45,"Rain":0.55,"Snow":0.65,
    "Squall":0.70,"Thunderstorm":0.85,"Tornado":1.0,
}

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Chennai":(13.0827,80.2707),"Mumbai":(19.0760,72.8777),
    "Delhi":(28.6139,77.2090),"Bangalore":(12.9716,77.5946),
    "Kolkata":(22.5726,88.3639),"Hyderabad":(17.3850,78.4867),
    "Pune":(18.5204,73.8567),"Surat":(21.1702,72.8311),
}

_CITY_FALLBACKS: dict[str, WeatherSignal] = {
    "Chennai":   {"condition":"Rain",        "severity":0.55,"temp_c":32.0,"humidity":85,"wind_kmh":18.0,"city":"Chennai",   "live":False},
    "Mumbai":    {"condition":"Clouds",       "severity":0.10,"temp_c":30.0,"humidity":78,"wind_kmh":14.0,"city":"Mumbai",    "live":False},
    "Delhi":     {"condition":"Haze",         "severity":0.20,"temp_c":28.0,"humidity":60,"wind_kmh":10.0,"city":"Delhi",     "live":False},
    "Bangalore": {"condition":"Clear",        "severity":0.00,"temp_c":25.0,"humidity":55,"wind_kmh": 8.0,"city":"Bangalore", "live":False},
    "Kolkata":   {"condition":"Thunderstorm", "severity":0.85,"temp_c":31.0,"humidity":90,"wind_kmh":25.0,"city":"Kolkata",   "live":False},
    "Hyderabad": {"condition":"Clouds",       "severity":0.10,"temp_c":29.0,"humidity":65,"wind_kmh":12.0,"city":"Hyderabad", "live":False},
    "Pune":      {"condition":"Clear",        "severity":0.00,"temp_c":26.0,"humidity":50,"wind_kmh": 9.0,"city":"Pune",      "live":False},
    "Surat":     {"condition":"Clouds",       "severity":0.10,"temp_c":31.0,"humidity":72,"wind_kmh":15.0,"city":"Surat",     "live":False},
}

def fetch_weather(city: str, api_key: str = "") -> WeatherSignal:
    fallback = _CITY_FALLBACKS.get(city, _CITY_FALLBACKS["Chennai"])
    if not api_key:
        return fallback
    try:
        url  = (f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={city},IN&appid={api_key}&units=metric")
        data = requests.get(url, timeout=6).json()
        cond = data["weather"][0]["main"]
        sev  = _SEVERITY_MAP.get(cond, 0.25)
        if data.get("rain", {}).get("1h", 0) > 20: sev = min(1.0, sev + 0.15)
        wind = round(data["wind"]["speed"] * 3.6, 1)
        if wind > 60: sev = min(1.0, sev + 0.10)
        return WeatherSignal(condition=cond, severity=round(sev, 3),
                             temp_c=round(data["main"]["temp"], 1),
                             humidity=data["main"]["humidity"],
                             wind_kmh=wind, city=city, live=True)
    except Exception:
        return fallback


# ── Congestion ─────────────────────────────────────────────────────────────

def compute_congestion(city: str, hour: int | None = None) -> CongestionSignal:
    if hour is None: hour = datetime.now().hour
    pat = {0:0.05,1:0.04,2:0.04,3:0.04,4:0.05,5:0.08,6:0.25,7:0.55,8:0.85,
           9:0.90,10:0.70,11:0.55,12:0.50,13:0.48,14:0.45,15:0.52,16:0.65,
           17:0.88,18:0.92,19:0.85,20:0.65,21:0.40,22:0.20,23:0.10}
    mult = {"Mumbai":1.20,"Delhi":1.15,"Chennai":1.05,"Bangalore":1.18,
            "Kolkata":1.10,"Hyderabad":1.08,"Pune":1.00,"Surat":0.92}
    idx  = round(min(1.0, pat.get(hour, 0.5) * mult.get(city, 1.0)), 3)
    if datetime.now().weekday() >= 5: idx = round(idx * 0.55, 3)
    if   idx < 0.25: lvl, desc = "Low",      "Roads clear — minimal delay expected"
    elif idx < 0.50: lvl, desc = "Moderate", "Normal traffic — minor slowdowns possible"
    elif idx < 0.75: lvl, desc = "High",     "Heavy congestion — expect significant delays"
    else:            lvl, desc = "Severe",   "Gridlock conditions — reroute immediately"
    return CongestionSignal(index=idx, level=lvl, peak_hours=idx>0.65, description=desc)


# ── News ───────────────────────────────────────────────────────────────────

_FALLBACK_HEADLINES: list[dict] = [
    {"title":"Cyclone warning issued for Bay of Bengal coastline",         "score":-0.82},
    {"title":"Chennai port faces congestion amid surge in imports",        "score":-0.55},
    {"title":"Fuel prices hiked by ₹2/litre — logistics cost impact",     "score":-0.48},
    {"title":"New trade corridor opens between India and UAE",             "score": 0.62},
    {"title":"Freight demand up 12% — capacity crunch likely next quarter","score":-0.35},
]

def fetch_news_risk(api_key: str = "", query: str = "supply chain India disruption") -> NewsSignal:
    def _score(headlines): return round((1.0 - sum(h["score"] for h in headlines)/len(headlines)) / 2.0, 3)
    if not api_key:
        return NewsSignal(risk_score=_score(_FALLBACK_HEADLINES), headlines=_FALLBACK_HEADLINES, live=False)
    try:
        url  = (f"https://newsapi.org/v2/everything?q={query}&language=en"
                f"&pageSize=10&sortBy=publishedAt&apiKey={api_key}")
        arts = requests.get(url, timeout=8).json().get("articles", [])
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            hl  = [{"title":a["title"],"score":round(sia.polarity_scores(a["title"])["compound"],3)}
                   for a in arts if a.get("title") and a["title"] != "[Removed]"]
        except ImportError:
            hl = [{"title":a["title"],"score":0.0} for a in arts if a.get("title")]
        if not hl: raise ValueError("empty")
        return NewsSignal(risk_score=_score(hl), headlines=hl[:6], live=True)
    except Exception:
        return NewsSignal(risk_score=_score(_FALLBACK_HEADLINES), headlines=_FALLBACK_HEADLINES, live=False)


# ── OpenRouteService ───────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1,p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

def _geocode_ors(place: str, api_key: str) -> tuple[float,float] | None:
    for name,(lat,lon) in CITY_COORDS.items():
        if name.lower() in place.lower(): return (lon, lat)
    try:
        r = requests.get("https://api.openrouteservice.org/geocode/search",
                         params={"api_key":api_key,"text":place,
                                 "boundary.country":"IN","size":1}, timeout=6)
        r.raise_for_status()
        feats = r.json().get("features",[])
        if feats: return tuple(feats[0]["geometry"]["coordinates"])
    except Exception: pass
    return None

def fetch_ors_route(origin: str, destination: str,
                    api_key: str = "", congestion_index: float = 0.5) -> RouteSignal:
    def _fallback(reason: str) -> RouteSignal:
        oc = CITY_COORDS.get(origin,      (13.0827, 80.2707))
        dc = CITY_COORDS.get(destination, (19.0760, 72.8777))
        dist  = round(_haversine_km(oc[0],oc[1],dc[0],dc[1]) * 1.35, 1)
        spd   = max(15.0, 60.0 * (1.0 - congestion_index * 0.6))
        dur   = round((dist/spd)*60, 1)
        cong  = congestion_index
        delay = round(cong * 2.5 + congestion_index * 0.5, 2)
        geom  = [[oc[1],oc[0]], [dc[1],dc[0]]]
        return RouteSignal(distance_km=dist, duration_min=dur, geometry=geom,
                           waypoints=[{"name":f"{origin} (origin)","lat":oc[0],"lon":oc[1]},
                                      {"name":f"{destination} (dest.)","lat":dc[0],"lon":dc[1]}],
                           congestion_est=round(cong,3), delay_days_est=round(delay,3),
                           live=False, error=reason)

    if not api_key:
        return _fallback("No ORS API key — using straight-line estimate")

    orig = _geocode_ors(origin, api_key)
    dest = _geocode_ors(destination, api_key)
    if not orig: return _fallback(f"Could not geocode: {origin}")
    if not dest: return _fallback(f"Could not geocode: {destination}")

    try:
        headers = {"Authorization": api_key, "Content-Type": "application/json"}
        body    = {"coordinates":[list(orig),list(dest)], "geometry_format":"geojson"}
        r       = requests.post("https://api.openrouteservice.org/v2/directions/driving-car",
                                json=body, headers=headers, timeout=12)
        r.raise_for_status()
        data    = r.json()
        route   = data["routes"][0]
        dist    = round(route["summary"]["distance"] / 1000, 2)
        dur     = round(route["summary"]["duration"] / 60, 1)
        geom    = route["geometry"]["coordinates"]
        # congestion: ratio of actual vs free-flow (80 km/h highway baseline)
        ff      = (dist / 80.0) * 60.0
        ratio   = max(0.0, min(1.0, (dur - ff) / max(ff, 1.0)))
        cong    = round(0.6*ratio + 0.4*congestion_index, 3)
        delay   = round(cong * 2.5, 3)
        wpts    = [{"name":f"{origin} (origin)","lat":orig[1],"lon":orig[0]},
                   {"name":f"{destination} (dest.)","lat":dest[1],"lon":dest[0]}]
        return RouteSignal(distance_km=dist, duration_min=dur, geometry=geom,
                           waypoints=wpts, congestion_est=cong, delay_days_est=delay,
                           live=True, error="")
    except Exception as e:
        return _fallback(f"ORS error: {str(e)[:80]}")


# ── Feature vector ─────────────────────────────────────────────────────────

MODEL_FEATURE_COLUMNS = [
    "avg_delay_days","disruptions_12m","on_time_rate",
    "inventory_days_remaining","chaos_score","demand_surge_score",
    "supplier_fragility","demand_supply_gap",
    "delivery_month","delivery_weekday","is_weekend","month_end",
]

def build_feature_vector(
    role: str, *,
    avg_delay_days: float       = 2.0,
    on_time_rate: float         = 0.85,
    supplier_fragility: float   = 0.05,
    disruptions_12m: float      = 5.0,
    chaos_score: float          = 0.01,
    inventory_days_remaining: float = 1000.0,
    demand_surge_score: float   = 0.05,
    demand_supply_gap: float    = 0.0,
    weather_severity: float     = 0.0,
    congestion_index: float     = 0.3,
    news_risk_score: float      = 0.5,
    route_congestion: float     = -1.0,
    route_delay_days: float     = -1.0,
) -> dict[str, float]:
    now         = datetime.now()
    eff_cong    = route_congestion if route_congestion >= 0 else congestion_index
    eff_delay   = route_delay_days  if route_delay_days  >= 0 else avg_delay_days

    if role == "Driver":
        chaos  = round(0.50*weather_severity + 0.30*eff_cong + 0.20*min(1.0,eff_delay/10.0), 4)
        inv    = 2658.0; surge = 0.052; gap = 0.0; frag = supplier_fragility
        delay  = round(eff_delay + eff_cong * 3.0, 2)
    elif role == "Supplier":
        chaos  = round(0.40*demand_surge_score + 0.30*supplier_fragility
                       + 0.30*max(0.0,1.0-inventory_days_remaining/30.0), 4)
        inv    = inventory_days_remaining; surge = demand_surge_score
        gap    = demand_supply_gap;        frag  = supplier_fragility; delay = avg_delay_days
    else:  # Manager — news risk adjusts fragility
        frag   = round(min(0.142, supplier_fragility + news_risk_score * 0.02), 4)
        chaos  = round(0.40*min(1.0,avg_delay_days/10.0) + 0.35*frag + 0.25*chaos_score, 4)
        inv    = 2658.0; surge = 0.052; gap = 0.0; delay = avg_delay_days

    return {
        "avg_delay_days":           round(delay, 2),
        "disruptions_12m":          round(disruptions_12m, 2),
        "on_time_rate":             round(on_time_rate, 4),
        "inventory_days_remaining": round(inv, 2),
        "chaos_score":              chaos,
        "demand_surge_score":       round(surge, 4),
        "supplier_fragility":       round(frag, 4),
        "demand_supply_gap":        round(gap, 2),
        "delivery_month":           float(now.month),
        "delivery_weekday":         float(now.weekday()),
        "is_weekend":               float(now.weekday() >= 5),
        "month_end":                float(now.day > 25),
    }


# ── Narrative (SHAP-driven) ────────────────────────────────────────────────

_FEAT_LABELS: dict[str, str] = {
    "avg_delay_days":           "high delivery delay",
    "chaos_score":              "elevated chaos score",
    "supplier_fragility":       "fragile supplier profile",
    "on_time_rate":             "poor on-time rate",
    "disruptions_12m":          "frequent past disruptions",
    "inventory_days_remaining": "critically low inventory",
    "demand_surge_score":       "demand surge pressure",
    "demand_supply_gap":        "demand-supply mismatch",
    "delivery_weekday":         "high-risk delivery day",
    "is_weekend":               "weekend delivery risk",
    "month_end":                "month-end pressure",
    "delivery_month":           "seasonal risk",
}

_ACTIONS: dict[str, dict[str, str]] = {
    "Manager": {
        "high":   "Escalate to operations team and activate supplier contingency protocol immediately.",
        "medium": "Review supplier SLAs, increase safety stock, and brief logistics coordinators.",
        "low":    "Continue routine monitoring — no immediate action required.",
    },
    "Driver": {
        "high":   "Switch to alternate route immediately. Notify dispatch and reduce speed in risk zones.",
        "medium": "Proceed with caution — check in with dispatch every 30 minutes.",
        "low":    "Clear to proceed on planned route. Conditions are favourable.",
    },
    "Supplier": {
        "high":   "Notify buyers immediately. Trigger safety-stock protocol and activate sub-suppliers.",
        "medium": "Prepare a 7-day contingency supply plan and monitor demand signals closely.",
        "low":    "Continue current production schedule. No disruption signals detected.",
    },
}

def generate_narrative(role: str, prob: float, feature_vector: dict,
                        shap_top: list[tuple[str, float]]) -> str:
    level  = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.5 else "LOW"
    tier   = "high" if prob > 0.7 else "medium"   if prob > 0.5 else "low"
    pct    = round(prob * 100, 1)
    icon   = "🔴" if prob > 0.7 else "🟡" if prob > 0.5 else "🟢"

    drivers    = [(f,v) for f,v in shap_top if v > 0][:3]
    mitigants  = [(f,v) for f,v in shap_top if v < 0][:1]

    dlabels = [_FEAT_LABELS.get(f, f.replace("_"," ")) for f,_ in drivers]
    if   len(dlabels) == 0: driver_txt = "no dominant risk factor identified"
    elif len(dlabels) == 1: driver_txt = dlabels[0]
    elif len(dlabels) == 2: driver_txt = f"{dlabels[0]} and {dlabels[1]}"
    else:                   driver_txt = f"{dlabels[0]}, {dlabels[1]}, and {dlabels[2]}"

    mit_txt = ""
    if mitigants:
        f, _ = mitigants[0]
        mit_txt = f"  Partially offset by {_FEAT_LABELS.get(f, f.replace('_',' '))}."

    action = _ACTIONS.get(role, _ACTIONS["Manager"])[tier]
    return (f"{icon} **{level} RISK** — {pct}% disruption probability.  \n"
            f"Primary drivers: **{driver_txt}**.{mit_txt}  \n"
            f"{action}")


# ── Financial impact (unchanged) ──────────────────────────────────────────

def compute_financial_impact(prob, avg_delay_days, inventory_days_remaining,
                              avg_order_value_inr=500_000) -> dict[str,int]:
    delay     = int(avg_delay_days * 18_000)
    stockout  = int(max(0.0, 7.0 - inventory_days_remaining) * 30_000)
    order_r   = int(prob * avg_order_value_inr)
    return {"delay_cost_inr":delay,"stockout_cost_inr":stockout,
            "order_risk_inr":order_r,"total_inr":delay+stockout+order_r}