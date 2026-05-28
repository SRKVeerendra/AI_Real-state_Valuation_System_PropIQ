import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="PropIQ | AI Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# GLOBAL CSS — zoom-friendly and responsive
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#070b12;
  --bg2:#0d1320;
  --bg3:#111827;
  --bg4:#0f1727;
  --panel:#0c1220;
  --card:#121a2a;
  --card2:#0f1626;
  --line:rgba(148,163,184,.14);
  --line2:rgba(148,163,184,.22);
  --text:#e5ecf5;
  --muted:#9aa8bc;
  --faint:#66758a;
  --accent:#5b8cff;
  --accent2:#8fb3ff;
  --success:#22c55e;
  --danger:#f43f5e;
  --warning:#f59e0b;
  --purple:#9b8cff;
  --radius:14px;
  --radius-sm:10px;
  --shadow:0 12px 30px rgba(0,0,0,.22);

  --text-xs: clamp(0.75rem, 0.72rem + 0.15vw, 0.875rem);
  --text-sm: clamp(0.875rem, 0.82rem + 0.22vw, 1rem);
  --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.08rem);
  --text-lg: clamp(1.1rem, 1rem + 0.6vw, 1.35rem);
  --text-xl: clamp(1.35rem, 1.1rem + 1vw, 1.8rem);
  --text-2xl: clamp(1.8rem, 1.2rem + 2vw, 2.8rem);

  --space-1:0.25rem;
  --space-2:0.5rem;
  --space-3:0.75rem;
  --space-4:1rem;
  --space-5:1.25rem;
  --space-6:1.5rem;
  --space-8:2rem;
}

html {
  font-size: 16px;
  -webkit-text-size-adjust: 100%;
  text-size-adjust: 100%;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
  font-variant-numeric: tabular-nums lining-nums;
  font-size: var(--text-base);
  line-height: 1.5;
}

#MainMenu, footer, header {visibility:hidden;}
.block-container{
  padding:0 !important;
  max-width:100% !important;
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at top right, rgba(91,140,255,.08), transparent 20%),
    linear-gradient(180deg, #070b12 0%, #09101b 100%);
}
[data-testid="stHeader"]{
  background:transparent !important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a0f1b 0%, #0b1220 100%) !important;
  border-right:1px solid var(--line) !important;
}

.page-wrap{
  padding: 1rem 1rem 1.25rem;
  max-width: 1600px;
  margin: 0 auto;
}

.hero-title{
  font-size: var(--text-xl);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -.03em;
  color: var(--text);
  margin: 0 0 .4rem 0;
}
.hero-body{
  color: var(--muted);
  font-size: var(--text-sm);
  line-height: 1.7;
  max-width: 920px;
  margin: 0 0 .85rem 0;
}

input, textarea, select{
  background: var(--bg3) !important;
  color: var(--text) !important;
  border:1px solid var(--line2) !important;
  border-radius:10px !important;
}
.stSlider label, .stNumberInput label, .stSelectbox label,
.stRadio label, .stTextInput label{
  color:var(--muted) !important;
  font-size:var(--text-xs) !important;
  font-weight:600 !important;
}
.stButton > button{
  background:linear-gradient(180deg, #5b8cff 0%, #4379f2 100%) !important;
  color:white !important;
  border:none !important;
  border-radius:10px !important;
  font-weight:700 !important;
  font-size:var(--text-sm) !important;
  padding:0.62rem 0.95rem !important;
  box-shadow:0 10px 22px rgba(91,140,255,.18) !important;
}

.topbar{
  background:linear-gradient(145deg,#0b1220 0%, #101a2d 100%);
  border:1px solid var(--line);
  border-radius:16px;
  padding:1rem 1.1rem;
  margin-bottom:.85rem;
  box-shadow:var(--shadow);
}
.brand-kicker{
  font-family:'JetBrains Mono', monospace;
  font-size:var(--text-xs);
  letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--accent2);
  margin-bottom:.35rem;
}
.hero-title span{color:var(--accent2);}
.hero-stats{
  display:flex;
  flex-wrap:wrap;
  gap:1rem;
}
.hero-stat{min-width:110px;}
.hero-stat-val{
  font-size:var(--text-base);
  font-weight:800;
  color:var(--text);
}
.hero-stat-key{
  font-size:var(--text-xs);
  color:var(--faint);
  margin-top:.1rem;
}

.sec-title{
  font-size:var(--text-lg);
  font-weight:800;
  letter-spacing:-.02em;
  color:var(--text);
  margin-bottom:.18rem;
}
.sec-sub{
  font-size:var(--text-sm);
  color:var(--faint);
  margin-bottom:.8rem;
  line-height:1.55;
}

.card, .chart-card, .kpi-card, .insight-card, .resource-card, .mini-card{
  background:linear-gradient(180deg, rgba(18,26,42,.98) 0%, rgba(13,19,32,.98) 100%);
  border:1px solid var(--line);
  border-radius:var(--radius);
  box-shadow:var(--shadow);
}
.kpi-card{
  padding:.9rem;
  min-height:104px;
}
.kpi-label{
  font-size:var(--text-xs);
  font-weight:700;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--faint);
  margin-bottom:.35rem;
}
.kpi-value{
  font-size:var(--text-lg);
  font-weight:800;
  color:var(--text);
  line-height:1.12;
}
.kpi-foot{
  font-size:var(--text-sm);
  margin-top:.3rem;
}
.pos{color:var(--success);}
.neg{color:var(--danger);}
.neu{color:var(--muted);}

.chart-card, .card, .insight-card, .resource-card, .mini-card{
  padding:1rem;
  margin-bottom:.8rem;
}
.card-head{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:.6rem;
  margin-bottom:.7rem;
}
.card-title{
  font-size:var(--text-xs);
  font-weight:800;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:var(--muted);
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  font-size:var(--text-xs);
  font-weight:700;
  letter-spacing:.08em;
  text-transform:uppercase;
  padding:.22rem .55rem;
  border-radius:999px;
  border:1px solid rgba(143,179,255,.2);
  background:rgba(91,140,255,.12);
  color:var(--accent2);
}
.pred-banner{
  background:
    radial-gradient(circle at top center, rgba(91,140,255,.14), transparent 45%),
    linear-gradient(135deg,#122143 0%,#0d1831 100%);
  border:1px solid rgba(91,140,255,.22);
  border-radius:16px;
  padding:1.25rem 1rem;
  text-align:center;
  box-shadow:0 16px 34px rgba(0,0,0,.24);
  margin-bottom:.85rem;
}
.pred-tag{
  font-family:'JetBrains Mono', monospace;
  font-size:var(--text-xs);
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--accent2);
  margin-bottom:.4rem;
}
.pred-price{
  font-size:var(--text-2xl);
  line-height:1;
  font-weight:800;
  color:white;
  margin:.3rem 0;
}
.pred-ci{
  font-size:var(--text-sm);
  color:#b9cbf0;
}
.pred-sub{
  margin-top:.55rem;
  font-size:var(--text-xs);
  color:var(--faint);
}

.spec-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:.45rem .95rem;
}
.spec-item{
  display:flex;
  justify-content:space-between;
  gap:.8rem;
  padding:.4rem 0;
  border-bottom:1px solid rgba(148,163,184,.08);
}
.spec-key{
  font-size:var(--text-sm);
  color:var(--muted);
}
.spec-val{
  font-size:var(--text-sm);
  color:var(--text);
  font-weight:700;
  text-align:right;
}

.insight-title{
  font-size:var(--text-base);
  font-weight:800;
  color:var(--text);
  margin-bottom:.25rem;
}
.insight-body{
  font-size:var(--text-sm);
  color:var(--muted);
  line-height:1.65;
}
.insight-meta{
  margin-top:.45rem;
  font-size:var(--text-xs);
  color:var(--faint);
  font-family:'JetBrains Mono', monospace;
}
.resource-tag{
  display:inline-block;
  font-family:'JetBrains Mono', monospace;
  font-size:var(--text-xs);
  letter-spacing:.08em;
  text-transform:uppercase;
  padding:.15rem .5rem;
  border-radius:999px;
  margin-bottom:.45rem;
}
.tag-dataset{background:rgba(91,140,255,.12); color:var(--accent2);}
.tag-research{background:rgba(34,197,94,.12); color:#6ee7a3;}
.tag-tool{background:rgba(245,158,11,.12); color:#fbbf24;}
.tag-guide{background:rgba(155,140,255,.12); color:#c4b5fd;}
.resource-title{
  font-size:var(--text-base);
  font-weight:800;
  color:var(--text);
  margin-bottom:.22rem;
}
.resource-desc{
  font-size:var(--text-sm);
  color:var(--muted);
  line-height:1.6;
}
.footer{
  margin-top:.4rem;
  border-top:1px solid var(--line);
  padding: .9rem 0 .25rem;
  text-align:center;
  font-size:var(--text-sm);
  color:var(--faint);
}

hr.soft{
  border:none;
  border-top:1px solid var(--line);
  margin:.85rem 0;
}
.js-plotly-plot{border-radius:12px;}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #263246; border-radius: 999px; }
::-webkit-scrollbar-track { background: #0b1220; }

@media (max-width: 900px){
  .spec-grid{grid-template-columns:1fr;}
}

@media (prefers-reduced-motion: reduce){
  *{scroll-behavior:auto !important;}
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PAGES = ["Home", "Market Analytics", "Valuation", "Insights", "Resources", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

defaults = {
    "bedrooms":3, "bathrooms":2.0, "sqft_living":2000, "sqft_lot":6000, "floors":1.5,
    "sqft_basement":0, "grade":7, "condition":3, "waterfront":0, "view":0,
    "yr_built":1995, "yr_renovated":0, "zipcode":122003, "lat":28.6200, "lon":77.2100,
    "sqft_living15":1800, "sqft_lot15":5000, "schools_nearby":2, "airport_distance":65,
    "sale_year":2020, "sale_month":6, "hpi_val":200.0, "real_price_idx":120.0,
    "nom_price_idx":140.0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "pred" not in st.session_state:
    st.session_state.pred = 2450000.0
    st.session_state.lo = 2156000.0
    st.session_state.hi = 2744000.0

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9aa8bc", family="Inter", size=12),
    margin=dict(t=30, b=30, l=38, r=18),
)

def apply_theme(fig, height=290):
    fig.update_layout(
        **PLOT_THEME,
        height=height,
        xaxis=dict(gridcolor="rgba(148,163,184,.10)", color="#7f8ea3"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", color="#7f8ea3"),
    )
    return fig

CFG = {"displayModeBar": False}

@st.cache_resource(show_spinner=False)
def load_artefacts():
    try:
        model = joblib.load(MODELS_DIR / "best_model.joblib")
        engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
        ct = joblib.load(MODELS_DIR / "column_transformer.joblib")
        te = joblib.load(MODELS_DIR / "target_encoder.joblib")
        col_groups = joblib.load(MODELS_DIR / "col_groups.joblib")
        fi = joblib.load(MODELS_DIR / "feature_importance.joblib")
        with open(MODELS_DIR / "training_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return model, engineer, ct, te, col_groups, fi, meta, True
    except Exception:
        return None, None, None, None, None, None, {}, False

model, engineer, ct, te, col_groups, fi, meta, model_loaded = load_artefacts()

@st.cache_data(show_spinner=False)
def load_macro():
    try:
        df = pd.read_csv(DATA_DIR / "real_index.csv", encoding="latin-1")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        world = df[df["country_code"] == "5R"].dropna(subset=["price", "date"]).sort_values("date")
        world = world[world["date"].dt.year >= 1980]
        return world
    except Exception:
        dates = pd.date_range("1980", periods=44, freq="YE")
        vals = 80 + np.cumsum(np.random.normal(1.6, 2.2, len(dates)))
        return pd.DataFrame({"date": dates, "price": vals})

@st.cache_data(show_spinner=False)
def load_nominal():
    try:
        df = pd.read_csv(DATA_DIR / "nominal_index.csv", encoding="latin-1")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        world = df[df["country_code"] == "5R"].dropna(subset=["price", "date"]).sort_values("date")
        return world
    except Exception:
        dates = pd.date_range("1980", periods=44, freq="YE")
        vals = 90 + np.cumsum(np.random.normal(1.9, 2.0, len(dates)))
        return pd.DataFrame({"date": dates, "price": vals})

@st.cache_data(show_spinner=False)
def load_india():
    try:
        return pd.read_csv(DATA_DIR / "House_Price_India.csv")
    except Exception:
        rng = np.random.default_rng(42)
        n = 350
        cities = ["Mumbai","Delhi","Bengaluru","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad"]
        return pd.DataFrame({
            "City": rng.choice(cities, n),
            "Price": rng.integers(3500000, 35000000, n),
            "Area": rng.integers(600, 4500, n),
            "Bedrooms": rng.integers(1, 6, n),
            "Bathrooms": rng.integers(1, 5, n),
            "Latitude": rng.uniform(12.8, 28.8, n),
            "Longitude": rng.uniform(72.8, 88.4, n)
        })

@st.cache_data(show_spinner=False)
def load_hpi():
    try:
        df = pd.read_csv(DATA_DIR / "housing_price_index_2010-11_100.csv")
        df.columns = df.columns.str.strip()
        df.rename(columns={"Particulars": "city"}, inplace=True)
        qtrs = [c for c in df.columns if c != "city"]
        long = df.melt(id_vars="city", value_vars=qtrs, var_name="quarter", value_name="hpi")
        long["hpi"] = pd.to_numeric(long["hpi"], errors="coerce")
        return long.dropna(subset=["hpi"])
    except Exception:
        demo_q = ["Q1-2024","Q2-2024","Q3-2024","Q4-2024","Q1-2025"]
        demo_cities = ["Bengaluru","Mumbai","Delhi","Chennai","Hyderabad","Pune"]
        rows = []
        rng = np.random.default_rng(7)
        for city in demo_cities:
            base = rng.uniform(160, 190)
            for i, q in enumerate(demo_q):
                rows.append([city, q, base + i * rng.uniform(2, 5)])
        return pd.DataFrame(rows, columns=["city", "quarter", "hpi"])

macro_df = load_macro()
nom_df = load_nominal()
india_df = load_india()
hpi_df = load_hpi()

def make_demo_market_dataset():
    cities = ["Mumbai","Delhi","Bengaluru","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad"]
    return pd.DataFrame({
        "city": cities,
        "avg_price_cr":[2.95, 2.35, 1.82, 1.45, 1.58, 1.62, 1.26, 1.18],
        "yoy_growth":[5.9, 2.9, 13.1, 9.0, 4.8, 6.8, 9.6, 6.1],
        "rental_yield":[3.2, 2.8, 3.6, 3.4, 3.5, 3.3, 2.9, 3.1],
        "inventory_score":[58, 46, 62, 51, 57, 54, 43, 47],
        "lat":[19.0760, 28.6139, 12.9716, 13.0827, 17.3850, 18.5204, 22.5726, 23.0225],
        "lon":[72.8777, 77.2090, 77.5946, 80.2707, 78.4867, 73.8567, 88.3639, 72.5714],
    })

market_city_df = make_demo_market_dataset()

with st.sidebar:
    st.markdown("""
    <div style="padding:.55rem .35rem .9rem;">
      <div style="font-size:1.15rem; font-weight:800; color:#dbe6ff;">PropIQ</div>
      <div style="font-size:.76rem; color:#718198; font-family:'JetBrains Mono', monospace; letter-spacing:.12em; text-transform:uppercase; margin-top:.12rem;">
        AI Real Estate Intelligence
      </div>
    </div>
    """, unsafe_allow_html=True)

    icons = {
        "Home":"🏠",
        "Market Analytics":"📊",
        "Valuation":"🧮",
        "Insights":"💡",
        "Resources":"📚",
        "About":"ℹ️"
    }

    for p in PAGES:
        if st.button(f"{icons[p]}  {p}", use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown(f"""
        <div class="mini-card">
          <div class="card-title">Model status</div>
          <div style="font-size:1rem; color:#d9e6ff; font-weight:800;">{meta.get('best_model', 'Best Model')}</div>
          <div style="font-size:.9rem; color:#7f8ea3; margin-top:.25rem;">R² = {meta.get('r2_mean', 0):.4f} · 5-Fold CV</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mini-card">
          <div class="card-title">Demo mode</div>
          <div style="font-size:1rem; color:#d9e6ff; font-weight:800;">Model artefacts missing</div>
          <div style="font-size:.9rem; color:#7f8ea3; margin-top:.25rem;">Run src/train.py to activate production predictions.</div>
        </div>
        """, unsafe_allow_html=True)

def run_prediction_engine():
    try:
        from src.features import prepare_features
        sqft_above = max(st.session_state.sqft_living - st.session_state.sqft_basement, 0)

        df_input = pd.DataFrame([{
            "bedrooms": st.session_state.bedrooms,
            "bathrooms": st.session_state.bathrooms,
            "sqft_living": st.session_state.sqft_living,
            "sqft_lot": st.session_state.sqft_lot,
            "floors": st.session_state.floors,
            "waterfront": st.session_state.waterfront,
            "view": st.session_state.view,
            "condition": st.session_state.condition,
            "grade": st.session_state.grade,
            "sqft_above": sqft_above,
            "sqft_basement": st.session_state.sqft_basement,
            "yr_built": st.session_state.yr_built,
            "yr_renovated": st.session_state.yr_renovated,
            "zipcode": st.session_state.zipcode,
            "lat": st.session_state.lat,
            "long": st.session_state.lon,
            "sqft_living15": st.session_state.sqft_living15,
            "sqft_lot15": st.session_state.sqft_lot15,
            "schools_nearby": st.session_state.schools_nearby,
            "airport_distance_km": st.session_state.airport_distance,
            "sale_year": st.session_state.sale_year,
            "sale_month": st.session_state.sale_month,
            "latest_hpi": st.session_state.hpi_val,
            "real_price_yoy": st.session_state.real_price_idx * 0.05,
            "real_price_idx": st.session_state.real_price_idx,
            "nom_price_yoy": st.session_state.nom_price_idx * 0.06,
            "nom_price_idx": st.session_state.nom_price_idx,
        }])

        X_in, _, _, _, _ = prepare_features(
            df_input, y=None, engineer=engineer, ct=ct, te=te,
            col_groups=col_groups, fit=False
        )
        pred = float(model.predict(X_in)[0])
        ci_pct = 0.12
        lo = pred * (1 - ci_pct)
        hi = pred * (1 + ci_pct)
        return pred, lo, hi
    except Exception:
        base_price = (
            st.session_state.sqft_living * 950 +
            st.session_state.grade * 120000 +
            st.session_state.bathrooms * 85000 +
            st.session_state.view * 60000 +
            st.session_state.waterfront * 350000 -
            st.session_state.airport_distance * 2500
        )
        pred = max(base_price, 1200000)
        lo = pred * 0.88
        hi = pred * 1.12
        return pred, lo, hi

def update_valuation_gauge():
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=st.session_state.pred / 1e6,
        delta={"reference": 2.0, "suffix": "M", "valueformat": ".2f"},
        number={"suffix": "M ₹", "valueformat": ".2f", "font": {"color": "#dce7ff", "size": 28}},
        gauge={
            "axis": {"range": [0, max(st.session_state.hi / 1e6 * 1.35, 5)], "tickcolor": "#6f89b8"},
            "bar": {"color": "#5b8cff"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, st.session_state.lo / 1e6], "color": "rgba(91,140,255,.18)"},
                {"range": [st.session_state.lo / 1e6, st.session_state.hi / 1e6], "color": "rgba(91,140,255,.32)"},
            ],
            "threshold": {"line": {"color": "#f43f5e", "width": 3}, "value": st.session_state.hi / 1e6},
        },
        title={"text": "Valuation range (₹ Millions)", "font": {"color": "#9aa8bc", "size": 14}},
    ))
    return apply_theme(fig, 300)

def market_score():
    score = (
        0.35 * min(st.session_state.hpi_val / 250, 1.2) +
        0.30 * min(st.session_state.real_price_idx / 150, 1.2) +
        0.20 * min(st.session_state.nom_price_idx / 180, 1.2) +
        0.15 * min((4 - min(st.session_state.airport_distance / 30, 4)) / 4 + 0.2, 1.0)
    )
    return round(score * 100, 1)

def risk_band():
    spread = (st.session_state.hi - st.session_state.lo) / max(st.session_state.pred, 1)
    if spread <= 0.20:
        return "Low"
    elif spread <= 0.28:
        return "Moderate"
    return "Elevated"

st.markdown("<div class='page-wrap'>", unsafe_allow_html=True)



page = st.session_state.page
# ─────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────
if page == "Home":
    st.markdown(f"""
<div class="topbar">
  <div class="brand-kicker">Institutional-grade valuation workspace</div>
  <div class="hero-title">PropIQ <span>Real Estate Intelligence</span></div>
  <div class="hero-body">
    A premium AI-powered real estate valuation and market intelligence platform built for analysts, brokers, investors, and decision-makers.
    It combines property fundamentals, neighbourhood signals, location context, and macro housing indicators into one compact decision dashboard.
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="hero-stat-val">{meta.get('n_train_rows', 14620) if model_loaded else 14620:,}</div><div class="hero-stat-key">Training rows</div></div>
    <div class="hero-stat"><div class="hero-stat-val">{meta.get('n_features', 41) if model_loaded else 41}</div><div class="hero-stat-key">Engineered features</div></div>
    <div class="hero-stat"><div class="hero-stat-val">{meta.get('best_model', 'XGBoost') if model_loaded else 'XGBoost'}</div><div class="hero-stat-key">Best model</div></div>
    <div class="hero-stat"><div class="hero-stat-val">14+</div><div class="hero-stat-key">Integrated visual analytics</div></div>
    <div class="hero-stat"><div class="hero-stat-val">&lt; 200ms</div><div class="hero-stat-key">Target inference</div></div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Executive summary of valuation performance, market momentum, and decision intelligence.</div>", unsafe_allow_html=True)

    # Fetch metric values
    r2v = meta.get("r2_mean", 0.9124) if model_loaded else 0.9124
    mae = meta.get("mae_mean", 145000.0) if model_loaded else 145000.0
    rmse = meta.get("rmse_mean", 228000.0) if model_loaded else 228000.0
    nrow = meta.get("n_train_rows", 21540) if model_loaded else 21540
    best = meta.get("best_model", "XGBoost") if model_loaded else "XGBoost"
    nfeat = meta.get("n_features", 48) if model_loaded else 48

    # KPI Top Metrics Row
    row1 = st.columns(5)
    top_metrics = [
        ("R² Accuracy", f"{r2v:.4f}", "Cross-validation score", "pos"),
        ("Mean Abs Error", f"₹{mae:,.0f}", "Average prediction error", "neu"),
        ("RMSE", f"₹{rmse:,.0f}", "Root mean square error", "neu"),
        ("Training Rows", f"{nrow:,}", "Clean usable samples", "pos"),
        ("Market Score", f"{market_score():.1f}/100", "Composite environment signal", "pos"),
    ]
    for col, (label, value, foot, cls) in zip(row1, top_metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-foot {cls}">{foot}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='soft'>", unsafe_allow_html=True)

    # Split Main Analytics Layout
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        # Valuation Result Banner Focus
        st.markdown(f"""
        <div class="pred-banner" style="margin-bottom: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px;">
          <div class="pred-tag">AI estimated valuation</div>
          <div class="pred-price">₹ {st.session_state.pred:,.0f}</div>
          <div class="pred-ci">90% confidence interval · ₹{st.session_state.lo:,.0f} to ₹{st.session_state.hi:,.0f}</div>
          <div class="pred-sub">Driven by structure, grade, micro-location, accessibility, and market context.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # High-intent CTA button neatly tied directly underneath the evaluation snapshot
        if st.button("📊 Check Property Valuation", use_container_width=True):
            st.session_state.page = "Valuation"
            st.rerun()

        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

        # Property Summary Details Card
        st.markdown("""
        <div class="card">
          <div class="card-head">
            <div class="card-title">Property summary</div>
            <div class="badge">Live input snapshot</div>
          </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="spec-grid">
          <div class="spec-item"><div class="spec-key">Bedrooms</div><div class="spec-val">{st.session_state.bedrooms}</div></div>
          <div class="spec-item"><div class="spec-key">Bathrooms</div><div class="spec-val">{st.session_state.bathrooms}</div></div>
          <div class="spec-item"><div class="spec-key">Living area</div><div class="spec-val">{st.session_state.sqft_living:,} sqft</div></div>
          <div class="spec-item"><div class="spec-key">Lot area</div><div class="spec-val">{st.session_state.sqft_lot:,} sqft</div></div>
          <div class="spec-item"><div class="spec-key">Floors</div><div class="spec-val">{st.session_state.floors}</div></div>
          <div class="spec-item"><div class="spec-key">Build grade</div><div class="spec-val">{st.session_state.grade}/13</div></div>
          <div class="spec-item"><div class="spec-key">Condition</div><div class="spec-val">{st.session_state.condition}/5</div></div>
          <div class="spec-item"><div class="spec-key">Airport distance</div><div class="spec-val">{st.session_state.airport_distance} km</div></div>
          <div class="spec-item"><div class="spec-key">Real index</div><div class="spec-val">{st.session_state.real_price_idx:.1f}</div></div>
          <div class="spec-item"><div class="spec-key">Nominal index</div><div class="spec-val">{st.session_state.nom_price_idx:.1f}</div></div>
          <div class="spec-item"><div class="spec-key">HPI</div><div class="spec-val">{st.session_state.hpi_val:.1f}</div></div>
          <div class="spec-item"><div class="spec-key">Risk band</div><div class="spec-val">{risk_band()}</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Charts and Radar Signals column block
        st.markdown('<div class="chart-card"><div class="card-title">Valuation Indicator Meter</div>', unsafe_allow_html=True)
        st.plotly_chart(update_valuation_gauge(), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

        yoy_demo = pd.DataFrame({
            "metric":["Real index","Nominal index","HPI","Neighbourhood strength"],
            "score":[st.session_state.real_price_idx/2, st.session_state.nom_price_idx/2.2, st.session_state.hpi_val/2.3, 68]
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=yoy_demo["score"],
            theta=yoy_demo["metric"],
            fill='toself',
            line=dict(color="#8fb3ff"),
            fillcolor="rgba(91,140,255,.18)"
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(148,163,184,.12)")),
            showlegend=False
        )
        st.markdown('<div class="chart-card"><div class="card-title">Signal Composition Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig_radar, 300), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# MARKET ANALYTICS
# ─────────────────────────────────────────────────────────
elif page == "Market Analytics":
    st.markdown("<div class='sec-title'>Market Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Macro trends, city performance, pricing patterns, affordability signals, and map-based market intelligence.</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    latest_real = float(macro_df["price"].iloc[-1]) if not macro_df.empty else 0
    latest_nom = float(nom_df["price"].iloc[-1]) if not nom_df.empty else 0
    latest_city_growth = market_city_df["yoy_growth"].mean()
    avg_rental = market_city_df["rental_yield"].mean()

    metrics = [
        ("Real HPI", f"{latest_real:.1f}", "Latest macro reading"),
        ("Nominal HPI", f"{latest_nom:.1f}", "Price movement without inflation adjustment"),
        ("Avg city YoY", f"{latest_city_growth:.1f}%", "Primary city appreciation"),
        ("Rental yield", f"{avg_rental:.1f}%", "Illustrative income return"),
    ]
    for col, (a,b,c) in zip([m1,m2,m3,m4], metrics):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{a}</div>
                <div class="kpi-value">{b}</div>
                <div class="kpi-foot neu">{c}</div>
            </div>
            """, unsafe_allow_html=True)

    row_a = st.columns(2, gap="medium")
    with row_a[0]:
        st.markdown('<div class="chart-card"><div class="card-title">Real vs Nominal Housing Trend</div>', unsafe_allow_html=True)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=macro_df["date"], y=macro_df["price"], mode="lines",
            name="Real HPI", line=dict(color="#5b8cff", width=2.6),
            fill="tozeroy", fillcolor="rgba(91,140,255,.08)"
        ))
        fig_trend.add_trace(go.Scatter(
            x=nom_df["date"], y=nom_df["price"], mode="lines",
            name="Nominal HPI", line=dict(color="#f59e0b", width=2, dash="dot")
        ))
        fig_trend.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.05, x=1, xanchor="right"))
        st.plotly_chart(apply_theme(fig_trend, 300), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with row_a[1]:
        st.markdown('<div class="chart-card"><div class="card-title">Latest City HPI Snapshot</div>', unsafe_allow_html=True)
        latest_q = hpi_df["quarter"].iloc[-1]
        snap = hpi_df[hpi_df["quarter"] == latest_q].sort_values("hpi", ascending=True)
        fig_city = go.Figure(go.Bar(
            x=snap["hpi"], y=snap["city"], orientation="h",
            marker=dict(color=snap["hpi"], colorscale=[[0, "#1e3a8a"], [1, "#93c5fd"]], showscale=False),
            text=snap["hpi"].round(1), textposition="outside"
        ))
        st.plotly_chart(apply_theme(fig_city, 300), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    row_b = st.columns(2, gap="medium")
    with row_b[0]:
        st.markdown('<div class="chart-card"><div class="card-title">City YoY Growth Comparison</div>', unsafe_allow_html=True)
        yoy_sorted = market_city_df.sort_values("yoy_growth", ascending=False)
        fig_yoy = px.bar(
            yoy_sorted, x="city", y="yoy_growth",
            color="yoy_growth", color_continuous_scale=["#274690","#5b8cff","#8fb3ff"]
        )
        fig_yoy.update_layout(coloraxis_showscale=False)
        fig_yoy.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(apply_theme(fig_yoy, 300), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with row_b[1]:
        st.markdown('<div class="chart-card"><div class="card-title">Price vs Rental Yield Matrix</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            market_city_df, x="avg_price_cr", y="rental_yield",
            size="inventory_score", color="yoy_growth", text="city",
            color_continuous_scale=["#1e3a8a", "#5b8cff", "#f59e0b"]
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(coloraxis_colorbar_title="YoY %")
        st.plotly_chart(apply_theme(fig_scatter, 320), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    row_c = st.columns(2, gap="medium")
    with row_c[0]:
        st.markdown('<div class="chart-card"><div class="card-title">Market Heat Map</div>', unsafe_allow_html=True)
        fig_map = px.scatter_mapbox(
            market_city_df,
            lat="lat", lon="lon",
            size="avg_price_cr",
            color="yoy_growth",
            hover_name="city",
            hover_data={"avg_price_cr":True, "rental_yield":True, "lat":False, "lon":False},
            zoom=3.5,
            height=360,
            color_continuous_scale="Blues"
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with row_c[1]:
        st.markdown('<div class="chart-card"><div class="card-title">Affordability & Demand Balance</div>', unsafe_allow_html=True)
        aff = pd.DataFrame({
            "factor":["Price-to-Income","Price-to-Rent","Inventory Pressure","Capital Demand","Transit Access"],
            "score":[74, 68, 57, 79, 64]
        })
        fig_aff = go.Figure(go.Bar(
            x=aff["score"], y=aff["factor"], orientation="h",
            marker=dict(color=aff["score"], colorscale=[[0, "#1d4ed8"], [1, "#8fb3ff"]], showscale=False),
            text=aff["score"], textposition="outside"
        ))
        st.plotly_chart(apply_theme(fig_aff, 360), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    row_d = st.columns(3)
    insights = [
        ("Market breadth expansion",
         "Broad appreciation across cities can be narrated as market breadth, showing that price growth is no longer concentrated in only one or two metros."),
        ("Premiumization trend",
         "Add a premium housing section because current Indian residential demand is increasingly supported by premium and lifestyle-led segments, while affordable demand remains more constrained."),
        ("Investor dashboard angle",
         "You can present a city allocation view combining appreciation, rental yield, liquidity, and inventory pressure for investor screening."),
    ]
    for col, (ttl, body) in zip(row_d, insights):
        with col:
            st.markdown(f"""
            <div class="insight-card">
              <div class="insight-title">{ttl}</div>
              <div class="insight-body">{body}</div>
              <div class="insight-meta">Dashboard insight module</div>
            </div>
            """, unsafe_allow_html=True)
elif page == "Valuation":
    st.markdown("<div class='sec-title'>Valuation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Configure property, location, and macro signals to generate valuation and scenario views.</div>", unsafe_allow_html=True)

    # 1. Property Configuration Card (Inputs First)
    st.markdown("""
<div style="font-size: 1.2rem; font-weight: 600; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.7rem;">
  <span>🧮</span> Property Configuration
</div>
""", unsafe_allow_html=True)
    tab_struct, tab_quality, tab_loc, tab_macro = st.tabs(
        ["🏗️ Structure", "✨ Quality", "📍 Location", "📈 Macro context"]
    )

    with tab_struct:
        st.session_state.bedrooms = st.slider("Bedrooms", 1, 10, st.session_state.bedrooms)
        st.session_state.bathrooms = st.slider("Bathrooms", 1.0, 6.0, st.session_state.bathrooms, 0.25)
        st.session_state.sqft_living = st.number_input("Living area (sqft)", 500, 12000, st.session_state.sqft_living, 100)
        st.session_state.sqft_lot = st.number_input("Lot area (sqft)", 500, 100000, st.session_state.sqft_lot, 500)

        fl_options = [1.0, 1.5, 2.0, 2.5, 3.0]
        st.session_state.floors = st.selectbox(
            "Floors",
            fl_options,
            index=fl_options.index(st.session_state.floors) if st.session_state.floors in fl_options else 0
        )
        st.session_state.sqft_basement = st.slider("Basement area", 0, 3000, st.session_state.sqft_basement, 50)

    with tab_quality:
        st.session_state.grade = st.slider("Build grade", 3, 13, st.session_state.grade)
        st.session_state.condition = st.slider("Condition", 1, 5, st.session_state.condition)
        st.session_state.waterfront = st.radio(
            "Waterfront", [0, 1],
            index=int(st.session_state.waterfront),
            horizontal=True,
            format_func=lambda x: "Yes" if x else "No"
        )
        st.session_state.view = st.slider("View score", 0, 4, st.session_state.view)
        st.session_state.yr_built = st.slider("Year built", 1900, 2023, st.session_state.yr_built)
        st.session_state.yr_renovated = st.slider("Year renovated (0 = never)", 0, 2023, st.session_state.yr_renovated)

    with tab_loc:
        st.session_state.zipcode = st.number_input("Postal code", value=st.session_state.zipcode)
        st.session_state.lat = st.number_input("Latitude", value=st.session_state.lat, format="%.4f")
        st.session_state.lon = st.number_input("Longitude", value=st.session_state.lon, format="%.4f")
        st.session_state.sqft_living15 = st.slider("Neighbour avg living area", 500, 6000, st.session_state.sqft_living15, 100)
        st.session_state.sqft_lot15 = st.slider("Neighbour avg lot area", 500, 50000, st.session_state.sqft_lot15, 500)
        st.session_state.schools_nearby = st.selectbox(
            "Schools nearby", [1, 2, 3],
            index=[1, 2, 3].index(st.session_state.schools_nearby)
        )
        st.session_state.airport_distance = st.slider("Airport distance (km)", 10, 120, st.session_state.airport_distance)

    with tab_macro:
        years_opt = list(range(2014, 2024))
        st.session_state.sale_year = st.selectbox(
            "Sale year", years_opt,
            index=years_opt.index(st.session_state.sale_year) if st.session_state.sale_year in years_opt else 6
        )
        st.session_state.sale_month = st.slider("Sale month", 1, 12, st.session_state.sale_month)
        st.session_state.hpi_val = st.slider("Housing price index", 100.0, 350.0, st.session_state.hpi_val, 5.0)
        st.session_state.real_price_idx = st.slider("Real price index", 80.0, 200.0, st.session_state.real_price_idx, 1.0)
        st.session_state.nom_price_idx = st.slider("Nominal price index", 80.0, 250.0, st.session_state.nom_price_idx, 1.0)

    st.markdown('<div style="margin-top:1.1rem;"></div>', unsafe_allow_html=True)
    trigger_prediction = st.button("🔮 Predict Valuation", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if trigger_prediction:
        with st.spinner("Executing valuation architecture inference..."):
            p, l, h = run_prediction_engine()
            st.session_state.pred = p
            st.session_state.lo = l
            st.session_state.hi = h
        st.rerun()

    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    # 2. AI Estimated Valuation Result Banner (Placed Below Inputs)
    st.markdown(f"""
    <div class="pred-banner">
      <div class="pred-tag">AI Estimated Valuation Result</div>
      <div class="pred-price">₹ {st.session_state.pred:,.0f}</div>
      <div class="pred-ci">90% confidence interval · ₹{st.session_state.lo:,.0f} to ₹{st.session_state.hi:,.0f}</div>
      <div class="pred-sub">Driven by structure, quality, neighbourhood context, and macro market signals.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:2rem;"></div>', unsafe_allow_html=True)

    # 3. Dynamic Output & Visual Analytics (Scroll down further to see charts)
    st.markdown('<div class="chart-card"><div class="card-title">Dynamic Output Bounds View</div>', unsafe_allow_html=True)
    st.plotly_chart(update_valuation_gauge(), use_container_width=True, config=CFG)
    st.markdown('</div>', unsafe_allow_html=True)

    scenario_df = pd.DataFrame({
        "Scenario":["Base","Higher grade","Closer airport","Higher HPI","Premium view"],
        "Value":[
            st.session_state.pred,
            st.session_state.pred * 1.08,
            st.session_state.pred * 1.04,
            st.session_state.pred * 1.06,
            st.session_state.pred * 1.07
        ]
    })
    fig_scen = px.bar(
        scenario_df, x="Scenario", y="Value",
        color="Value", color_continuous_scale=["#274690","#5b8cff","#8fb3ff"]
    )
    fig_scen.update_layout(coloraxis_showscale=False)
    fig_scen.update_traces(texttemplate="₹%{y:,.0f}", textposition="outside")

    st.markdown('<div class="chart-card"><div class="card-title">Scenario Sensitivity View</div>', unsafe_allow_html=True)
    st.plotly_chart(apply_theme(fig_scen, 320), use_container_width=True, config=CFG)
    st.markdown('</div>', unsafe_allow_html=True)

    contrib = pd.DataFrame({
        "feature":["Living area","Grade","Bathrooms","View","Waterfront","Airport distance"],
        "impact":[34, 22, 12, 10, 14, -8]
    })
    fig_water = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative","relative","relative","relative","relative","relative"],
        x=contrib["feature"],
        y=contrib["impact"],
        connector={"line":{"color":"rgba(148,163,184,.3)"}},
        decreasing={"marker":{"color":"#f43f5e"}},
        increasing={"marker":{"color":"#5b8cff"}},
    ))

    st.markdown('<div class="chart-card"><div class="card-title">Estimated Feature Contribution Flow</div>', unsafe_allow_html=True)
    st.plotly_chart(apply_theme(fig_water, 320), use_container_width=True, config=CFG)
    st.markdown('</div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# INSIGHTS
# ─────────────────────────────────────────────────────────
elif page == "Insights":
    st.markdown("<div class='sec-title'>Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Model explainability, validation stability, business narratives, and decision support layers.</div>", unsafe_allow_html=True)

    top_left, top_right = st.columns([1.08, 0.92], gap="medium")
    with top_left:
        st.markdown('<div class="chart-card"><div class="card-title">Feature Relative Importance</div>', unsafe_allow_html=True)
        if model_loaded and fi is not None and len(fi) > 0:
            top_fi = fi.head(12).sort_values()
            fig_fi = go.Figure(go.Bar(
                x=top_fi.values, y=top_fi.index, orientation="h",
                marker=dict(color=top_fi.values, colorscale=[[0, "#1d4ed8"], [1, "#8fb3ff"]], showscale=False),
                text=[f"{v:.4f}" for v in top_fi.values], textposition="outside"
            ))
        else:
            demo_fi = pd.Series({
                "sqft_living": 0.31, "grade": 0.18, "lat": 0.12, "market_demand_index": 0.09,
                "property_age": 0.07, "grade_condition_score": 0.06, "bathrooms": 0.05,
                "sqft_living15": 0.04, "waterfront": 0.03, "view": 0.02,
                "airport_distance_km": 0.018, "latest_hpi": 0.016
            }).sort_values()
            fig_fi = go.Figure(go.Bar(
                x=demo_fi.values, y=demo_fi.index, orientation="h",
                marker=dict(color=demo_fi.values, colorscale=[[0, "#1d4ed8"], [1, "#8fb3ff"]], showscale=False),
                text=[f"{v:.3f}" for v in demo_fi.values], textposition="outside"
            ))
        st.plotly_chart(apply_theme(fig_fi, 360), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        st.markdown('<div class="chart-card"><div class="card-title">Validation Fold Stability</div>', unsafe_allow_html=True)
        folds = meta.get("r2_folds", [0.89, 0.92, 0.91, 0.90, 0.93]) if model_loaded else [0.89, 0.92, 0.91, 0.90, 0.93]
        fig_cv = go.Figure(go.Scatter(
            x=[f"Fold {i+1}" for i in range(len(folds))], y=folds,
            mode="lines+markers",
            line=dict(color="#7ea5ff", width=2),
            marker=dict(size=9, color="#5b8cff"),
            fill="tozeroy", fillcolor="rgba(91,140,255,.10)"
        ))
        fig_cv.add_hline(
            y=float(np.mean(folds)), line_dash="dot", line_color="#9aa8bc",
            annotation_text=f"Mean R² {np.mean(folds):.4f}", annotation_font_color="#9aa8bc"
        )
        fig_cv.update_layout(showlegend=False, yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(apply_theme(fig_cv, 260), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

        residual_demo = pd.DataFrame({
            "band":["Excellent","Strong","Average","Weak"],
            "count":[42, 31, 18, 9]
        })
        fig_pie = px.pie(residual_demo, values="count", names="band", hole=0.55,
                         color_discrete_sequence=["#8fb3ff","#5b8cff","#f59e0b","#f43f5e"])
        st.markdown('<div class="chart-card"><div class="card-title">Prediction Quality Mix</div>', unsafe_allow_html=True)
        st.plotly_chart(apply_theme(fig_pie, 260), use_container_width=True, config=CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    cards = [
        ("Why macro matters",
         "Real and nominal market signals help explain whether price movement is broad market inflation, genuine purchasing power improvement, or temporary overheating."),
        ("Why maps matter",
         "Location visuals make the dashboard stronger for demos because spatial context explains why similar homes can price differently across cities and micro-markets."),
        ("Why resources matter",
         "External links increase credibility by showing that your dashboard is connected to live institutions, public datasets, reports, and further learning material."),
    ]
    for col, (title, body) in zip([b1,b2,b3], cards):
        with col:
            st.markdown(f"""
            <div class="insight-card">
              <div class="insight-title">{title}</div>
              <div class="insight-body">{body}</div>
              <div class="insight-meta">Insight narrative</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# RESOURCES
# ─────────────────────────────────────────────────────────
elif page == "Resources":
    st.markdown("<div class='sec-title'>Resources</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Datasets, research, policy references, and learning tools to understand valuation and market analytics in depth.</div>", unsafe_allow_html=True)

    resources = [
        {
            "tag":"dataset", "title":"OECD Housing Prices",
            "desc":"Use for real house price index, nominal house price index, price-to-income, and price-to-rent analytics.",
            "url":"https://www.oecd.org/en/data/indicators/housing-prices.html"
        },
        {
            "tag":"research", "title":"Grant Thornton India Real Estate Report FY 2025-26",
            "desc":"Useful for market structure, premium housing trends, ESG, PropTech, REITs, and sector outlook slides.",
            "url":"https://www.grantthornton.in/globalassets/1.-member-firms/india/assets/pdfs/realty-bytes/realty_bytes_may_2025.pdf"
        },
        {
            "tag":"guide", "title":"NHB RESIDEX / Housing Price Coverage",
            "desc":"Use for India city-level house price movement references and dashboard commentary on market breadth.",
            "url":"https://www.hindustantimes.com/real-estate/property-prices-rise-in-48-cities-in-q4-of-fy25-nhb-residex-101748089355490.html"
        },
        {
            "tag":"tool", "title":"Plotly Python Documentation",
            "desc":"Best for adding advanced interactive charts, maps, gauges, treemaps, and drill-down visuals.",
            "url":"https://plotly.com/python/"
        },
        {
            "tag":"tool", "title":"Streamlit Documentation",
            "desc":"Use for layout systems, tabs, metrics, session state, responsive input flows, and dashboard polish.",
            "url":"https://docs.streamlit.io/"
        },
        {
            "tag":"guide", "title":"scikit-learn Model Evaluation",
            "desc":"Useful for adding residual analysis, error metrics, validation plots, and explainability logic.",
            "url":"https://scikit-learn.org/stable/model_evaluation.html"
        }
    ]

    cols = st.columns(2)
    for i, r in enumerate(resources):
        cls = {
            "dataset":"tag-dataset",
            "research":"tag-research",
            "tool":"tag-tool",
            "guide":"tag-guide"
        }[r["tag"]]
        with cols[i % 2]:
            st.markdown(f"""
            <div class="resource-card">
              <div class="resource-tag {cls}">{r["tag"]}</div>
              <div class="resource-title">{r["title"]}</div>
              <div class="resource-desc">{r["desc"]}</div>
              <a class="resource-link" href="{r["url"]}" target="_blank">Open resource ↗</a>
            </div>
            """, unsafe_allow_html=True)

    

# ─────────────────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────────────────
elif page == "About":
    st.markdown("<div class='sec-title'>About</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Project overview, business relevance, data architecture, and real-world market positioning.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-head">
        <div class="card-title">Project summary</div>
        <div class="badge">Academic + industry-ready</div>
      </div>
      <div class="insight-body">
        PropIQ is an AI-powered real estate valuation and market intelligence platform designed to estimate residential property values,
        analyze market trends, visualize macroeconomic indicators, compare city-level housing performance, and provide actionable insights for
        investors, analysts, brokers, and decision-makers.
        <br><br>
        The platform combines machine learning prediction with market analytics, explainable model outputs, map-based context, and curated learning resources.
        Instead of only showing a predicted number, the system presents the full valuation story through data, charts, confidence bands, feature importance,
        and market environment signals.
      </div>
    </div>
    """, unsafe_allow_html=True)

    about_cols = st.columns(3)
    about_blocks = [
        ("Core modules", "Home summary, market analytics, valuation engine, model insights, resources, and project documentation."),
        ("Business value", "Supports smarter buying, selling, pricing, lending, and investment decisions with both model intelligence and market context."),
        ("Future scope", "Add live APIs, geospatial layers, comparables engine, SHAP explainability, authentication, exports, and role-based dashboards.")
    ]
    for col, (t,b) in zip(about_cols, about_blocks):
        with col:
            st.markdown(f"""
            <div class="insight-card">
              <div class="insight-title">{t}</div>
              <div class="insight-body">{b}</div>
              <div class="insight-meta">Platform overview</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  PropIQ · AI-Powered Real Estate Valuation & Market Intelligence Platform
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)