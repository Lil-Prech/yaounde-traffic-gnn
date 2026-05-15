# ═══════════════════════════════════════════════════════════
# Yaoundé Real-Time Traffic Congestion Prediction System
# Live Dashboard — Phase 5
# ═══════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import sqlite3
import folium
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from streamlit_folium import st_folium
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Yaoundé Traffic · Real-Time Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────
BASE      = r"C:\Users\LUM\Documents\yaoundé-traffic"
DATA      = os.path.join(BASE, "data")
PROCESSED = os.path.join(BASE, "processed")
MODELS    = os.path.join(BASE, "models")
DB_PATH   = os.path.join(DATA, "traffic.db")

# ── CSS Styling ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: #0A0E1A;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0F1525;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 14px;
    color: #94A3B8;
    margin: 0 0 20px 0;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(56,189,248,0.12);
    color: #38BDF8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid rgba(56,189,248,0.2);
    font-family: 'DM Mono', monospace;
}
.badge-green {
    background: rgba(34,197,94,0.12);
    color: #22C55E;
    border-color: rgba(34,197,94,0.2);
}
.badge-orange {
    background: rgba(251,146,60,0.12);
    color: #FB923C;
    border-color: rgba(251,146,60,0.2);
}

/* Section headers */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* Corridor cards */
.corridor-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 22px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.corridor-card:hover {
    border-color: rgba(56,189,248,0.2);
}
.corridor-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.card-free::before  { background: linear-gradient(90deg, #22C55E, #16A34A); }
.card-light::before { background: linear-gradient(90deg, #EAB308, #CA8A04); }
.card-mod::before   { background: linear-gradient(90deg, #F97316, #EA580C); }
.card-heavy::before { background: linear-gradient(90deg, #EF4444, #DC2626); }
.card-severe::before{ background: linear-gradient(90deg, #DC2626, #991B1B); }

.corridor-name {
    font-size: 16px;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 4px;
}
.corridor-route {
    font-size: 12px;
    color: #64748B;
    font-family: 'DM Mono', monospace;
    margin-bottom: 18px;
}
.index-value {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.status-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pill-free   { background: rgba(34,197,94,0.15);  color: #22C55E; }
.pill-light  { background: rgba(234,179,8,0.15);  color: #EAB308; }
.pill-mod    { background: rgba(249,115,22,0.15); color: #F97316; }
.pill-heavy  { background: rgba(239,68,68,0.15);  color: #EF4444; }
.pill-severe { background: rgba(220,38,38,0.15);  color: #DC2626; }

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stat-row:last-child { border-bottom: none; }
.stat-label { font-size: 12px; color: #64748B; }
.stat-value { font-size: 13px; color: #CBD5E1; font-weight: 500; font-family: 'DM Mono', monospace; }

/* Forecast bar */
.forecast-section { margin-top: 14px; }
.forecast-label { font-size: 11px; color: #64748B; margin-bottom: 6px; }
.forecast-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.forecast-time { font-size: 11px; color: #94A3B8; width: 40px; font-family: 'DM Mono', monospace; }
.forecast-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}
.forecast-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.forecast-pct { font-size: 11px; color: #94A3B8; width: 38px; text-align: right; font-family: 'DM Mono', monospace; }

/* Stat boxes */
.stat-box {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.stat-box-num {
    font-size: 28px;
    font-weight: 700;
    color: #38BDF8;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
}
.stat-box-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 500;
}

/* Simulator */
.sim-result {
    background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 14px;
    padding: 24px;
    margin-top: 20px;
}
.sim-result-title {
    font-size: 14px;
    font-weight: 600;
    color: #38BDF8;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Override streamlit defaults */
.stSlider > div > div > div { background: #38BDF8 !important; }
.stSelectbox > div > div { background: #111827 !important; color: #E2E8F0 !important; border-color: rgba(255,255,255,0.1) !important; }
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #38BDF8, #0EA5E9) !important;
    transform: translateY(-1px);
}
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label { color: #64748B !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────
def get_db():
    return sqlite3.connect(DB_PATH)

@st.cache_data(ttl=60)
def get_latest_polls():
    conn = get_db()
    df   = pd.read_sql_query("""
        SELECT * FROM api_polls
        WHERE timestamp = (SELECT MAX(timestamp) FROM api_polls)
        ORDER BY corridor
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_latest_predictions():
    conn = get_db()
    df   = pd.read_sql_query("""
        SELECT * FROM predictions
        WHERE timestamp = (SELECT MAX(timestamp) FROM predictions)
        ORDER BY corridor
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_latest_snapshots():
    conn = get_db()
    df   = pd.read_sql_query("""
        SELECT * FROM graph_snapshots
        WHERE timestamp = (SELECT MAX(timestamp) FROM graph_snapshots)
        ORDER BY corridor
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_historical():
    conn = get_db()
    df   = pd.read_sql_query("""
        SELECT g.timestamp, g.corridor, g.congestion_index,
               g.is_congested, g.predicted_congestion,
               p.predicted_15min, p.predicted_30min
        FROM graph_snapshots g
        LEFT JOIN predictions p
            ON g.timestamp = p.timestamp
            AND g.corridor  = p.corridor
        ORDER BY g.timestamp ASC
    """, conn)
    conn.close()
    return df

@st.cache_data(ttl=60)
def get_counts():
    conn  = get_db()
    polls = pd.read_sql_query("SELECT COUNT(*) as n FROM api_polls", conn).iloc[0]['n']
    preds = pd.read_sql_query("SELECT COUNT(*) as n FROM predictions", conn).iloc[0]['n']
    snaps = pd.read_sql_query("SELECT COUNT(*) as n FROM graph_snapshots", conn).iloc[0]['n']
    conn.close()
    return polls, preds, snaps

def corridor_style(index):
    if index >= 5.0:   return "#DC2626", "card-severe", "pill-severe", "Severe"
    elif index >= 3.0: return "#EF4444", "card-heavy",  "pill-heavy",  "Heavy"
    elif index >= 1.5: return "#F97316", "card-mod",    "pill-mod",    "Moderate"
    elif index >= 1.1: return "#EAB308", "card-light",  "pill-light",  "Light"
    else:              return "#22C55E", "card-free",   "pill-free",   "Free flow"

def map_color(index):
    if index >= 5.0:   return "#DC2626"
    elif index >= 3.0: return "#EF4444"
    elif index >= 1.5: return "#F97316"
    elif index >= 1.1: return "#EAB308"
    else:              return "#22C55E"

# ── Load data ─────────────────────────────────────────────
polls     = get_latest_polls()
preds     = get_latest_predictions()
snapshots = get_latest_snapshots()
hist      = get_historical()
poll_count, pred_count, snap_count = get_counts()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size:18px;font-weight:700;color:#F1F5F9;margin-bottom:4px;'>
            🚦 Yaoundé Traffic
        </div>
        <div style='font-size:12px;color:#64748B;font-family:DM Mono,monospace;'>
            Real-Time Prediction System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Study corridors**")
    corridors_info = {
        "Soa":      {"dist": "20.2 km", "base": "21.9 min"},
        "Mfou":     {"dist": "28.6 km", "base": "28.2 min"},
        "Mbankomo": {"dist": "17.4 km", "base": "19.4 min"},
    }
    for c, info in corridors_info.items():
        st.markdown(f"**{c}** — {info['dist']} · baseline {info['base']}")

    st.markdown("---")
    st.markdown("**Congestion index**")
    st.markdown("🟢 `< 1.1` Free flow")
    st.markdown("🟡 `1.1–1.5` Light")
    st.markdown("🟠 `1.5–3.0` Moderate")
    st.markdown("🔴 `3.0–5.0` Heavy")
    st.markdown("🔴 `> 5.0` Severe")

    st.markdown("---")
    st.markdown("**Database**")
    st.markdown(f"API polls: {poll_count}")
    st.markdown(f"Predictions: {pred_count}")
    st.markdown(f"Graph snapshots: {snap_count}")

    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Graph Attention Network")
    st.markdown("3 layers · 4 attention heads")
    st.markdown("14 input features")

    st.markdown("---")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ── Hero banner ───────────────────────────────────────────
now_str = datetime.now().strftime("%A, %B %d %Y · %H:%M:%S")
st.markdown(f"""
<div class="hero">
    <div class="hero-title">Yaoundé Traffic Congestion Prediction</div>
    <div class="hero-sub">{now_str}</div>
    <div class="hero-badges">
        <span class="badge">Graph Attention Network</span>
        <span class="badge">26,215 road nodes</span>
        <span class="badge">64,213 road edges</span>
        <span class="badge badge-green">System active</span>
        <span class="badge badge-orange">Polling every 10 min</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick stats ───────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-box-num">3</div>
        <div class="stat-box-label">Active corridors</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-box-num">{poll_count}</div>
        <div class="stat-box-label">Total API polls</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-box-num">{pred_count}</div>
        <div class="stat-box-label">Predictions made</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-box-num">10m</div>
        <div class="stat-box-label">Poll interval</div>
    </div>""", unsafe_allow_html=True)

# ── Live corridor status ───────────────────────────────────
st.markdown('<div class="section-header">Live corridor status</div>',
            unsafe_allow_html=True)

corridors = ["Soa", "Mfou", "Mbankomo"]
routes    = {"Soa": "Soa → Yaoundé · 20.2 km",
             "Mfou": "Mfou → Yaoundé · 28.6 km",
             "Mbankomo": "Mbankomo → Yaoundé · 17.4 km"}
cols = st.columns(3)

for col, corridor in zip(cols, corridors):
    snap = snapshots[snapshots["corridor"] == corridor]
    pred = preds[preds["corridor"] == corridor]
    poll = polls[polls["corridor"] == corridor]

    idx      = float(snap.iloc[0]["congestion_index"]) if not snap.empty else 1.0
    prob_now = float(pred.iloc[0]["predicted_now"])    if not pred.empty else 0.05
    prob_15  = float(pred.iloc[0]["predicted_15min"])  if not pred.empty else 0.06
    prob_30  = float(pred.iloc[0]["predicted_30min"])  if not pred.empty else 0.07
    duration = float(poll.iloc[0]["duration_mins"])    if not poll.empty else 0
    speed    = float(poll.iloc[0]["avg_speed_kmh"])    if not poll.empty else 0

    color, card_cls, pill_cls, label = corridor_style(idx)
    bar_color = "#22C55E" if prob_now < 0.3 else "#F97316" if prob_now < 0.6 else "#EF4444"
    bar_now   = min(int(prob_now * 100), 100)
    bar_15    = min(int(prob_15 * 100), 100)
    bar_30    = min(int(prob_30 * 100), 100)

    with col:
        st.markdown(f"#### {corridor}")
        st.caption(routes[corridor])
        st.markdown(
            f"<div style='font-size:48px;font-weight:700;color:{color};"
            f"font-family:monospace;line-height:1;margin-bottom:8px;'>"
            f"{idx:.3f}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<span style='background:rgba(34,197,94,0.15);color:#22C55E;"
            f"padding:3px 12px;border-radius:20px;font-size:12px;"
            f"font-weight:700;text-transform:uppercase;'>{label}</span>"
            if label == "Free flow" else
            f"<span style='background:rgba(239,68,68,0.15);color:#EF4444;"
            f"padding:3px 12px;border-radius:20px;font-size:12px;"
            f"font-weight:700;text-transform:uppercase;'>{label}</span>",
            unsafe_allow_html=True
        )
        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Travel time", f"{duration:.1f} min")
        with col_b:
            st.metric("Avg speed", f"{speed:.1f} km/h")

        st.markdown("**GAT Congestion Probability**")
        st.markdown(
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:11px;color:#64748B;'>Now</span> "
            f"<span style='float:right;font-size:11px;color:#94A3B8;'>{prob_now:.1%}</span>"
            f"</div>"
            f"<div style='height:6px;background:rgba(255,255,255,0.06);"
            f"border-radius:3px;overflow:hidden;margin-bottom:8px;'>"
            f"<div style='width:{bar_now}%;height:100%;background:{bar_color};"
            f"border-radius:3px;'></div></div>"
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:11px;color:#64748B;'>+15 min</span> "
            f"<span style='float:right;font-size:11px;color:#94A3B8;'>{prob_15:.1%}</span>"
            f"</div>"
            f"<div style='height:6px;background:rgba(255,255,255,0.06);"
            f"border-radius:3px;overflow:hidden;margin-bottom:8px;'>"
            f"<div style='width:{bar_15}%;height:100%;background:{bar_color};"
            f"opacity:0.75;border-radius:3px;'></div></div>"
            f"<div style='margin-bottom:4px;'>"
            f"<span style='font-size:11px;color:#64748B;'>+30 min</span> "
            f"<span style='float:right;font-size:11px;color:#94A3B8;'>{prob_30:.1%}</span>"
            f"</div>"
            f"<div style='height:6px;background:rgba(255,255,255,0.06);"
            f"border-radius:3px;overflow:hidden;'>"
            f"<div style='width:{bar_30}%;height:100%;background:{bar_color};"
            f"opacity:0.5;border-radius:3px;'></div></div>",
            unsafe_allow_html=True
        )
# ── Live map ──────────────────────────────────────────────
st.markdown('<div class="section-header">Live road network map</div>',
            unsafe_allow_html=True)

m = folium.Map(
    location=[3.8480, 11.5021], zoom_start=11,
    tiles="CartoDB positron"
)

corridor_lines = {
    "Soa"     : [[3.9957, 11.5215], [3.8480, 11.5021]],
    "Mfou"    : [[3.7244, 11.6380], [3.8480, 11.5021]],
    "Mbankomo": [[3.8667, 11.3833], [3.8480, 11.5021]],
}

for corridor, coords in corridor_lines.items():
    snap  = snapshots[snapshots["corridor"] == corridor]
    idx   = snap.iloc[0]["congestion_index"] if not snap.empty else 1.0
    color = map_color(idx)
    _, _, _, label = corridor_style(idx)

    folium.PolyLine(
        coords, weight=6, color=color, opacity=0.9,
        tooltip=f"{corridor} — {label} (index: {idx:.3f})"
    ).add_to(m)
    folium.CircleMarker(
        location=coords[0], radius=12,
        color=color, fill=True, fill_color=color,
        fill_opacity=1.0,
        tooltip=f"{corridor} origin"
    ).add_to(m)
    folium.Marker(
        location=coords[0],
        icon=folium.DivIcon(
            html=f"""<div style="font-size:12px;font-weight:700;
                     background:rgba(0,0,0,0.8);color:{color};
                     padding:4px 10px;border-radius:6px;
                     border:1px solid {color};
                     white-space:nowrap;">{corridor}</div>""",
            icon_size=(90, 26), icon_anchor=(-5, 32)
        )
    ).add_to(m)

# City centre
folium.Marker(
    location=[3.8480, 11.5021],
    tooltip="Yaoundé City Centre",
    icon=folium.Icon(color="white", icon="star", prefix="fa")
).add_to(m)

# Bottlenecks
for lat, lon, label in [
    (3.8668, 11.5129, "Bottleneck #1 — Blvd Rodolphe Manga Bell · betweenness: 0.229"),
    (3.8673, 11.5126, "Bottleneck #2 — Blvd Rodolphe Manga Bell · betweenness: 0.229"),
    (3.8690, 11.5100, "Bottleneck #3 — Rue Martin Paul Samba · betweenness: 0.227"),
]:
    folium.CircleMarker(
        location=[lat, lon], radius=9,
        color="#FFFFFF", weight=2,
        fill=True, fill_color="#EF4444",
        fill_opacity=1.0, tooltip=label
    ).add_to(m)

st_folium(m, width="100%", height=600)

# ── Historical trends ─────────────────────────────────────
st.markdown('<div class="section-header">Historical congestion trends</div>',
            unsafe_allow_html=True)

if not hist.empty:
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    colors = {"Soa": "#EF4444", "Mfou": "#FB923C", "Mbankomo": "#38BDF8"}
    for corridor in ["Soa", "Mfou", "Mbankomo"]:
        subset = hist[hist["corridor"] == corridor].sort_values("timestamp")
        if not subset.empty:
            ax.plot(subset["timestamp"], subset["congestion_index"],
                    label=corridor, color=colors[corridor],
                    linewidth=2.5, marker="o", markersize=5,
                    markerfacecolor=colors[corridor], markeredgewidth=0)
            ax.fill_between(subset["timestamp"], subset["congestion_index"],
                            alpha=0.08, color=colors[corridor])

    ax.axhline(y=1.3, color="#22C55E", linestyle="--",
               linewidth=1, alpha=0.5, label="Congestion threshold (1.3)")
    ax.axhline(y=3.0, color="#F97316", linestyle="--",
               linewidth=1, alpha=0.5, label="Heavy threshold (3.0)")
    ax.axhline(y=5.0, color="#EF4444", linestyle="--",
               linewidth=1, alpha=0.5, label="Severe threshold (5.0)")

    ax.set_xlabel("Time", color="#64748B", fontsize=10)
    ax.set_ylabel("Congestion Index", color="#64748B", fontsize=10)
    ax.tick_params(colors="#64748B", labelsize=9)
    ax.spines["bottom"].set_color("#1E293B")
    ax.spines["left"].set_color("#1E293B")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.1, color="#334155")
    ax.legend(fontsize=9, facecolor="#1E293B", edgecolor="#334155",
              labelcolor="#CBD5E1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Historical trends will appear here as the system accumulates data.")

# ── Scenario simulator ────────────────────────────────────
st.markdown('<div class="section-header">Scenario simulator</div>',
            unsafe_allow_html=True)

st.markdown("""
<p style="color:#64748B;font-size:13px;margin-bottom:20px;">
Simulate different traffic conditions and see how the GAT model responds.
Adjust the corridor, congestion level and day to explore predictions.
</p>
""", unsafe_allow_html=True)

sim1, sim2, sim3 = st.columns(3)
with sim1:
    sim_corridor = st.selectbox("Corridor", ["Soa", "Mfou", "Mbankomo"],
                                key="sim_corridor")
with sim2:
    sim_index = st.slider("Congestion index", 1.0, 10.0, 3.5, 0.1,
                          key="sim_index")
with sim3:
    sim_day = st.selectbox("Day of week",
                           ["Monday","Tuesday","Wednesday","Thursday",
                            "Friday","Saturday","Sunday"],
                           key="sim_day")

run_sim = st.button("Run simulation", key="run_sim")

if run_sim:
    day_map      = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                    "Friday":4,"Saturday":5,"Sunday":6}
    corridor_map = {"Soa":0,"Mfou":1,"Mbankomo":2}
    travel_df    = pd.read_csv(os.path.join(PROCESSED,
                               "household_travel_features.csv"))
    hist_row     = travel_df[travel_df["corridor"] == sim_corridor].iloc[0]
    day_num      = day_map[sim_day]
    is_weekend   = 1 if day_num >= 5 else 0

    sim_features = pd.DataFrame([{
        "avg_congestion_index"     : sim_index,
        "max_congestion_index"     : sim_index * 1.2,
        "total_emission_kg"        : sim_index * 50,
        "pct_stretches_congested"  : 1.0 if sim_index > 1.3 else 0.0,
        "total_daily_vehicles"     : max(0, (10 - sim_index) * 100),
        "peak_hour_vehicles"       : max(0, (10 - sim_index) * 40),
        "avg_trip_duration_min"    : float(hist_row["avg_trip_duration_min"]) * sim_index,
        "transport_cost_burden_pct": float(hist_row["transport_cost_burden_pct"]),
        "pct_private_car"          : float(hist_row["pct_private_car"]),
        "pct_moto_taxi"            : float(hist_row["pct_moto_taxi"]),
        "pct_minibus"              : float(hist_row["pct_minibus"]),
        "is_weekend"               : float(is_weekend),
        "day_encoded"              : float(day_num),
        "corridor_encoded"         : float(corridor_map[sim_corridor])
    }])

    scaler     = MinMaxScaler()
    sim_scaled = scaler.fit_transform(sim_features)
    X          = torch.tensor(sim_scaled, dtype=torch.float)
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)

    class TrafficGAT(nn.Module):
        def __init__(self, ic, hc, oc, heads=4, dropout=0.3):
            super().__init__()
            self.dropout = dropout
            self.conv1   = GATConv(ic, hc, heads=heads, dropout=dropout)
            self.conv2   = GATConv(hc*heads, hc, heads=heads, dropout=dropout)
            self.conv3   = GATConv(hc*heads, oc, heads=1, concat=False, dropout=dropout)
            self.bn1     = nn.BatchNorm1d(hc*heads)
            self.bn2     = nn.BatchNorm1d(hc*heads)
        def forward(self, x, ei):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.bn1(self.conv1(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.bn2(self.conv2(x, ei)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.conv3(x, ei)

    sim_model = TrafficGAT(14, 64, 2).to("cpu")
    sim_model.load_state_dict(torch.load(
        os.path.join(MODELS, "model_gat.pt"),
        map_location="cpu", weights_only=False
    ), strict=False)
    sim_model.eval()

    with torch.no_grad():
        out   = sim_model(X, edge_index)
        probs = torch.softmax(out, dim=1).numpy()[0]

    prob_cong = float(probs[1])
    prob_free = float(probs[0])
    color, _, pill_cls, status = corridor_style(sim_index)
    bar_color = "#22C55E" if prob_cong < 0.3 else "#F97316" if prob_cong < 0.6 else "#EF4444"
    bar_w     = min(int(prob_cong * 100), 100)

    st.markdown(f"""
    <div class="sim-result">
        <div class="sim-result-title">Simulation result</div>
        <div style="display:flex;gap:30px;flex-wrap:wrap;margin-bottom:20px;">
            <div>
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">CORRIDOR</div>
                <div style="font-size:20px;font-weight:700;color:#E2E8F0;">{sim_corridor}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">CONGESTION INDEX</div>
                <div style="font-size:20px;font-weight:700;color:{color};
                     font-family:'DM Mono',monospace;">{sim_index:.1f}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">STATUS</div>
                <span class="status-pill {pill_cls}">{status}</span>
            </div>
            <div>
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">DAY</div>
                <div style="font-size:20px;font-weight:700;color:#E2E8F0;">
                    {sim_day} {'(Weekend)' if is_weekend else '(Weekday)'}
                </div>
            </div>
        </div>

        <div style="font-size:12px;color:#64748B;margin-bottom:8px;">
            GAT PREDICTED CONGESTION PROBABILITY
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
            <div style="flex:1;height:10px;background:rgba(255,255,255,0.06);
                        border-radius:5px;overflow:hidden;">
                <div style="width:{bar_w}%;height:100%;background:{bar_color};
                             border-radius:5px;"></div>
            </div>
            <div style="font-size:22px;font-weight:700;color:{bar_color};
                        font-family:'DM Mono',monospace;min-width:60px;">
                {prob_cong:.1%}
            </div>
        </div>

        <div style="display:flex;gap:20px;">
            <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2);
                        border-radius:8px;padding:12px 16px;flex:1;text-align:center;">
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">FREE FLOW PROBABILITY</div>
                <div style="font-size:20px;font-weight:700;color:#22C55E;
                             font-family:'DM Mono',monospace;">{prob_free:.1%}</div>
            </div>
            <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);
                        border-radius:8px;padding:12px 16px;flex:1;text-align:center;">
                <div style="font-size:11px;color:#64748B;margin-bottom:4px;">CONGESTED PROBABILITY</div>
                <div style="font-size:20px;font-weight:700;color:#EF4444;
                             font-family:'DM Mono',monospace;">{prob_cong:.1%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 20px 0;
            color:#334155;font-size:12px;border-top:1px solid rgba(255,255,255,0.04);
            margin-top:40px;">
    Yaoundé Real-Time Traffic Congestion Prediction System &nbsp;·&nbsp;
    Graph Attention Network &nbsp;·&nbsp;
    Masters Thesis 2026 &nbsp;·&nbsp;
    University of Yaoundé
</div>
""", unsafe_allow_html=True)