"""
Enhanced Analysis Dashboard Page
===================================

Streamlit page that visualizes zone intrusion analysis, temporal
patterns, and enhanced pipeline metrics.  Reads data from the
enhanced_analysis.json file produced by pipeline.py.

This is a SEPARATE PAGE — it does not modify dashboard/app.py.
Streamlit's multi-page app system automatically detects files
in the dashboard/pages/ directory and adds them to the sidebar.

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enhanced Analysis — Border Defence AI",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PAGE_DIR  = Path(__file__).resolve().parent          # pages/
_DASH_DIR  = _PAGE_DIR.parent                         # dashboard/
_PROJ_ROOT = _DASH_DIR.parent                         # project root
BASE_DIR   = Path(os.getenv("DATA_ROOT", str(_PROJ_ROOT)))

ENHANCED_LOG  = BASE_DIR / "data" / "results" / "enhanced_analysis.json"
RESULTS_DIR   = BASE_DIR / "data" / "results"

# ---------------------------------------------------------------------------
# Design System (mirrors app.py)
# ---------------------------------------------------------------------------

NAVY_DARK   = "#060d1f"
NAVY        = "#0b1730"
NAVY_LIGHT  = "#112240"
NAVY_PANEL  = "#0d1e3a"
BORDER_CLR  = "#1e3d6b"
CYAN        = "#00d4ff"
GOLD        = "#c9a84c"
RED         = "#e63946"
AMBER       = "#f4a261"
GREEN       = "#2ec4b6"
TEXT_PRIMARY   = "#e8edf5"
TEXT_SECONDARY = "#8899b4"
TEXT_DIM       = "#4a6080"

CHART_BG    = "rgba(0,0,0,0)"
CHART_PAPER = "rgba(0,0,0,0)"
GRID_CLR    = "rgba(30,61,107,0.5)"
FONT_FAM    = "Rajdhani, DM Sans, sans-serif"

ZONE_COLORS = {
    "RESTRICTED":  RED,
    "BUFFER":      AMBER,
    "OBSERVATION": GREEN,
    "SAFE":        TEXT_DIM,
}

TEMPORAL_ALERT_COLORS = {
    "sudden_appearance":    RED,
    "crowd_buildup":        AMBER,
    "loitering":            GOLD,
    "approach_trajectory":  "#c77dff",
    "coordinated_movement": CYAN,
}

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_page_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {NAVY_DARK} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .main {{ background-color: {NAVY_DARK} !important; }}
    .block-container {{
        padding-top: 2rem !important;
        max-width: 1600px !important;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DARK} 100%) !important;
        border-right: 1px solid {BORDER_CLR} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

    .kpi-card {{
        background: linear-gradient(135deg, {NAVY_PANEL} 0%, {NAVY_LIGHT} 100%);
        border: 1px solid {BORDER_CLR};
        border-radius: 8px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        min-height: 100px;
    }}
    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, {CYAN});
    }}
    .kpi-label {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.68rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: {TEXT_SECONDARY};
        margin-bottom: 0.3rem;
    }}
    .kpi-value {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        color: var(--accent, {CYAN});
    }}
    .kpi-delta {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        color: {TEXT_DIM};
        margin-top: 0.3rem;
    }}
    .kpi-icon {{
        position: absolute;
        right: 0.8rem;
        bottom: 0.7rem;
        font-size: 1.4rem;
        opacity: 0.22;
    }}

    .panel {{
        background: linear-gradient(135deg, {NAVY_PANEL} 0%, {NAVY} 100%);
        border: 1px solid {BORDER_CLR};
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .panel-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.72rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {CYAN};
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {BORDER_CLR};
    }}

    .section-sep {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 4px;
        color: {TEXT_DIM};
        text-transform: uppercase;
        padding: 0.3rem 0;
        margin: 0.5rem 0 0.8rem;
        border-top: 1px solid {BORDER_CLR};
    }}

    ::-webkit-scrollbar       {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {NAVY_DARK}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_CLR}; border-radius: 2px; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_enhanced_data() -> Optional[dict]:
    """Load enhanced_analysis.json."""
    if ENHANCED_LOG.exists():
        try:
            with open(ENHANCED_LOG) as f:
                return json.load(f)
        except Exception:
            pass

    # Try to find the most recent enhanced session
    for path in sorted(RESULTS_DIR.glob("enhanced_session_*.json"),
                       reverse=True):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass

    return None


def _demo_enhanced_data() -> dict:
    """Demo data for the enhanced analysis page."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(1, 101):
        frames.append({
            "frame_id": i * 3,
            "detection_count": int(rng.integers(0, 15)),
            "anomaly_score": round(float(rng.uniform(-0.3, 0.1)), 4),
            "alert_level": rng.choice(["normal", "normal", "normal",
                                        "high", "critical"],
                                       p=[0.6, 0.15, 0.1, 0.1, 0.05]),
            "zone_risk": round(float(rng.uniform(0, 0.8)), 4),
            "zone_severity": rng.choice(["SAFE", "OBSERVATION", "BUFFER",
                                          "RESTRICTED"],
                                         p=[0.5, 0.25, 0.15, 0.1]),
            "zone_violations": int(rng.integers(0, 4)),
            "temporal_risk": round(float(rng.uniform(0, 0.6)), 4),
            "temporal_alerts": int(rng.integers(0, 3)),
            "tracked_objects": int(rng.integers(0, 10)),
            "detection_trend": round(float(rng.uniform(-1, 2)), 4),
        })

    return {
        "pipeline_type": "enhanced",
        "total_frames": 100,
        "frames": frames,
        "zone_summary": {
            "total_frames_analyzed": 100,
            "total_violations": 34,
            "violations_by_zone": {
                "border_zone": 8,
                "buffer_zone": 14,
                "observation_zone": 12,
            },
            "zones_defined": 3,
        },
        "temporal_summary": {
            "total_frames_analyzed": 100,
            "total_temporal_alerts": 12,
            "alert_type_counts": {
                "sudden_appearance": 2,
                "crowd_buildup": 3,
                "loitering": 4,
                "approach_trajectory": 2,
                "coordinated_movement": 1,
            },
            "active_tracks": 7,
            "window_size": 30,
        },
    }


# ---------------------------------------------------------------------------
# Helper: KPI card
# ---------------------------------------------------------------------------

def kpi_card(label: str, value, delta: str = "", icon: str = "",
             accent: str = CYAN) -> str:
    return f"""
    <div class="kpi-card" style="--accent: {accent};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {accent};">{value}</div>
        <div class="kpi-delta">{delta}</div>
        <div class="kpi-icon">{icon}</div>
    </div>
    """


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_BG,
        font=dict(family=FONT_FAM, color=TEXT_PRIMARY, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_zone_violations(zone_summary: dict):
    """Zone violation distribution bar chart."""
    violations = zone_summary.get("violations_by_zone", {})
    if not violations:
        return None

    names  = list(violations.keys())
    counts = list(violations.values())
    colors = [
        ZONE_COLORS.get("RESTRICTED") if "border" in n else
        ZONE_COLORS.get("BUFFER") if "buffer" in n else
        ZONE_COLORS.get("OBSERVATION")
        for n in names
    ]

    fig = go.Figure(go.Bar(
        x=names, y=counts,
        marker=dict(color=colors, line=dict(color=NAVY_DARK, width=1)),
        hovertemplate="<b>%{x}</b><br>Violations: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Zone", gridcolor=GRID_CLR, tickfont=dict(size=10)),
        yaxis=dict(title="Violations", gridcolor=GRID_CLR, showgrid=True),
        height=280,
    ))
    return fig


def chart_temporal_alerts(temporal_summary: dict):
    """Temporal alert type distribution donut chart."""
    counts = temporal_summary.get("alert_type_counts", {})
    if not counts or sum(counts.values()) == 0:
        return None

    labels = []
    values = []
    colors = []
    label_map = {
        "sudden_appearance":    "Sudden Appearance",
        "crowd_buildup":        "Crowd Buildup",
        "loitering":            "Loitering",
        "approach_trajectory":  "Approach Trajectory",
        "coordinated_movement": "Coordinated Movement",
    }

    for k, v in counts.items():
        if v > 0:
            labels.append(label_map.get(k, k))
            values.append(v)
            colors.append(TEMPORAL_ALERT_COLORS.get(k, CYAN))

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color=NAVY_DARK, width=2)),
        textfont=dict(family=FONT_FAM, size=10),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b>", x=0.5, y=0.55,
        font=dict(size=22, color=TEXT_PRIMARY, family=FONT_FAM),
        showarrow=False,
    )
    fig.add_annotation(
        text="ALERTS", x=0.5, y=0.4,
        font=dict(size=9, color=TEXT_SECONDARY, family=FONT_FAM),
        showarrow=False,
    )
    fig.update_layout(**_base_layout(height=300))
    return fig


def chart_risk_timeline(df: pd.DataFrame):
    """Zone risk + temporal risk over time."""
    if df.empty:
        return None

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Zone Risk Score", "Temporal Risk Score"),
    )

    if "zone_risk" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["frame_id"], y=df["zone_risk"],
            mode="lines", name="Zone Risk",
            line=dict(color=RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(230,57,70,0.08)",
        ), row=1, col=1)

    if "temporal_risk" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["frame_id"], y=df["temporal_risk"],
            mode="lines", name="Temporal Risk",
            line=dict(color="#c77dff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(199,125,255,0.08)",
        ), row=2, col=1)

    fig.update_layout(**_base_layout(
        height=350,
        showlegend=False,
    ))
    fig.update_xaxes(
        title_text="Frame ID", gridcolor=GRID_CLR,
        tickfont=dict(size=9), row=2, col=1,
    )
    fig.update_yaxes(
        title_text="Risk", gridcolor=GRID_CLR, showgrid=True,
        range=[0, 1], tickfont=dict(size=9),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=10, color=CYAN, family=FONT_FAM))

    return fig


def chart_zone_heatmap(df: pd.DataFrame):
    """Heatmap of zone severity over time."""
    if df.empty or "zone_severity" not in df.columns:
        return None

    severity_map = {"SAFE": 0, "OBSERVATION": 1, "BUFFER": 2, "RESTRICTED": 3}
    df_copy = df.copy()
    df_copy["severity_num"] = df_copy["zone_severity"].map(severity_map).fillna(0)

    # Create bins of 10 frames for a cleaner heatmap
    df_copy["frame_bin"] = (df_copy["frame_id"] // 30) * 30

    fig = go.Figure(go.Scatter(
        x=df_copy["frame_id"],
        y=df_copy["severity_num"],
        mode="markers",
        marker=dict(
            size=8,
            color=df_copy["severity_num"],
            colorscale=[
                [0, TEXT_DIM],
                [0.33, GREEN],
                [0.66, AMBER],
                [1, RED],
            ],
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>Frame %{x}</b><br>"
            "Zone: %{customdata}<extra></extra>"
        ),
        customdata=df_copy["zone_severity"],
    ))

    fig.update_layout(**_base_layout(
        xaxis=dict(title="Frame ID", gridcolor=GRID_CLR, tickfont=dict(size=9)),
        yaxis=dict(
            title="Zone Level",
            gridcolor=GRID_CLR,
            showgrid=True,
            tickfont=dict(size=9),
            tickvals=[0, 1, 2, 3],
            ticktext=["Safe", "Observation", "Buffer", "Restricted"],
        ),
        height=220,
    ))
    return fig


def chart_tracking_overview(df: pd.DataFrame):
    """Active tracked objects over time."""
    if df.empty or "tracked_objects" not in df.columns:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["frame_id"], y=df["tracked_objects"],
        mode="lines+markers",
        line=dict(color=CYAN, width=1.5),
        marker=dict(size=3, color=CYAN),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        hovertemplate="<b>Frame %{x}</b><br>Tracked: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        xaxis=dict(title="Frame ID", gridcolor=GRID_CLR, tickfont=dict(size=9)),
        yaxis=dict(title="Tracked Objects", gridcolor=GRID_CLR,
                   showgrid=True, tickfont=dict(size=9)),
        height=200,
    ))
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def main():
    inject_page_css()

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {NAVY} 0%, {NAVY_LIGHT} 60%, {NAVY} 100%);
                border: 1px solid {BORDER_CLR}; border-radius: 8px;
                padding: 1rem 1.8rem; margin-bottom: 1.2rem;
                display: flex; align-items: center; justify-content: space-between;
                box-shadow: 0 0 30px rgba(0,212,255,0.06);">
        <div>
            <div style="font-family: 'Rajdhani', sans-serif; font-size: 1.6rem;
                        font-weight: 700; letter-spacing: 3px; color: {TEXT_PRIMARY};">
                🗺️ ENHANCED ANALYSIS
            </div>
            <div style="font-family: 'Rajdhani', sans-serif; font-size: 0.75rem;
                        letter-spacing: 4px; color: {CYAN};">
                ZONE INTRUSION · TEMPORAL PATTERNS · OBJECT TRACKING
            </div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
                    color: {TEXT_SECONDARY}; text-align: right;">
            <span style="display: inline-block; width: 8px; height: 8px;
                         background: {GREEN}; border-radius: 50%; margin-right: 6px;
                         box-shadow: 0 0 8px {GREEN};"></span>
            {datetime.now().strftime("%H:%M:%S")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    data = load_enhanced_data()
    if data is None:
        st.info("🔄 No enhanced pipeline data yet. Using demo data for preview.")
        data = _demo_enhanced_data()

    frames_data    = data.get("frames", [])
    zone_summary   = data.get("zone_summary", {})
    temp_summary   = data.get("temporal_summary", {})

    df = pd.DataFrame(frames_data) if frames_data else pd.DataFrame()

    # ── KPI Cards ────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.markdown(kpi_card(
            "Zone Violations",
            zone_summary.get("total_violations", 0),
            delta=f"{zone_summary.get('zones_defined', 3)} zones active",
            icon="🗺️", accent=RED,
        ), unsafe_allow_html=True)

    with k2:
        border_v = zone_summary.get("violations_by_zone", {}).get("border_zone", 0)
        st.markdown(kpi_card(
            "Border Intrusions",
            border_v,
            delta="RESTRICTED zone entries",
            icon="🚨", accent=RED,
        ), unsafe_allow_html=True)

    with k3:
        buffer_v = zone_summary.get("violations_by_zone", {}).get("buffer_zone", 0)
        st.markdown(kpi_card(
            "Buffer Alerts",
            buffer_v,
            delta="Proximity warnings",
            icon="⚠️", accent=AMBER,
        ), unsafe_allow_html=True)

    with k4:
        st.markdown(kpi_card(
            "Temporal Alerts",
            temp_summary.get("total_temporal_alerts", 0),
            delta=f"Window: {temp_summary.get('window_size', 30)} frames",
            icon="  ⏱️ ", accent="#c77dff",
        ), unsafe_allow_html=True)

    with k5:
        st.markdown(kpi_card(
            "Active Tracks",
            temp_summary.get("active_tracks", 0),
            delta="Objects being followed",
            icon="🎯", accent=CYAN,
        ), unsafe_allow_html=True)

    with k6:
        approach = temp_summary.get("alert_type_counts", {}).get(
            "approach_trajectory", 0
        )
        st.markdown(kpi_card(
            "Border Approaches",
            approach,
            delta="Objects moving toward border",
            icon="↗️", accent=GOLD,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main content ─────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Risk timeline
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">📈 RISK SCORE TIMELINE</div>
        </div>
        """, unsafe_allow_html=True)
        fig = chart_risk_timeline(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="risk_timeline")
        else:
            st.info("No risk data available yet")

        # Zone severity heatmap
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🗺️ ZONE SEVERITY OVER TIME</div>
        </div>
        """, unsafe_allow_html=True)
        fig = chart_zone_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="zone_heatmap")
        else:
            st.info("No zone data available yet")

    with col_right:
        # Zone violations chart
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🗺️ ZONE VIOLATION DISTRIBUTION</div>
        </div>
        """, unsafe_allow_html=True)
        fig = chart_zone_violations(zone_summary)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="zone_dist")
        else:
            st.info("No zone violations recorded")

        # Temporal alert types
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">⏱️ TEMPORAL ALERT TYPES</div>
        </div>
        """, unsafe_allow_html=True)
        fig = chart_temporal_alerts(temp_summary)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="temp_alerts")
        else:
            st.info("No temporal alerts recorded")

    # ── Object tracking overview ──────────────────────────────────────
    st.markdown(f"""
    <div class="section-sep">🎯 OBJECT TRACKING OVERVIEW</div>
    """, unsafe_allow_html=True)

    col_track, col_trend = st.columns(2)

    with col_track:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🎯 ACTIVE TRACKED OBJECTS</div>
        </div>
        """, unsafe_allow_html=True)
        fig = chart_tracking_overview(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="tracking")
        else:
            st.info("No tracking data available")

    with col_trend:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">📊 DETECTION TREND</div>
        </div>
        """, unsafe_allow_html=True)
        if not df.empty and "detection_trend" in df.columns:
            fig = go.Figure(go.Scatter(
                x=df["frame_id"], y=df["detection_trend"],
                mode="lines",
                line=dict(color=GOLD, width=1.5),
                fill="tozeroy",
                fillcolor="rgba(201,168,76,0.08)",
                hovertemplate="<b>Frame %{x}</b><br>Trend: %{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0, line=dict(color=TEXT_DIM, dash="dot", width=1))
            fig.update_layout(**_base_layout(
                xaxis=dict(title="Frame ID", gridcolor=GRID_CLR,
                           tickfont=dict(size=9)),
                yaxis=dict(title="Detection Trend (slope)",
                           gridcolor=GRID_CLR, showgrid=True,
                           tickfont=dict(size=9)),
                height=200,
            ))
            st.plotly_chart(fig, use_container_width=True, key="trend")
        else:
            st.info("No trend data available")

    # ── Temporal alert details (expandable) ───────────────────────────
    st.markdown(f"""
    <div class="section-sep">📋 TEMPORAL ALERT BREAKDOWN</div>
    """, unsafe_allow_html=True)

    alert_counts = temp_summary.get("alert_type_counts", {})
    if alert_counts:
        cols = st.columns(5)
        labels = {
            "sudden_appearance":    ("⚡ Sudden Appearance", RED),
            "crowd_buildup":        ("👥 Crowd Buildup", AMBER),
            "loitering":            ("🚶 Loitering", GOLD),
            "approach_trajectory":  ("↗️ Approach Trajectory", "#c77dff"),
            "coordinated_movement": ("🔗 Coordinated Movement", CYAN),
        }
        for i, (key, (label, color)) in enumerate(labels.items()):
            with cols[i]:
                count = alert_counts.get(key, 0)
                st.markdown(kpi_card(
                    label.split(" ", 1)[1] if " " in label else label,
                    count,
                    delta=label.split(" ")[0],
                    icon="", accent=color,
                ), unsafe_allow_html=True)
    else:
        st.info("No temporal alerts have been generated yet. "
                "Run the enhanced pipeline to populate this data.")

    # ── Zone configuration display ────────────────────────────────────
    with st.expander("🔧 Zone Configuration", expanded=False):
        st.markdown(f"""
        | Zone | Level | Y Range | Color |
        |------|-------|---------|-------|
        | Border Zone | 🔴 RESTRICTED | 0.00 – 0.25 | {RED} |
        | Buffer Zone | 🟠 BUFFER | 0.25 – 0.45 | {AMBER} |
        | Observation Zone | 🟢 OBSERVATION | 0.45 – 1.00 | {GREEN} |

        *Zones use normalised coordinates (0–1) matching detector output.*
        *Night boost (21:00–05:00): zone risk multiplied by 1.3×*
        """)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
else:
    main()
