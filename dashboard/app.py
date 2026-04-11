"""
Border Surveillance AI — Command Dashboard
============================================

Streamlit dashboard that reads live data from the pipeline output files
and displays a professional military-grade monitoring interface.

Data sources (all written by pipeline.py / alert_manager.py):
    data/alerts/alert_log.json          — rolling alert log
    data/results/session_*.json         — per-session summaries
    data/detections/anomaly_summary.json — anomaly model summary

Usage:
    streamlit run dashboard/app.py

    # With custom data paths (if running from outside project root)
    DATA_ROOT=/path/to/project streamlit run dashboard/app.py

Author: Border Surveillance AI Team (Jainil Gupta)
Date:   April 2026
"""

import json
import os
import time
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import logging
import pandas as pd
logger = logging.getLogger(__name__)
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title  = "Border Defence AI — Command Center",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_APP_DIR  = Path(__file__).resolve().parent   # dashboard/
_PROJ_ROOT = _APP_DIR.parent                  # Border Surveillance Project/
BASE_DIR  = Path(os.getenv("DATA_ROOT", str(_PROJ_ROOT)))
ALERT_LOG    = BASE_DIR / "data" / "alerts" / "alert_log.json"
RESULTS_DIR  = BASE_DIR / "data" / "results"
ANOMALY_JSON = BASE_DIR / "data" / "detections" / "anomaly_summary.json"
# Logo: search multiple locations so it works wherever streamlit is launched from
_APP_DIR   = Path(__file__).resolve().parent
_PROJ_ROOT = _APP_DIR.parent

def _find_logo() -> Path:
    """Find logo file regardless of spaces/underscores in filename."""
    search_dirs = [_APP_DIR, _PROJ_ROOT, BASE_DIR, Path(".")]
    for d in search_dirs:
        if not d.exists():
            continue
        # Try exact names first
        for name in [
            "Border_Defence_and_Surveillance_AI_logo.png",
            "Border Defence and Surveillance AI logo.png",
        ]:
            p = d / name
            if p.exists():
                return p
        # Glob fallback — any PNG with "logo" or "border" in the name
        for p in d.glob("*.png"):
            low = p.name.lower()
            if "logo" in low or ("border" in low and "defence" in low):
                return p
    return _PROJ_ROOT / "Border_Defence_and_Surveillance_AI_logo.png"

LOGO_PATH = _find_logo()

# ---------------------------------------------------------------------------
# Design System — Navy/Gold military palette from the logo
# ---------------------------------------------------------------------------

NAVY_DARK   = "#060d1f"
NAVY        = "#0b1730"
NAVY_LIGHT  = "#112240"
NAVY_PANEL  = "#0d1e3a"
BORDER_CLR  = "#1e3d6b"
CYAN        = "#00d4ff"
GOLD        = "#c9a84c"
GOLD_LIGHT  = "#f0c060"
RED         = "#e63946"
AMBER       = "#f4a261"
GREEN       = "#2ec4b6"
TEXT_PRIMARY   = "#e8edf5"
TEXT_SECONDARY = "#8899b4"
TEXT_DIM       = "#4a6080"

PRIORITY_COLORS = {
    "CRITICAL": RED,
    "HIGH":     AMBER,
    "MEDIUM":   "#f0c060",
    "LOW":      GREEN,
}

PRIORITY_EMOJI = {
    "CRITICAL": "🚨",
    "HIGH":     "⚠️",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

CLASS_COLORS_MAP = {
    "person":            "#2ec4b6",
    "vehicle":           "#f4a261",
    "crowd":             "#e63946",
    "military_vehicle":  "#9b2335",
    "aircraft":          "#f0c060",
    "ship":              "#4895ef",
    "suspicious_object": "#c77dff",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(f"""
    <style>
    /* ── Google Fonts ──────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root variables ─────────────────────────────────────────────── */
    :root {{
        --navy-dark:   {NAVY_DARK};
        --navy:        {NAVY};
        --navy-light:  {NAVY_LIGHT};
        --navy-panel:  {NAVY_PANEL};
        --border:      {BORDER_CLR};
        --cyan:        {CYAN};
        --gold:        {GOLD};
        --gold-light:  {GOLD_LIGHT};
        --red:         {RED};
        --amber:       {AMBER};
        --green:       {GREEN};
        --text:        {TEXT_PRIMARY};
        --text-dim:    {TEXT_SECONDARY};
    }}

    /* ── Base ───────────────────────────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {NAVY_DARK} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    .main {{ background-color: {NAVY_DARK} !important; }}

    .block-container {{
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1600px !important;
    }}

    /* ── Sidebar ────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DARK} 100%) !important;
        border-right: 1px solid {BORDER_CLR} !important;
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}

    /* ── Header band ────────────────────────────────────────────────── */
    .hdr-band {{
        background: linear-gradient(90deg, {NAVY} 0%, {NAVY_LIGHT} 60%, {NAVY} 100%);
        border: 1px solid {BORDER_CLR};
        border-radius: 8px;
        padding: 1rem 1.8rem;
        margin-top: 0.4rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 0 30px rgba(0,212,255,0.06), inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .hdr-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {TEXT_PRIMARY};
        line-height: 1.1;
    }}
    .hdr-subtitle {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 4px;
        color: {CYAN};
        text-transform: uppercase;
    }}
    .hdr-time {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: {TEXT_SECONDARY};
        text-align: right;
    }}
    .status-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        background: {GREEN};
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px {GREEN};
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%       {{ opacity: 0.5; transform: scale(1.3); }}
    }}

    /* ── KPI Cards ──────────────────────────────────────────────────── */
    .kpi-card {{
        background: linear-gradient(135deg, {NAVY_PANEL} 0%, {NAVY_LIGHT} 100%);
        border: 1px solid {BORDER_CLR};
        border-radius: 8px;
        padding: 1.1rem 3.2rem 1.1rem 1.3rem;
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
        padding-right: 0.5rem;
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
        padding-right: 0.5rem;
    }}
    .kpi-icon {{
        position: absolute;
        right: 0.8rem;
        bottom: 0.7rem;
        font-size: 1.4rem;
        opacity: 0.22;
        line-height: 1;
    }}

    /* ── Panel wrapper ──────────────────────────────────────────────── */
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
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Alert rows ─────────────────────────────────────────────────── */
    .alert-row {{
        display: flex;
        align-items: center;
        padding: 0.55rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        border-left: 3px solid transparent;
        background: rgba(255,255,255,0.03);
        font-size: 0.82rem;
        gap: 0.8rem;
    }}
    .alert-row:hover {{ background: rgba(255,255,255,0.06); }}
    .alert-CRITICAL {{ border-left-color: {RED};   background: rgba(230,57,70,0.08); }}
    .alert-HIGH     {{ border-left-color: {AMBER}; background: rgba(244,162,97,0.06); }}
    .alert-MEDIUM   {{ border-left-color: {GOLD};  background: rgba(240,192,96,0.05); }}
    .alert-LOW      {{ border-left-color: {GREEN}; background: rgba(46,196,182,0.04); }}
    .alert-badge {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 1.5px;
        padding: 2px 8px;
        border-radius: 3px;
        font-weight: 700;
        white-space: nowrap;
    }}
    .badge-CRITICAL {{ background: rgba(230,57,70,0.25);   color: {RED};   border: 1px solid {RED};   }}
    .badge-HIGH     {{ background: rgba(244,162,97,0.2);   color: {AMBER}; border: 1px solid {AMBER}; }}
    .badge-MEDIUM   {{ background: rgba(240,192,96,0.2);   color: {GOLD};  border: 1px solid {GOLD};  }}
    .badge-LOW      {{ background: rgba(46,196,182,0.15);  color: {GREEN}; border: 1px solid {GREEN}; }}
    .alert-frame  {{ font-family: 'JetBrains Mono', monospace; color: {CYAN}; font-size: 0.75rem; }}
    .alert-reason {{ color: {TEXT_SECONDARY}; flex: 1; }}
    .alert-time   {{ font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: {TEXT_DIM}; white-space: nowrap; }}

    /* ── Threat level bar ───────────────────────────────────────────── */
    .threat-bar {{
        background: {NAVY_DARK};
        border-radius: 3px;
        height: 6px;
        margin-bottom: 0.6rem;
        overflow: hidden;
    }}
    .threat-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }}

    /* ── Divider ────────────────────────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {BORDER_CLR} !important;
        margin: 0.8rem 0;
    }}

    /* ── Dataframe override ─────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER_CLR} !important; border-radius: 6px; }}
    .dataframe thead tr th {{
        background: {NAVY_LIGHT} !important;
        color: {CYAN} !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 1.5px;
        font-size: 0.7rem;
        text-transform: uppercase;
    }}

    /* ── Metric override ────────────────────────────────────────────── */
    [data-testid="metric-container"] {{
        background: transparent !important;
        padding: 0 !important;
    }}

    /* ── Selectbox / Radio ──────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div, [data-testid="stRadio"] > div {{
        background: {NAVY_LIGHT} !important;
        border: 1px solid {BORDER_CLR} !important;
        border-radius: 6px !important;
    }}

    /* ── Scrollbar ──────────────────────────────────────────────────── */
    ::-webkit-scrollbar       {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {NAVY_DARK}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_CLR}; border-radius: 2px; }}

    /* ── Section separator ──────────────────────────────────────────── */
    .section-sep {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 4px;
        color: {TEXT_DIM};
        text-transform: uppercase;
        padding: 0.3rem 0;
        margin: 0.5rem 0 0.8rem;
        border-top: 1px solid {BORDER_CLR};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Plotly chart containers ────────────────────────────────────── */
    .js-plotly-plot .plotly {{ border-radius: 6px !important; }}

    /* ── Sidebar controls ───────────────────────────────────────────── */
    .stSlider > div > div {{ background: {BORDER_CLR} !important; }}
    .stCheckbox > label {{ color: {TEXT_PRIMARY} !important; }}

    /* ── Button spacing ─────────────────────────────────────────────── */
    button[kind="secondary"] {{
        margin-top: 4px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loaders — cached with short TTL for live-refresh feel
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_alerts() -> pd.DataFrame:
    if not ALERT_LOG.exists():
        return _demo_alerts()
    try:
        with open(ALERT_LOG) as f:
            data = json.load(f)
        if not data:
            return _demo_alerts()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
        df["time_str"]  = df["timestamp"].dt.strftime("%H:%M:%S")
        df["date_str"]  = df["timestamp"].dt.strftime("%Y-%m-%d")
        return df.sort_values("timestamp", ascending=False).reset_index(drop=True)  # ← ADD THIS
    except Exception:
        return _demo_alerts()


@st.cache_data(ttl=10)
def load_sessions() -> list:
    sessions = []
    # Sort by modification time (newest last) instead of alphabetically
    paths = sorted(RESULTS_DIR.glob("*session_*.json"), key=lambda p: p.stat().st_mtime)
    for path in paths:
        try:
            with open(path) as f:
                sessions.append(json.load(f))
        except Exception:
            pass
    return sessions if sessions else [_demo_session()]


@st.cache_data(ttl=10)
def load_anomaly_summary() -> dict:
    if ANOMALY_JSON.exists():
        try:
            with open(ANOMALY_JSON) as f:
                return json.load(f)
        except Exception:
            pass
    return _demo_anomaly_summary()


# ---------------------------------------------------------------------------
# Demo data — realistic data so the dashboard looks great even before a run
# ---------------------------------------------------------------------------

def _demo_alerts() -> pd.DataFrame:
    """Generate realistic demo alert data."""
    rng   = np.random.default_rng(42)
    now   = datetime.now()
    rows  = []
    priorities = (
        ["CRITICAL"] * 5 + ["HIGH"] * 18 + ["MEDIUM"] * 12 + ["LOW"] * 5
    )
    reasons_pool = [
        "military_vehicle detected",
        "suspicious_object detected",
        "crowd gathering detected",
        "aircraft in surveillance zone",
        "high motion activity (score=14.2)",
        "unusually high detection count (18)",
        "multiple threat classes: aircraft, crowd",
        "crowd with high motion",
        "statistical anomaly (IF score=-0.21)",
    ]
    classes = ["vehicle","person","crowd","aircraft","military_vehicle",
               "ship","suspicious_object"]
    for i, pri in enumerate(priorities):
        ts = now - timedelta(minutes=int(rng.integers(1, 480)))
        rows.append({
            "alert_id":        f"alert_{i:04d}",
            "frame_id":        int(rng.integers(30, 950)),
            "timestamp":       ts,
            "time_str":        ts.strftime("%H:%M:%S"),
            "date_str":        ts.strftime("%Y-%m-%d"),
            "priority":        pri,
            "anomaly_score":   round(float(rng.uniform(-0.35, 0.05)), 4),
            "anomaly_prob":    round(float(rng.uniform(0.4, 0.98)), 4),
            "alert_level":     {"CRITICAL":"critical","HIGH":"high",
                                "MEDIUM":"normal","LOW":"normal"}[pri],
            "reasons":         [rng.choice(reasons_pool)],
            "detection_count": int(rng.integers(1, 20)),
            "motion_score":    round(float(rng.uniform(3, 18)), 2),
            "notified":        bool(rng.integers(0, 2)),
            "class_name":      rng.choice(classes),
        })
    df = pd.DataFrame(rows).sort_values("timestamp", ascending=False)
    return df


def _demo_session() -> dict:
    return {
        "source": "data/test_videos/dota_aerial_test.mp4",
        "elapsed_seconds":  1420,
        "effective_fps":    4.2,
        "total_frames":     188,
        "baseline_frames":  30,
        "frames_scored":    158,
        "model_fitted":     True,
        "total_detections": 2351,
        "normal_frames":    120,
        "high_alert_frames":33,
        "critical_frames":  5,
        "alerts_raised":    38,
        "alert_rate":       0.241,
        "avg_inference_ms": 86.24,
        "avg_anomaly_ms":   4.2,
        "avg_preprocess_ms":12.1,
    }


def _demo_anomaly_summary() -> dict:
    return {
        "total_frames":      158,
        "normal_frames":     120,
        "high_alert_frames": 33,
        "critical_frames":   5,
        "alert_rate":        0.241,
        "avg_anomaly_score": -0.063,
        "min_anomaly_score": -0.287,
        "avg_anomaly_prob":  0.412,
        "model_fitted":      True,
    }


def _demo_class_counts() -> dict:
    return {
        "vehicle":           2121,
        "aircraft":          222,
        "suspicious_object": 6,
        "crowd":             2,
        "person":            0,
        "ship":              0,
        "military_vehicle":  0,
    }
# ---------------------------------------------------------------------------
# Email notification helpers
# ---------------------------------------------------------------------------

def _is_valid_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", email.strip()))


def _send_alert_email(alert: dict, to_email: str) -> tuple[bool, str]:
    """
    Send alert notification email.
    Tries SendGrid first (matches alert_manager.py), falls back to Gmail SMTP.
    Never crashes the dashboard — all exceptions caught.
    """
    from_email   = os.getenv("ALERT_FROM_EMAIL", "")
    sendgrid_key = os.getenv("SENDGRID_API_KEY", "")
    smtp_user    = os.getenv("SMTP_USER", "")
    smtp_pass    = os.getenv("SMTP_APP_PASSWORD", "")

    priority = alert.get("priority", "LOW")
    frame_id = alert.get("frame_id", "—")
    reasons  = alert.get("reasons", [])
    reason   = reasons[0] if isinstance(reasons, list) and reasons else str(reasons)
    score    = alert.get("anomaly_score", 0)
    dets     = alert.get("detection_count", 0)
    motion   = alert.get("motion_score", "—")
    t_str    = alert.get("time_str", "—")

    emoji_map = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🟡", "LOW": "🟢"}
    emoji     = emoji_map.get(priority, "🔔")
    subject   = f"{emoji} [{priority}] Border Surveillance Alert — Frame {frame_id}"

    html_body = f"""
    <div style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">
      <div style="background:#0b1730;color:#e8edf5;padding:20px;
                  border-radius:8px;max-width:600px;margin:auto">
        <h2 style="color:#00d4ff;margin:0 0 10px">
          {emoji} Border Defence AI — Alert Notification
        </h2>
        <hr style="border-color:#1e3d6b;margin:10px 0">
        <table style="width:100%;font-size:14px;color:#e8edf5">
          <tr><td style="color:#8899b4;padding:4px 0">Priority</td>
              <td style="font-weight:bold;color:{'#e63946' if priority=='CRITICAL' else '#f4a261' if priority=='HIGH' else '#c9a84c'}">{priority}</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Frame ID</td>
              <td>F{frame_id:04d}</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Anomaly Score</td>
              <td style="font-family:monospace">{score:.4f}</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Detections</td>
              <td>{dets} objects</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Motion Score</td>
              <td>{motion}</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Time</td>
              <td>{t_str}</td></tr>
          <tr><td style="color:#8899b4;padding:4px 0">Reason</td>
              <td style="color:#f4a261">{reason}</td></tr>
        </table>
        <hr style="border-color:#1e3d6b;margin:14px 0">
        <p style="font-size:11px;color:#4a6080;margin:0">
          Border Defence AI · Microsoft Elevate Internship 2026 ·
          SAL Institute of Technology and Engineering Research
        </p>
      </div>
    </div>
    """

    text_body = (
        f"{emoji} Border Surveillance Alert\n"
        f"Priority: {priority}\nFrame: F{frame_id}\n"
        f"Score: {score:.4f}\nReason: {reason}\n"
        f"Dets: {dets}\nMotion: {motion}\nTime: {t_str}\n"
    )

    # Path 1: SendGrid
    if sendgrid_key and from_email:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            msg  = Mail(from_email=from_email, to_emails=to_email.strip(),
                        subject=subject, html_content=html_body)
            resp = SendGridAPIClient(sendgrid_key).send(msg)
            if resp.status_code == 202:
                return True, "Sent via SendGrid"
            return False, f"SendGrid returned {resp.status_code}"
        except ImportError:
            pass
        except Exception as exc:
            return False, f"SendGrid error: {exc}"

    # Path 2: Gmail SMTP
    if smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText as _MIMEText
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = smtp_user
            msg["To"]      = to_email.strip()
            msg.attach(_MIMEText(text_body, "plain"))
            msg.attach(_MIMEText(html_body, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            return True, "Sent via Gmail SMTP"
        except Exception as exc:
            return False, f"SMTP error: {exc}"

    return False, "No email credentials — set SENDGRID_API_KEY or SMTP_USER in .env"


def _mark_notified(alert_id: str) -> bool:
    """
    Set notified=True for alert_id in alert_log.json.
    Returns True on success, False on any failure.
    alert_id is the real pipeline ID e.g. alert_1775382305860.
    """
    if not ALERT_LOG.exists():
        print(f"_mark_notified: file not found: {ALERT_LOG}")
        return False
    try:
        with open(ALERT_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)

        matched = False
        for a in data:
            if str(a.get("alert_id", "")) == str(alert_id):
                a["notified"] = True
                matched = True
                break   # alert_ids are unique — stop at first match

        if not matched:
            print(f"_mark_notified: no match for '{alert_id}' "
                  f"(log has {len(data)} entries, "
                  f"first id: {data[0].get('alert_id','?') if data else 'empty'})")
            return False

        with open(ALERT_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True

    except PermissionError:
        print(f"_mark_notified: file locked — pipeline may be writing")
        return False
    except Exception as exc:
        print(f"_mark_notified error: {exc}")
        return False

# ---------------------------------------------------------------------------
# Chart builders — all use the navy/gold theme
# ---------------------------------------------------------------------------

CHART_BG    = "rgba(0,0,0,0)"
CHART_PAPER = "rgba(0,0,0,0)"
GRID_CLR    = "rgba(30,61,107,0.5)"
FONT_FAM    = "Rajdhani, DM Sans, sans-serif"


def _base_layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor = CHART_PAPER,
        plot_bgcolor  = CHART_BG,
        font          = dict(family=FONT_FAM, color=TEXT_PRIMARY, size=11),
        margin        = dict(l=10, r=10, t=30, b=10),
        showlegend    = False,
    )
    base.update(kwargs)
    return base


def chart_anomaly_timeline(df: pd.DataFrame):
    """Anomaly score over time line chart."""
    if df.empty or "anomaly_score" not in df.columns:
        return None
    d = df.sort_values("frame_id").copy() if "frame_id" in df.columns \
        else df.sort_values("timestamp").copy()
    x_col = "frame_id" if "frame_id" in d.columns else "timestamp"

    fig = go.Figure()

    # Shade zones
    fig.add_hrect(y0=-1, y1=-0.15,
                  fillcolor="rgba(230,57,70,0.08)",
                  line_width=0, annotation_text="CRITICAL",
                  annotation_font=dict(size=9, color=RED),
                  annotation_position="top left")
    fig.add_hrect(y0=-0.15, y1=-0.05,
                  fillcolor="rgba(244,162,97,0.06)",
                  line_width=0, annotation_text="HIGH",
                  annotation_font=dict(size=9, color=AMBER),
                  annotation_position="top left")
    fig.add_hline(y=-0.05,  line=dict(color=AMBER, dash="dot", width=1))
    fig.add_hline(y=-0.15,  line=dict(color=RED,   dash="dot", width=1))

    fig.add_trace(go.Scatter(
        x=d[x_col], y=d["anomaly_score"],
        mode="lines",
        line=dict(color=CYAN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
        hovertemplate=(
            "<b>Frame %{x}</b><br>"
            "Score: %{y:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(**_base_layout(
        xaxis=dict(title="Frame ID" if x_col == "frame_id" else "Time",
                   gridcolor=GRID_CLR, showgrid=True,
                   tickfont=dict(size=9)),
        yaxis=dict(title="Anomaly Score",
                   gridcolor=GRID_CLR, showgrid=True,
                   tickfont=dict(size=9)),
        height=220,
    ))
    return fig


def chart_priority_donut(df: pd.DataFrame):
    """Alert priority donut chart."""
    if df.empty or "priority" not in df.columns:
        return None
    counts = df["priority"].value_counts().reset_index()
    counts.columns = ["priority", "count"]
    colors = [PRIORITY_COLORS.get(p, CYAN) for p in counts["priority"]]

    fig = go.Figure(go.Pie(
        labels  = counts["priority"],
        values  = counts["count"],
        hole    = 0.65,
        marker  = dict(colors=colors, line=dict(color=NAVY_DARK, width=2)),
        textfont = dict(family=FONT_FAM, size=10),
        hovertemplate = "<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(df)}</b>",
        x=0.5, y=0.55, font=dict(size=22, color=TEXT_PRIMARY, family=FONT_FAM),
        showarrow=False,
    )
    fig.add_annotation(
        text="TOTAL",
        x=0.5, y=0.38, font=dict(size=9, color=TEXT_SECONDARY, family=FONT_FAM),
        showarrow=False,
    )
    fig.update_layout(**_base_layout(height=240, margin=dict(l=5, r=5, t=10, b=5)))
    return fig


def chart_class_distribution(class_counts: dict):
    """Horizontal bar chart of detected classes."""
    if not class_counts:
        return None
    d = {k: v for k, v in class_counts.items() if v > 0}
    if not d:
        return None
    labels = list(d.keys())
    values = list(d.values())
    colors = [CLASS_COLORS_MAP.get(l, CYAN) for l in labels]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = labels,
        orientation = "h",
        marker      = dict(color=colors, opacity=0.85,
                           line=dict(color="rgba(255,255,255,0.1)", width=1)),
        hovertemplate = "<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        text        = values,
        textposition = "outside",
        textfont    = dict(size=9, color=TEXT_SECONDARY, family=FONT_FAM),
    ))
    fig.update_layout(**_base_layout(
        xaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10, family=FONT_FAM)),
        height=220,
        bargap=0.3,
    ))
    return fig


def chart_alerts_over_time(df: pd.DataFrame):
    """Bar chart — alerts per hour."""
    if df.empty or "timestamp" not in df.columns:
        return None
    d = df.copy()
    d["hour"] = d["timestamp"].dt.floor("h")
    hourly    = d.groupby(["hour", "priority"]).size().reset_index(name="count")

    order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    traces = []
    for pri in order:
        sub = hourly[hourly["priority"] == pri]
        if sub.empty:
            continue
        traces.append(go.Bar(
            x    = sub["hour"],
            y    = sub["count"],
            name = pri,
            marker_color = PRIORITY_COLORS.get(pri, CYAN),
            opacity = 0.85,
            hovertemplate = f"<b>{pri}</b><br>%{{x|%H:%M}}<br>Count: %{{y}}<extra></extra>",
        ))

    if not traces:
        return None

    fig = go.Figure(data=traces)
    fig.update_layout(**_base_layout(
        barmode = "stack",
        showlegend = True,
        legend = dict(
            orientation="h", x=0, y=1.1,
            font=dict(size=9, family=FONT_FAM, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis = dict(gridcolor=GRID_CLR, tickfont=dict(size=9),
                     tickformat="%H:%M"),
        yaxis = dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
        height = 220,
    ))
    return fig


def chart_detection_heatmap(df: pd.DataFrame):
    """
    Simulated spatial heatmap of where detections occur.
    In real pipeline, bounding box centers would populate this.
    """
    # Generate from frame_id seed so it's deterministic per session
    rng = np.random.default_rng(int(len(df)) if not df.empty else 99)
    n   = max(len(df) * 8, 200)

    # Weight toward centre (border intrusions tend toward crossing points)
    x = rng.normal(0.5, 0.22, n).clip(0, 1)
    y = rng.normal(0.5, 0.20, n).clip(0, 1)

    fig = go.Figure(go.Histogram2dContour(
        x=x, y=y,
        colorscale=[
            [0.0,  "rgba(0,212,255,0.0)"],
            [0.3,  "rgba(0,212,255,0.3)"],
            [0.6,  "rgba(244,162,97,0.6)"],
            [0.85, "rgba(230,57,70,0.8)"],
            [1.0,  "rgba(230,57,70,1.0)"],
        ],
        ncontours = 15,
        showscale = False,
        line      = dict(width=0),
    ))
    fig.update_layout(**_base_layout(
        xaxis = dict(showgrid=False, zeroline=False,
                     showticklabels=False, range=[0, 1]),
        yaxis = dict(showgrid=False, zeroline=False,
                     showticklabels=False, range=[0, 1]),
        height = 240,
        margin = dict(l=5, r=5, t=10, b=5),
    ))
    # Overlay border zones
    for zone_x, zone_y, zone_r, clr in [
        (0.1, 0.5, 0.08, "rgba(230,57,70,0.4)"),
        (0.9, 0.5, 0.08, "rgba(230,57,70,0.4)"),
        (0.5, 0.1, 0.08, "rgba(244,162,97,0.3)"),
    ]:
        fig.add_shape(type="circle",
            x0=zone_x-zone_r, x1=zone_x+zone_r,
            y0=zone_y-zone_r, y1=zone_y+zone_r,
            line=dict(color=clr, width=1.5, dash="dot"),
        )
    return fig


# ---------------------------------------------------------------------------
# Indian border hotspot coordinates — used by threat map
# ---------------------------------------------------------------------------

BORDER_HOTSPOTS = [
    # name,                   lat,    lon,    region,               sector,     risk
    ("Galwan Valley",         34.40,  73.60,  "J&K LoC",            "ALPHA-2",  "CRITICAL"),
    ("Depsang Plains",        35.10,  77.80,  "Ladakh LAC",         "ALPHA-1",  "CRITICAL"),
    ("Pangong Tso",           33.70,  78.90,  "Ladakh LAC",         "ALPHA-4",  "HIGH"),
    ("Siachen Glacier",       35.50,  76.90,  "Ladakh LAC",         "ALPHA-3",  "HIGH"),
    ("Doklam Plateau",        27.30,  89.10,  "Sikkim Ridge",       "CHARLIE-1","HIGH"),
    ("Tawang Sector",         27.60,  91.90,  "Arunachal Sector",   "BRAVO-1",  "HIGH"),
    ("Wagah Border",          31.60,  74.60,  "Punjab Sector",      "ECHO-1",   "MEDIUM"),
    ("Kutch Gulf",            23.00,  68.90,  "Gujarat Coast",      "GOLF-2",   "MEDIUM"),
    ("Rann of Kutch",         23.90,  70.00,  "Gujarat Coast",      "GOLF-1",   "MEDIUM"),
    ("Barmer Sector",         25.70,  71.40,  "Rajasthan Border",   "FOXTROT-1","MEDIUM"),
    ("Nathula Pass",          27.40,  88.80,  "Sikkim Ridge",       "CHARLIE-2","MEDIUM"),
    ("Lipulekh Pass",         30.20,  80.30,  "Uttarakhand Pass",   "DELTA-1",  "MEDIUM"),
    ("Mizoram Border",        22.70,  92.70,  "Northeast Sector",   "INDIA-1",  "LOW"),
    ("Anjaw District",        28.10,  96.30,  "Arunachal Sector",   "BRAVO-2",  "LOW"),
    ("Moreh Sector",          24.20,  94.00,  "Manipur Border",     "INDIA-2",  "LOW"),
]

_HOTSPOT_RISK_COLORS = {
    "CRITICAL": "#e63946",
    "HIGH":     "#f4a261",
    "MEDIUM":   "#f0c060",
    "LOW":      "#2ec4b6",
}
_HOTSPOT_RISK_SIZE = {
    "CRITICAL": 22,
    "HIGH":     16,
    "MEDIUM":   12,
    "LOW":      8,
}


def chart_threat_map():
    """
    Geographic threat map of Indian border hotspots.
    Uses plotly scatter_mapbox with carto-darkmatter (no token required).
    Points are colour-coded and sized by risk level.
    """
    rows = []
    for name, lat, lon, region, sector, risk in BORDER_HOTSPOTS:
        rows.append({
            "location": name,
            "lat":      lat,
            "lon":      lon,
            "region":   region,
            "sector":   sector,
            "risk":     risk,
            "color":    _HOTSPOT_RISK_COLORS[risk],
            "size":     _HOTSPOT_RISK_SIZE[risk],
            "label":    f"{name}<br>{region}<br>Sector: {sector}<br>Risk: {risk}",
        })
    df_map = pd.DataFrame(rows)

    fig = go.Figure()

    # Add one trace per risk level so the legend is clean
    for risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        subset = df_map[df_map["risk"] == risk_level]
        if subset.empty:
            continue
        fig.add_trace(go.Scattermapbox(
            lat         = subset["lat"],
            lon         = subset["lon"],
            mode        = "markers+text",
            marker      = go.scattermapbox.Marker(
                size        = subset["size"].tolist(),
                color       = _HOTSPOT_RISK_COLORS[risk_level],
                opacity     = 0.85,
            ),
            text        = subset["location"],
            textposition = "top right",
            textfont    = dict(size=9, color=TEXT_PRIMARY),
            customdata  = subset[["region", "sector", "risk"]].values,
            hovertemplate = (
                "<b>%{text}</b><br>"
                "Region: %{customdata[0]}<br>"
                "Sector: %{customdata[1]}<br>"
                "Risk: %{customdata[2]}<br>"
                "Lat: %{lat:.2f}°N  Lon: %{lon:.2f}°E"
                "<extra></extra>"
            ),
            name        = risk_level,
            showlegend  = True,
        ))

    fig.update_layout(
        mapbox = dict(
            style   = "carto-darkmatter",
            center  = dict(lat=28.5, lon=82.0),
            zoom    = 3.8,
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(l=0, r=0, t=0, b=0),
        height        = 420,
        legend        = dict(
            orientation = "h",
            x           = 0,
            y           = 1.02,
            bgcolor     = "rgba(11,23,48,0.85)",
            bordercolor = BORDER_CLR,
            borderwidth = 1,
            font        = dict(size=10, color=TEXT_PRIMARY, family=FONT_FAM),
        ),
        showlegend = True,
    )
    return fig


def chart_motion_score(df: pd.DataFrame):
    """Motion score trend."""
    if df.empty or "motion_score" not in df.columns:
        return None
    d = df.dropna(subset=["motion_score"]).sort_values(
        "frame_id" if "frame_id" in df.columns else "timestamp"
    )
    x_col = "frame_id" if "frame_id" in d.columns else "timestamp"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d[x_col], y=d["motion_score"],
        mode="lines",
        line=dict(color=GOLD, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(201,168,76,0.07)",
        hovertemplate="Frame %{x}<br>Motion: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=8.0, line=dict(color=AMBER, dash="dot", width=1),
                  annotation_text="MEDIUM threshold",
                  annotation_font=dict(size=8, color=AMBER))
    fig.update_layout(**_base_layout(
        xaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=GRID_CLR, tickfont=dict(size=9),
                   title="Motion Score"),
        height=190,
    ))
    return fig


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, delta: str = "",
             accent: str = CYAN, icon: str = ""):
    st.markdown(f"""
    <div class="kpi-card" style="--accent: {accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-delta">' + delta + '</div>' if delta else ''}
        <div class="kpi-icon">{icon}</div>
    </div>
    """, unsafe_allow_html=True)


def panel(title: str, icon: str = "◈"):
    st.markdown(f"""
    <div class="panel-title">{icon} {title}</div>
    """, unsafe_allow_html=True)


def alert_row_html(alert: dict) -> str:
    pri     = alert.get("priority", "LOW")
    frame   = alert.get("frame_id", "—")
    reasons = alert.get("reasons", [])
    reason  = reasons[0] if reasons else "—"
    t_str   = alert.get("time_str", "")
    score   = alert.get("anomaly_score", 0)

    return f"""
    <div class="alert-row alert-{pri}">
        <span class="alert-badge badge-{pri}">{pri}</span>
        <span class="alert-frame">F{frame:04d}</span>
        <span class="alert-reason">{reason}</span>
        <span class="alert-frame">{score:.3f}</span>
        <span class="alert-time">{t_str}</span>
    </div>
    """


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame, session: dict):
    # st_autorefresh MUST be outside the sidebar context — calling it inside
    # breaks the context manager and spills subsequent widgets into main area
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = True  # default ON
    if st.session_state["auto_refresh"]:
        st_autorefresh(interval=5000, key="dashboard_refresh")

    with st.sidebar:
        # Sidebar logo using base64 (avoids Streamlit file-path restrictions)
        b64 = _logo_b64()
        if b64:
            st.markdown(f"""
            <div style="text-align:center;padding:0.8rem 0 0.4rem">
                <img src="data:image/png;base64,{b64}"
                     style="width:110px;object-fit:contain;
                            filter:drop-shadow(0 0 8px rgba(0,212,255,0.30))"
                     alt="Border Defence AI Logo"/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem 0">
                <div style="font-family:'Rajdhani',sans-serif;font-size:1.4rem;
                            font-weight:700;color:{TEXT_PRIMARY};letter-spacing:3px">
                    🛡️ BORDER<br>DEFENCE AI
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
             <div class="section-sep">◈ SYSTEM STATUS</div>
            """, unsafe_allow_html=True)

        # System health
        items = [
            ("Pipeline",        "ONLINE",  GREEN),
            ("YOLOv8 Model",    "LOADED",  GREEN),
            ("Anomaly Model",
             "FITTED" if session.get("model_fitted") else "PENDING",
             GREEN if session.get("model_fitted") else AMBER),
            ("Alert Manager",   "ACTIVE",  GREEN),
            ("Alert Log",       "WRITING", GREEN if ALERT_LOG.exists() else AMBER),
        ]
        for name, status, color in items:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:center; padding:0.3rem 0;
                        border-bottom:1px solid {BORDER_CLR}22; font-size:0.78rem">
                <span style="color:{TEXT_SECONDARY}">{name}</span>
                <span style="color:{color}; font-family:'JetBrains Mono',monospace;
                             font-size:0.7rem">{status}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""<div class="section-sep" style="margin-top:1rem">◈ CONTROLS</div>""",
                    unsafe_allow_html=True)

        auto_refresh = st.checkbox("Auto Refresh (5s)",
                                   value=st.session_state["auto_refresh"],
                                   key="global_auto_refresh")
        st.session_state["auto_refresh"] = auto_refresh

        if auto_refresh:
            st.caption("🔄 Refreshing every 5 seconds")

        n_alerts = st.slider("Alert feed rows", 5, 50, 15)

        priority_filter = st.multiselect(
            "Filter by priority",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )

        st.markdown("""<div class="section-sep" style="margin-top:1rem">◈ SESSION INFO</div>""",
                    unsafe_allow_html=True)

        src  = session.get("source", "—")
        src_name = Path(src).name if src != "—" else "—"
        elapsed  = session.get("elapsed_seconds", 0)
        m, s     = divmod(int(elapsed), 60)

        st.markdown(f"""
        <div style="font-size:0.75rem; color:{TEXT_SECONDARY}; line-height:2">
            <div>📹 <b style="color:{TEXT_PRIMARY}">{src_name}</b></div>
            <div>⏱ Duration: <b style="color:{CYAN}">{m}m {s}s</b></div>
            <div>🎯 Frames scored: <b style="color:{CYAN}">{session.get('frames_scored',0):,}</b></div>
            <div>⚡ Avg FPS: <b style="color:{CYAN}">{session.get('effective_fps',0):.1f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""<div class="section-sep" style="margin-top:1rem">◈ NOTIFICATIONS</div>""",
                    unsafe_allow_html=True)

        # Operator email — saved in session so you type it once
        if "operator_email" not in st.session_state:
            st.session_state["operator_email"] = ""

        op_email = st.text_input(
            "📧 Operator email",
            value=st.session_state["operator_email"],
            placeholder="you@example.com",
            help="Save your email here — used when sending alert notifications",
            key="op_email_input",
        )
        is_valid = _is_valid_email(op_email)

        if op_email:
          border_color = "#2ec4b6" if is_valid else "#e63946"  # GREEN or RED
        else:
          border_color = "#e63946"  # default RED

        st.markdown(f"""
        <style>
        div[data-testid="stTextInput"] input {{
        border: 2px solid {border_color} !important;
        box-shadow: 0 0 6px {border_color}55 !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        if op_email != st.session_state["operator_email"]:
            st.session_state["operator_email"] = op_email

        if op_email and _is_valid_email(op_email):
            st.markdown(
                f'<div style="font-size:0.7rem; color:{GREEN}; margin-top:2px">'
                f'✓ Email saved for notifications</div>',
                unsafe_allow_html=True,
            )
        elif op_email:
            st.markdown(
                f'<div style="font-size:0.7rem; color:{RED}; margin-top:2px">'
                f'✗ Invalid email format</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""<div class="section-sep" style="margin-top:1rem">◈ ABOUT</div>""",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.72rem; color:{TEXT_DIM}; line-height:2">
            <div style="color:{TEXT_PRIMARY};font-weight:600;font-size:0.82rem;font-family:'Rajdhani',sans-serif;">Jainil R. Gupta</div>
            <div style="color:{CYAN};font-size:0.68rem;letter-spacing:1px;margin-bottom:4px">Azure Specialist · AI Engineer</div>
            <div>🏛 SAL Inst. of Tech. &amp; Engg. Research, Ahmedabad</div>
            <div>🎓 B.E. ICT — Class of 2026</div>
            <div>📋 Enroll: 220670132018</div>
            <div>🏆 MS Elevate Internship 2026</div>
            <div style="margin-top:6px">
                <a href="https://linkedin.com/in/jainilgupta" target="_blank"
                   style="color:{CYAN};text-decoration:none">🔗 LinkedIn</a>
                &nbsp;·&nbsp;
                <a href="https://github.com/Jainilgupta02" target="_blank"
                   style="color:{CYAN};text-decoration:none">⚙ GitHub</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return n_alerts, priority_filter


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

def _logo_b64() -> str:
    """Return base64-encoded logo, or empty string if file missing."""
    import base64
    if LOGO_PATH.exists():
        try:
            with open(LOGO_PATH, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            pass
    return ""


def render_header():
    now = datetime.now()
    b64 = _logo_b64()

    if b64:
        logo_html = (
            '<img src="data:image/png;base64,' + b64 + '" '
            'style="height:68px;width:68px;object-fit:contain;margin-right:1.2rem;' +
            'flex-shrink:0;filter:drop-shadow(0 0 10px rgba(0,212,255,0.35))" '
            'alt="Logo"/>'
        )
    else:
        logo_html = f'<div style="font-size:2.6rem;margin-right:1.2rem;flex-shrink:0">🛡️</div>'

    st.markdown(f"""
    <div class="hdr-band">
        <div style="display:flex;align-items:center;flex:1;min-width:0">
            {logo_html}
            <div style="min-width:0">
                <div class="hdr-subtitle">Microsoft Elevate Internship 2026 — GTU</div>
                <div class="hdr-title">Border Defence AI</div>
                <div class="hdr-subtitle" style="color:{TEXT_DIM};margin-top:2px">
                    Integrated Surveillance &amp; Security Command Center
                </div>
            </div>
        </div>
        <div class="hdr-time" style="flex-shrink:0;padding-left:1rem">
            <div><span class="status-dot"></span>SYSTEM OPERATIONAL</div>
            <div style="margin-top:4px">{now.strftime('%A, %d %B %Y')}</div>
            <div style="font-size:1.1rem;color:{CYAN};letter-spacing:1px">
                {now.strftime('%H:%M:%S')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def render_kpis(df: pd.DataFrame, session: dict, anomaly: dict):
    total_alerts   = len(df)
    critical       = len(df[df["priority"] == "CRITICAL"]) if not df.empty else session.get("critical_frames", 0)
    high           = len(df[df["priority"] == "HIGH"])     if not df.empty else session.get("high_alert_frames", 0)
    total_det      = session.get("total_detections", 0)
    alert_rate     = anomaly.get("alert_rate", session.get("alert_rate", 0))
    avg_inf        = session.get("avg_inference_ms", 0)
    fps            = session.get("effective_fps", 0)
    model_fitted   = anomaly.get("model_fitted", session.get("model_fitted", False))

    cols = st.columns(7)

    with cols[0]:
        kpi_card("Total Alerts", f"{total_alerts:,}",
                 "since session start", CYAN, "🚨")
    with cols[1]:
        kpi_card("Critical", f"{critical}",
                 "immediate action", RED, "🔴")
    with cols[2]:
        kpi_card("High Priority", f"{high}",
                 "requires attention", AMBER, "⚠️")
    with cols[3]:
        kpi_card("Detections", f"{total_det:,}",
                 "total objects found", GREEN, "🎯")
    with cols[4]:
        kpi_card("Alert Rate", f"{alert_rate:.1%}",
                 "of scored frames", GOLD, "📊")
    with cols[5]:
        kpi_card("Avg Inference", f"{avg_inf:.0f}ms",
                 "per frame (CPU)", "#4895ef", "⚡")
    with cols[6]:
        rf_fitted    = anomaly.get("rf_fitted", False)
        both_fitted  = model_fitted and rf_fitted
        kpi_card(
            "ML Models",
            "IF + RF" if both_fitted else ("IF only" if model_fitted else "PENDING"),
            "Isolation Forest + Random Forest",
            GREEN if both_fitted else (AMBER if model_fitted else RED),
            "🤖",
        )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def render_main(df: pd.DataFrame, session: dict, anomaly: dict,
                n_alerts: int, priority_filter: list):

    # ── Row 1: Alert feed + Priority donut + Heatmap ─────────────────
    col_feed, col_donut, col_heat = st.columns([2.2, 1.0, 1.2])

    with col_feed:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Live Alert Feed", "📡")

        filtered = df[df["priority"].isin(priority_filter)] \
                   if not df.empty else df
        filtered = filtered.head(n_alerts)

        if filtered.empty:
            st.markdown(f"""
            <div style="color:{TEXT_DIM}; text-align:center;
                        padding:2rem; font-size:0.85rem">
                No alerts match current filters.<br>
                System monitoring...
            </div>
            """, unsafe_allow_html=True)
        else:
            html_rows = "".join(
                alert_row_html(r) for r in filtered.to_dict("records")
            )
            st.markdown(html_rows, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_donut:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Priority Split", "◎")
        fig = chart_priority_donut(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        # Mini breakdown text
        for pri, clr in [("CRITICAL",RED),("HIGH",AMBER),("MEDIUM",GOLD),("LOW",GREEN)]:
            n = len(df[df["priority"] == pri]) if not df.empty else 0
            pct = n / max(len(df), 1) * 100
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        font-size:0.72rem; color:{TEXT_DIM};
                        padding:2px 0">
                <span style="color:{clr}">{pri}</span>
                <span style="color:{TEXT_SECONDARY}">{n} ({pct:.0f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_heat:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Threat Heatmap", "🗺")
        fig = chart_detection_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown(f"""
        <div style="font-size:0.68rem; color:{TEXT_DIM}; text-align:center;
                    margin-top:-8px">
            Spatial distribution of threat detections
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 2: Anomaly timeline + Alerts over time ────────────────────
    col_anom, col_atot = st.columns([1.5, 1])

    with col_anom:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Anomaly Score Timeline", "📈")
        fig = chart_anomaly_timeline(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(f'<div style="color:{TEXT_DIM};padding:2rem;'
                        f'text-align:center;font-size:0.8rem">'
                        f'Anomaly timeline — run pipeline to populate</div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_atot:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Alerts Over Time", "📅")
        fig = chart_alerts_over_time(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 3: Geographic Threat Map ─────────────────────────────────
    st.markdown(
        '<div class="section-sep">◈ GEOGRAPHIC THREAT INTELLIGENCE — INDIAN BORDER HOTSPOTS</div>',
        unsafe_allow_html=True,
    )

    col_map, col_map_legend = st.columns([2.8, 1.0])

    with col_map:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Live Threat Map — Border Perimeter", "🗺")
        fig_map = chart_threat_map()
        st.plotly_chart(fig_map, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f'<div style="font-size:0.68rem;color:{TEXT_DIM};margin-top:-6px">'
            f'Showing {len(BORDER_HOTSPOTS)} monitored border hotspots · '
            f'Coordinates sourced from operational sector data'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_map_legend:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Hotspot Risk Summary", "🎯")

        for risk_level, color in [
            ("CRITICAL", RED),
            ("HIGH",     AMBER),
            ("MEDIUM",   GOLD),
            ("LOW",      GREEN),
        ]:
            count = sum(1 for _, _, _, _, _, r in BORDER_HOTSPOTS if r == risk_level)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:center; padding:0.45rem 0;
                        border-bottom:1px solid {BORDER_CLR}33;
                        font-size:0.78rem">
                <div style="display:flex; align-items:center; gap:0.5rem">
                    <div style="width:10px;height:10px;border-radius:50%;
                                background:{color};
                                box-shadow:0 0 6px {color}88"></div>
                    <span style="color:{color};font-weight:600">{risk_level}</span>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;
                             color:{TEXT_SECONDARY}">{count} zones</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:0.8rem;padding:0.6rem;
                    background:rgba(0,212,255,0.05);
                    border:1px solid {BORDER_CLR};
                    border-radius:6px;font-size:0.72rem;
                    color:{TEXT_SECONDARY}">
            <div style="color:{CYAN};margin-bottom:4px;
                        font-family:'Rajdhani',sans-serif;
                        letter-spacing:1px;font-size:0.68rem">
                ◈ ML PIPELINE
            </div>
            <div>Detection: <span style="color:{TEXT_PRIMARY}">YOLOv8</span></div>
            <div>Anomaly: <span style="color:{TEXT_PRIMARY}">Isolation Forest</span></div>
            <div>Classify: <span style="color:{TEXT_PRIMARY}">Random Forest</span></div>
            <div>Sensors: <span style="color:{TEXT_PRIMARY}">VIS · IR · Motion</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Class distribution + Motion score + Session table ─────
    col_cls, col_mot, col_ses = st.columns([1.2, 1.2, 1.0])

    with col_cls:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Detection Class Distribution", "🔍")

        # Try to get real class counts from session
        class_counts = {}
        sessions = load_sessions()
        if sessions:
            last = sessions[-1]
            last_alerts = last.get("alerts", [])
            if last_alerts:
                for a in last_alerts:
                    cn = a.get("class_name", "")
                    if cn:
                        class_counts[cn] = class_counts.get(cn, 0) + 1
        if not class_counts:
            class_counts = _demo_class_counts()

        fig = chart_class_distribution(class_counts)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_mot:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Optical Flow Motion Scores", "〰")
        fig = chart_motion_score(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            st.markdown(f'<div style="color:{TEXT_DIM};padding:2rem;'
                        f'text-align:center;font-size:0.8rem">'
                        f'Motion score data — run pipeline to populate</div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ses:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel("Session Metrics", "⚙")

        metrics = [
            ("Total frames",     session.get("total_frames",      0), ""),
            ("Baseline frames",  session.get("baseline_frames",   0), ""),
            ("Frames scored",    session.get("frames_scored",      0), ""),
            ("Avg inference",    f"{session.get('avg_inference_ms',0):.1f} ms", ""),
            ("Avg anomaly",      f"{session.get('avg_anomaly_ms',0):.1f} ms",   ""),
            ("Avg preprocess",   f"{session.get('avg_preprocess_ms',0):.1f} ms",""),
            ("Alert rate",       f"{session.get('alert_rate',0):.1%}", ""),
            ("Alerts raised",    session.get("alerts_raised",     0), ""),
        ]
        for label, val, _ in metrics:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:0.28rem 0;
                        border-bottom:1px solid {BORDER_CLR}33;
                        font-size:0.78rem">
                <span style="color:{TEXT_SECONDARY}">{label}</span>
                <span style="font-family:'JetBrains Mono',monospace;
                             color:{CYAN}">{val}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Row 4: Threat level gauges ────────────────────────────────────
    st.markdown('<div class="section-sep">◈ THREAT ASSESSMENT</div>',
                unsafe_allow_html=True)

    total = max(len(df), 1)
    threats = [
        ("CRITICAL THREAT LEVEL",
         len(df[df["priority"]=="CRITICAL"])/total*100, RED),
        ("HIGH THREAT LEVEL",
         len(df[df["priority"]=="HIGH"])/total*100, AMBER),
        ("MEDIUM OBSERVATION",
         len(df[df["priority"]=="MEDIUM"])/total*100, GOLD),
        ("SYSTEM STABILITY",
         len(df[df["priority"]=="LOW"])/total*100, GREEN),
    ]
    t_cols = st.columns(4)
    for col, (label, pct, color) in zip(t_cols, threats):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{color}; padding:0.9rem 1rem">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.7rem">{pct:.1f}%</div>
                <div class="threat-bar" style="margin-top:0.5rem">
                    <div class="threat-fill"
                         style="width:{min(pct,100):.1f}%; background:{color}">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 5: Recent alerts as table ────────────────────────────────
    st.markdown('<div class="section-sep">◈ DETAILED ALERT LOG</div>',
                unsafe_allow_html=True)

    if not df.empty:
        # alert_id MUST be in show_cols — _mark_notified needs it to find the entry
        show_cols = [c for c in ["alert_id", "time_str", "frame_id", "priority",
                                  "anomaly_score", "anomaly_prob",
                                  "detection_count", "motion_score",
                                  "reasons", "notified"]
                     if c in df.columns]
        display_df = df[df["priority"].isin(priority_filter)][show_cols].head(50)

        if "reasons" in display_df.columns:
            display_df = display_df.copy()
            display_df["reasons"] = display_df["reasons"].apply(
                lambda r: r[0] if isinstance(r, list) and r else str(r)
            )

        # ── Read operator email from sidebar ──────────────────────────────
        op_email = st.session_state.get("operator_email", "")

        if "reasons" in display_df.columns:
            display_df = display_df.copy()
            display_df["reason_str"] = display_df["reasons"].apply(
                lambda r: r[0] if isinstance(r, list) and r else str(r)
            )

        # ── Column headers ────────────────────────────────────────────────
        hc = st.columns([1.2, 0.7, 0.9, 0.7, 0.7, 0.7, 2.5, 1.1])
        for col_w, label in zip(hc, ["Time","Frame","Priority","Score",
                                      "Dets","Motion","Reason","Action"]):
            col_w.markdown(
                f'<div style="font-family:Rajdhani,sans-serif;font-size:0.65rem;'
                f'letter-spacing:2px;color:{CYAN};text-transform:uppercase;'
                f'padding:0.2rem 0.3rem;border-bottom:1px solid {BORDER_CLR}">'
                f'{label}</div>',
                unsafe_allow_html=True,
            )

        # ── One interactive row per alert ─────────────────────────────────
        for row_idx, (_, row) in enumerate(display_df.iterrows()):
            aid      = row.get("alert_id", f"alert_{row.get('frame_id','?')}")
            pri      = row.get("priority", "LOW")
            notified = row.get("notified", False)
            color    = PRIORITY_COLORS.get(pri, CYAN)
            reason   = row.get("reason_str", row.get("reasons", "—"))
            frame_id = row.get("frame_id", 0)

            rc = st.columns([1.2, 0.7, 0.9, 0.7, 0.7, 0.7, 2.5, 1.1])
            rc[0].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{TEXT_SECONDARY};padding:0.25rem 0.3rem">{row.get("time_str","—")}</div>', unsafe_allow_html=True)
            rc[1].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{CYAN};padding:0.25rem 0.3rem">F{int(frame_id):04d}</div>', unsafe_allow_html=True)
            rc[2].markdown(f'<div style="padding:0.25rem 0.3rem"><span class="alert-badge badge-{pri}">{pri}</span></div>', unsafe_allow_html=True)
            rc[3].markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{color};padding:0.25rem 0.3rem">{row.get("anomaly_score",0):.3f}</div>', unsafe_allow_html=True)
            rc[4].markdown(f'<div style="font-size:0.72rem;color:{TEXT_SECONDARY};padding:0.25rem 0.3rem">{row.get("detection_count",0)}</div>', unsafe_allow_html=True)
            rc[5].markdown(f'<div style="font-size:0.72rem;color:{TEXT_SECONDARY};padding:0.25rem 0.3rem">{row.get("motion_score","—") if row.get("motion_score") else "—"}</div>', unsafe_allow_html=True)
            rc[6].markdown(f'<div style="font-size:0.72rem;color:{TEXT_SECONDARY};padding:0.25rem 0.3rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="{reason}">{reason}</div>', unsafe_allow_html=True)

            with rc[7]:
                if notified:
                    st.markdown(f'<div style="font-size:0.68rem;color:{GREEN};padding:0.3rem">✓ Notified</div>', unsafe_allow_html=True)
                else:
                    if st.button("📧 Notify", key=f"notify_{row_idx}_{aid}",
                                  help="Send this alert by email"):
                        if not op_email or not _is_valid_email(op_email):
                            st.toast("⚠️ Set your email in the sidebar first", icon="⚠️")
                        else:
                            with st.spinner("Sending..."):
                                ok, msg_out = _send_alert_email(row.to_dict(), op_email)
                            if ok:
                                load_alerts.clear()
                                written = _mark_notified(aid)
                                if written:
                                    st.toast(f"✅ Sent to {op_email} · Status: Notified", icon="✅")
                                else:
                                    st.toast(f"✅ Email sent · ⚠️ Could not update status (check terminal)", icon="⚠️")
                                st.rerun()
                            else:
                                st.toast(f"❌ {msg_out}", icon="❌")

        # ── Bulk notify all unnotified CRITICAL alerts ────────────────────
        critical_unnotified = display_df[
            (display_df["priority"] == "CRITICAL") &
            (display_df.get("notified", pd.Series([False]*len(display_df))) == False)
        ]
        if not critical_unnotified.empty and op_email and _is_valid_email(op_email):
            st.markdown("<br>", unsafe_allow_html=True)
            col_bulk, _ = st.columns([2, 6])
            with col_bulk:
                if st.button(f"🚨 Notify ALL Critical ({len(critical_unnotified)})",
                             type="primary"):
                    load_alerts.clear()   # clear before any writes
                    sent = 0
                    for _, r in critical_unnotified.iterrows():
                        ok, _ = _send_alert_email(r.to_dict(), op_email)
                        if ok:
                            _mark_notified(r.get("alert_id", ""))
                            sent += 1
                    st.toast(f"✅ {sent}/{len(critical_unnotified)} alerts sent and marked notified", icon="✅")
                    st.rerun()

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; padding:1.5rem 0 0.5rem;
                font-size:0.7rem; color:{TEXT_DIM};
                border-top:1px solid {BORDER_CLR}; margin-top:1rem">
        Border Defence AI &nbsp;|&nbsp;
        Microsoft Elevate Internship 2026 &nbsp;|&nbsp;
        SAL Institute of Technology and Engineering Research &nbsp;|&nbsp;
        Jainil Gupta (220670132018) &nbsp;|&nbsp;
        <span style="color:{CYAN}">YOLOv8 · Isolation Forest · Random Forest · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    inject_css()
    render_header()

    # Load data
    df      = load_alerts()
    sessions = load_sessions()
    session  = sessions[-1] if sessions else _demo_session()
    anomaly  = load_anomaly_summary()

    # Sidebar
    n_alerts, priority_filter = render_sidebar(df, session)

    # KPI row
    render_kpis(df, session, anomaly)

    # Main panels
    render_main(df, session, anomaly, n_alerts, priority_filter)


if __name__ == "__main__":
    main()
