<div align="center">

<!-- TOP WAVE BANNER WITH TITLE BAKED IN -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0e1a,30:0d1b3e,65:1a3a6b,100:0d47a1&height=240&section=header&text=AI-Powered%20Border%20Surveillance%20System&fontSize=38&fontColor=00D9FF&fontAlignY=42&fontAlign=50&desc=Real-Time%20Anomaly%20Detection%20%7C%20YOLOv8%20%7C%20Azure%20%7C%20Streamlit&descSize=16&descColor=CADCFC&descAlignY=62&descAlign=50" width="100%"/>

<!-- PROJECT LOGO -->
<br/>
<img src="dashboard/Border Defence and Surveillance AI logo.png" width="155" alt="Border Defence Project Logo"/>

<br/><br/>

<!-- ANIMATED TYPING LINES -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=28&duration=2500&pause=800&color=00D9FF&center=true&vCenter=true&multiline=false&width=860&height=55&lines=Border+Surveillance+AI;Integrated+Surveillance+%26+Security;Real-Time+Threat+Detection;AI-Powered+Border+Monitoring" alt="Animated Title"/>

<br/><br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Microsoft%20Azure-Cloud-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GTU-Internship%202026-00D9FF?style=flat-square"/>
  <img src="https://img.shields.io/badge/Microsoft%20Elevate-Program-0078D4?style=flat-square&logo=microsoft"/>
  <img src="https://img.shields.io/badge/AZ--900-Certified%20820%2F1000-0078D4?style=flat-square&logo=microsoftazure"/>
  <img src="https://img.shields.io/badge/SAL%20Institute-ICT%20Dept-4CAF50?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Tests-pytest-yellow?style=flat-square&logo=pytest"/>
</p>

<br/>

> **🛡️ An end-to-end AI surveillance pipeline** that processes surveillance footage, detects threats using YOLOv8, scores behavioural anomalies, prioritises operational alerts, and surfaces everything in a Streamlit command dashboard — with optional Azure cloud integration throughout.

<br/>

```
  Custom YOLOv8 Model (v2 Balanced):  7 classes · mAP50 47.8% · mAP50-95 28.5% · 114.7ms/frame CPU inference
```

</div>

---

## 📌 Table of Contents

<details>
<summary><b>Click to expand full contents</b></summary>

- [🎯 Overview](#-overview)
- [🚨 Problem Statement](#-problem-statement)
- [🧠 Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 How It Works](#-how-it-works)
- [⚙️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [▶️ Quick Start — Evaluator Guide](#️-quick-start--evaluator-guide)
- [☁️ Azure Integration](#️-azure-integration)
- [📧 Alert System](#-alert-system)
- [📊 Dashboard](#-dashboard)
- [🧩 Key Modules](#-key-modules)
- [🧪 Sample Output & Results](#-sample-output--results)
- [🧾 Testing](#-testing)
- [📈 Future Improvements](#-future-improvements)
- [🎓 Academic Context](#-academic-context)
- [👨‍💻 Author](#-author)
- [📜 License](#-license)

</details>

---

## 🎯 Overview

Border surveillance environments generate large volumes of visual data that are difficult to monitor continuously by hand. This project automates that entire workflow end-to-end.

```
  Traditional Monitoring                  Border Surveillance AI
  ──────────────────────                  ──────────────────────
  👁️  1 operator, many cameras             🤖  AI processes all feeds 24/7
  ⏳  Slow, delayed human reaction          ⚡  Sub-second anomaly detection
  ❌  Alert fatigue from false alarms       ✅  Priority-filtered notifications
  📁  Isolated logs, manual reports         ☁️  Centralised Azure cloud storage
  📉  No trend or pattern insights          📊  Streamlit operational dashboard
```

The system accepts a video file or live camera feed, extracts and analyses every frame, assigns anomaly scores, generates prioritised alerts, logs everything locally and optionally to Azure, and makes it all readable through an auto-refreshing dashboard.

---

## 🚨 Problem Statement

> *Domain: Border Defence and Surveillance — GTU Internship 2026*

| Challenge | Real-World Impact |
|-----------|-------------------|
| **Large-scale monitoring** | Vast border areas exceed human monitoring capacity |
| **Delayed threat detection** | Manual analysis causes late identification of intrusions |
| **High false-alarm rates** | Animals, weather, and noise trigger unnecessary responses |
| **Resource constraints** | Limited manpower must cover extensive remote regions |
| **Siloed data** | Sensor, camera, and historical data never integrated |

**This system addresses all five** through automated AI detection, cloud integration, and confidence-filtered smart alerting.

---

## 🧠 Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🔍 Object Detection
- Custom-trained YOLOv8n model on balanced 7-class border surveillance dataset (DOTA + xView + Visdrone)
- Detects: `person`, `vehicle`, `crowd`, `military_vehicle`, `aircraft`, `ship`, `suspicious_object`
- Structured per-detection output with class, confidence, bounding box, and threat tags
- `has_high` and `has_critical` flags per frame for downstream prioritisation

### 🧠 Anomaly Detection (Dual ML Pipeline)
- **Isolation Forest** — unsupervised anomaly scoring on 10-dimensional behavioural features
- **Random Forest Classifier** — supervised threat classification (CRITICAL/HIGH/MEDIUM/LOW) trained from anomaly labels
- Baseline learned from first 30 frames (configurable)
- Features: detection count, class diversity, confidence stats, motion score, object location, spatial distribution, suspicious class presence
- Frame-level severity: `normal` → `high` → `critical`

### 🗺️ Zone-Based Intrusion Detection
- Three configurable surveillance zones: 🔴 Border (RESTRICTED) · 🟠 Buffer (HIGH) · 🟢 Observation (MEDIUM)
- Ray-casting point-in-polygon spatial checks on every detection
- High-threat class boosting (person/military_vehicle in restricted zone → instant CRITICAL)
- Night activity boost (21:00–05:00 → 1.3× risk multiplier)
- Composite zone risk scoring (0–1) with crowding penalty

</td>
<td width="50%" valign="top">

### ⏱️ Temporal Video Intelligence
- Multi-frame analysis across a sliding window of consecutive frames
- IoU-based lightweight object tracker (no GPU dependency)
- **5 temporal detectors**: sudden appearance, crowd buildup, loitering, approach trajectory, coordinated movement
- Trajectory analysis: objects moving toward the border flagged as CRITICAL
- Speed estimation and object persistence scoring

### 🚨 Smart Alert System
- Four priority levels: 🔴 `CRITICAL` / 🟠 `HIGH` / 🟡 `MEDIUM` / 🟢 `LOW`
- Rolling JSON alert log written locally on every run
- Cooldown logic to suppress duplicate notifications
- Zone + temporal + anomaly reasons merged into unified alert explanations
- Email notifications via SendGrid for HIGH and CRITICAL alerts
- Azure Cosmos DB persistence when credentials are configured

### 📊 Operational Dashboard
- Streamlit command-centre interface with multi-page layout
- 🗺️ **Enhanced Analysis page** — zone violations, temporal alerts, risk timelines, object tracking
- 🌍 **Threat Heatmap** — geographic threat map of Indian border zones using Plotly Mapbox
- Auto-refreshing view of alerts, sessions, and trends
- Manual email notification trigger from the UI
- Falls back to demo data when no live pipeline output is present

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                     │
│   📹 Video File  OR  🎥 Live Camera Index  →  OpenCV frame reader       │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                                     │
│   Resize (640×640) │ Normalize │ Optional Optical Flow (motion score)    │
│   → structured frame_item dicts passed downstream                        │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     OBJECT DETECTION                  [src/detector.py]  │
│                                                                          │
│   ┌────────────────────────────────────────────────────────────────────┐ │
│   │   Custom YOLOv8n (border_v2_balanced) — 30 epochs, 7 classes       │ │
│   │   Detects → person │ vehicle │ crowd │ military_vehicle │ aircraft │ │
│   │             ship │ suspicious_object                               │ │
│   │   Output → class · confidence · bbox · threat_tag · flags          │ │
│   └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
┌─────────────────────┐ ┌──────────────────────┐ ┌────────────────────────┐
│  🗺️ ZONE ANALYSIS   │ │ ⏱️ TEMPORAL ANALYSIS  │ │ 🧠 ANOMALY DETECTION │
│ [zone_analyzer.py]  │ │[temporal_analyzer.py] │ │ [anomaly.py]          │
│                     │ │                      │ │                        │
│ • 3 border zones    │ │ • IoU object tracker │ │ • Isolation Forest     │
│ • Intrusion detect  │ │ • Approach trajectory│ │ • Random Forest        │
│ • Night boost       │ │ • Crowd buildup      │ │ • Rule-based fallback  │
│ • Risk scoring      │ │ • Loitering detect   │ │ • 10-dim features      │
│                     │ │ • Coordinated move   │ │                        │
└────────┬────────────┘ └──────────┬───────────┘ └───────────┬────────────┘
         └──────────────┬──────────┘                         │
                        ↓                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                    ALERT MANAGEMENT              [src/alert_manager.py]  │
│                                                                          │
│   Zone + Temporal + Anomaly → 🔴 CRITICAL │ 🟠 HIGH │ 🟡 MED │ 🟢 LOW  │
│   Rolling JSON log  │  Cooldown dedup  │  SendGrid email notifications   │
│   Azure Cosmos DB write (if configured)                                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│               OUTPUT LAYER + AZURE + DASHBOARD                           │
│                                                                          │
│  📄 data/alerts/alert_log.json       🗄️  Azure Cosmos DB (alerts)       │
│  📄 data/results/session_*.json      📦  Azure Blob Storage (sessions)  │
│  🖼️  data/detections/frame_*.jpg     📄  enhanced_analysis.json         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │             📊  Streamlit Dashboard  (dashboard/app.py)         │    │
│  │  Alert feed │ Priority chart │ Session summaries │ Trend lines   │    │
│  │  Anomaly overview │ Manual notify │ Auto-refresh │ Threat Map    │    │
│  │  🗺️ Enhanced Analysis: Zone map │ Temporal alerts │ Tracking    │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 How It Works

### 1️⃣ Input and Preprocessing
The pipeline accepts either a video file path or a live camera index. Frames are loaded through OpenCV, resized to 640×640, and optionally passed through an optical flow computation that estimates per-frame motion intensity. Each frame becomes a structured `frame_item` dictionary passed to downstream modules.

### 2️⃣ Object Detection
`BorderDetector` loads the YOLOv8 model once and reuses it across all frames. Each frame produces a list of structured detections with class name, confidence score, bounding box coordinates, normalised spatial features, per-class threat tags, and `has_high` / `has_critical` flags for fast downstream filtering.

### 3️⃣ Anomaly Analysis
`AnomalyDetector` operates in two phases. The first 30 frames (default) build a normal baseline. Every subsequent frame is scored using Isolation Forest against these behavioural features:

| Feature | Description |
|---------|-------------|
| `detection_count` | Total objects detected in the frame |
| `class_diversity` | Number of unique detected classes |
| `confidence_stats` | Mean and max detection confidence |
| `critical_class_count` | Count of high-threat class detections |
| `location_distribution` | Spatial spread of detected objects |
| `object_size` | Average bounding box area |
| `motion_score` | Optical-flow estimated motion intensity |
| `suspicious_presence` | Binary flag for suspicious class in frame |

Frames are classified as `normal`, `high`, or `critical` with human-readable anomaly reasons attached. A **Random Forest Classifier** trained from anomaly labels provides supervised threat classification alongside the unsupervised Isolation Forest.

### 3️⃣.a Zone-Based Intrusion Detection *(Enhanced Pipeline)*
`ZoneAnalyzer` divides the camera field of view into three configurable zones — Border (RESTRICTED), Buffer (HIGH), and Observation (MEDIUM). Every detection's normalised centre coordinate is tested against zone polygons using ray-casting. Detections of high-threat classes (person, military_vehicle, suspicious_object) in the RESTRICTED zone trigger immediate CRITICAL alerts. A night activity boost multiplies zone risk during 21:00–05:00 hours.

### 3️⃣.b Temporal Video Intelligence *(Enhanced Pipeline)*
`TemporalAnalyzer` maintains a sliding window of the last 30 frames and analyses detection patterns across time. It tracks objects across frames using lightweight IoU matching and runs five temporal detectors:

| Detector | What It Catches | Severity |
|----------|----------------|----------|
| ⚡ Sudden Appearance | Empty scene → 5+ new detections | HIGH |
| 👥 Crowd Buildup | Gradual increase in person count | HIGH |
| 🚶 Loitering | Object stationary in one spot for 10+ frames | MEDIUM |
| ↗️ Approach Trajectory | Objects moving toward the border zone | CRITICAL |
| 🔗 Coordinated Movement | 3+ objects moving in the same direction | HIGH |

### 4️⃣ Alert Generation
`AlertManager` merges anomaly scores, zone violations, and temporal alerts into unified priority levels. Zone intrusions and approach trajectories can upgrade alert severity beyond what anomaly scoring alone would assign. HIGH and CRITICAL alerts trigger email notifications subject to a configurable cooldown window. Every non-normal alert is appended to the rolling JSON log. Alert records are optionally written to Azure Cosmos DB.

### 5️⃣ Storage and Monitoring
Session summaries are saved as timestamped JSON files. The enhanced pipeline additionally writes `enhanced_analysis.json` with per-frame zone risk, temporal risk, and object tracking data for the dashboard. When Azure credentials are present, session results are uploaded to Blob Storage and alerts are written to Cosmos DB. The Streamlit dashboard reads local output files directly and auto-refreshes to show the latest operational state.

---

## ⚙️ Tech Stack

<div align="center">

| Layer | Technology | Role |
|-------|-----------|------|
| **Language** | Python 3.9+ | Core runtime |
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time frame inference |
| **Deep Learning** | PyTorch 2.x | Model backend |
| **Computer Vision** | OpenCV 4.x | Video I/O, frame processing, optical flow |
| **Anomaly Detection** | scikit-learn (Isolation Forest + Random Forest) | Unsupervised anomaly scoring + supervised threat classification |
| **Zone Analysis** | NumPy (ray-casting PIP) | Spatial intrusion detection across border zones |
| **Temporal Analysis** | NumPy + IoU tracker | Multi-frame pattern detection and object tracking |
| **Data** | NumPy, Pandas | Feature arrays and session analytics |
| **Dashboard** | Streamlit + Plotly | Operational monitoring UI |
| **Alerting** | SendGrid | Email notifications |
| **Cloud Storage** | Azure Blob Storage SDK | Session result uploads |
| **Cloud Database** | Azure Cosmos DB SDK | Alert persistence |
| **Testing** | pytest + pytest-cov | Automated validation |
| **Environment** | python-dotenv | Credential management |

</div>

---

## 📂 Project Structure

```
Border Surveillance Project/
│
├── 📁 src/                           # Core application modules
│   ├── pipeline.py                   # Main pipeline with zone + temporal intelligence
│   ├── detector.py                   # YOLOv8 wrapper + structured detections
│   ├── anomaly.py                    # Dual ML: Isolation Forest + Random Forest scoring
│   ├── zone_analyzer.py              # Zone-based intrusion detection (3 border zones)
│   ├── temporal_analyzer.py          # Multi-frame temporal video analysis + IoU tracker
│   ├── alert_manager.py              # Priority assignment, logging, email, cooldown
│   └── azure_client.py              # Blob Storage + Cosmos DB integration
│
├── 📁 dashboard/
│   ├── app.py                        # Streamlit command-centre dashboard (main page)
│   ├── pages/
│   │   └── 1_🗺️_Enhanced_Analysis.py # Zone + temporal analysis dashboard page
│   └── Border Defence AI logo.png    # Project branding asset
│
├── 📁 scripts/                       # Utility and dataset preparation scripts
│   ├── pilot.py                      # Manual integration checker across all modules
│   ├── smoke_test.py                 # Quick pipeline smoke test
│   ├── preprocess_all_datasets.py    # Full dataset preprocessing
│   ├── preprocess_balanced_v2.py     # Local dataset preprocessing
│   ├── xview_geojson_to_yolo.py      # xView GeoJSON → YOLO labels
│   └── smart_extract.py              # Intelligent frame extractor
│
├── 📁 data/                          # Runtime data (gitignored — not in repo)
│   ├── alerts/                       # alert_log.json — rolling alert output
│   ├── results/                      # session_*.json + enhanced_analysis.json
│   ├── test_videos/                  # Sample videos for local runs
│   ├── processed/                    # Preprocessed training-ready data (7 classes)
│   ├── annotations/                  # Dataset annotation files
│   ├── raw/                          # Source datasets (DOTA, xView, Visdrone)
│   └── logs/                         # Pipeline runtime logs
│
├── 📁 tests/                         # Automated test suite (285 tests)
│   ├── test_detector.py
│   ├── test_anomaly_and_alert.py
│   ├── test_pipeline.py
│   ├── test_zone_analyzer.py         # 23 tests — zone intrusion detection
│   └── test_temporal_analyzer.py     # 28 tests — temporal pattern detection
│
├── 📁 models/                        # YOLO weights + anomaly model artefacts
├── 📁 notebooks/                     # ppt and Report
├── 📁 docs/                          # Architecture diagrams + presentations
├── 📁 overview/                      # Implementation guide and references
│
├── 📁 models/                        # Trained model artefacts (committed to repo)
│   ├── border_yolo.pt                # Custom-trained YOLOv8 border detection model (~6 MB)
│   └── anomaly_model.pkl            # Trained Isolation Forest anomaly detector (751 KB)
│
├── yolov8n.pt                        # YOLOv8 nano base weights (fallback)
├── yolov8s.pt                        # YOLOv8 small base weights (fallback)
├── requirements.txt                  # All Python dependencies
├── pyproject.toml                    # Project metadata
├── pytest.ini                        # Test runner configuration
├── makefile                          # Common task shortcuts
├── .env                              # Local credentials (never committed)
├── .gitignore                        # Excludes data/, venv/, .env
└── README.md                         # This file
```

---

## ▶️ Quick Start — Evaluator Guide

> ⏱️ **Estimated setup time: ~5 minutes.** Follow these six steps in order and the system will run end-to-end from video input to a live dashboard.

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/jainilgupta02/Border-Surveillance-Project.git
cd "Border-Surveillance-Project"
```

---

### Step 2 — Create and Activate a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat
```

---

### Step 3 — Install All Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 **Dependencies installed.** Now verify your trained models are present before running.

---

### Step 3.5 — Verify Trained Models Are Present

After cloning, confirm the `models/` folder contains both trained artefacts:

```
models/
├── border_yolo.pt       ← Custom-trained YOLOv8 border detection model (~22 MB)
└── anomaly_model.pkl    ← Trained Isolation Forest anomaly detector (99 KB)
```

> ✅ Both files are committed directly to the repository — **no separate download needed.**
> The pipeline loads `models/border_yolo.pt` for detection and `models/anomaly_model.pkl` for anomaly scoring automatically.
> If `border_yolo.pt` is missing, the pipeline falls back gracefully to `yolov8n.pt` (base YOLO weights in repo root).

---

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root. **Azure and SendGrid fields are optional** — leave them blank to run the system in fully local mode.

```env
# ── Azure Storage ──────────────────────────────────
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_CONTAINER_ALERTS=alert-frames
AZURE_STORAGE_CONTAINER_RESULTS=session-results

# ── Azure Cosmos DB ────────────────────────────────
AZURE_COSMOS_ENDPOINT=
AZURE_COSMOS_KEY=
AZURE_COSMOS_DATABASE=SurveillanceDB
AZURE_COSMOS_CONTAINER=Alerts

# ── Email Alerts (SendGrid) ────────────────────────
SENDGRID_API_KEY=
ALERT_FROM_EMAIL=
ALERT_TO_EMAIL=

# ── Email Alerts (SMTP fallback) ───────────────────
SMTP_USER=
SMTP_APP_PASSWORD=

# ── Dashboard (only if running outside project root)
DATA_ROOT=
```

> ✅ **The full pipeline — detection, anomaly scoring, alerts, dashboard — works entirely without cloud credentials.** All outputs are saved locally.

---

### Step 5 — Run the AI Pipeline

**Recommended evaluation command:**

```bash
python src/pipeline.py --video data/test_videos/dota_aerial_test.mp4 --save-frames
```

**With additional controls:**

```bash
python src/pipeline.py \
  --video data/test_videos/dota_aerial_test.mp4 \
  --frame-skip 3 \
  --save-frames \
  --temporal-window 30 \
  --results-dir data/results \
  --annotated-dir data/detections
```

**Live camera mode:**

```bash
python src/pipeline.py --camera 0
```

**Expected terminal output (enhanced pipeline):**

```
✅ Preprocessing complete   — frames extracted and resized to 640×640
✅ Detection complete       — structured detections logged per frame
✅ Zone analysis complete   — intrusion detection across 3 border zones
✅ Temporal analysis done   — object tracking + 5 temporal detectors
✅ Anomaly scoring complete — Isolation Forest + Random Forest scored all frames
✅ Alerts generated         — zone + temporal + anomaly priority levels assigned
✅ Session saved            — data/results/enhanced_session_<source>_<timestamp>.json
✅ Alert log written        — data/alerts/alert_log.json
✅ Enhanced log written     — data/results/enhanced_analysis.json
```

---

### Step 6 — Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at **`http://localhost:8501`** to see the live operational command-centre view.

---

### Step 7 — Run the Test Suite *(optional but recommended)*

```bash
# All tests with verbose output
pytest tests -v

# With HTML coverage report
pytest tests --cov=src --cov-report=html
# Report opens at: htmlcov/index.html
```

---

### 🔍 Other Useful Entry Points

| Command | Purpose |
|---------|---------|
| `python src/pipeline.py --video <path>` | Full pipeline with zone + temporal intelligence |
| `python scripts/main.py` | Minimal single-video detector demo |
| `python scripts/pilot.py data/test_videos/dota_aerial_test.mp4` | Manual integration check across all modules |
| `python scripts/smoke_test.py` | Fast pipeline smoke test |
| `python scripts/generate_test_video.py` | Generate a synthetic test video if none is present |

---

## ☁️ Azure Integration

When credentials are present in `.env`, two cloud paths activate automatically — no code changes required.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Azure Services Used                                                 │
│                                                                      │
│  📦 Blob Storage   → session result JSON files uploaded per run     │
│                      Container: session-results                      │
│                                                                      │
│  🗄️  Cosmos DB      → individual alert documents written per event  │
│                      Database: SurveillanceDB                        │
│                      Container: Alerts                               │
└──────────────────────────────────────────────────────────────────────┘
```

**If Azure credentials are missing or invalid, the system falls back silently to local-only mode.** No errors are raised and the pipeline continues normally.

Data stored in Azure per run:
- Session result summary (`session_<source>_<timestamp>.json`) → Blob Storage
- Per-alert records with priority, anomaly score, detection count, motion score, and reasons → Cosmos DB

---

## 📧 Alert System

The alerting layer is designed for operational triage, not raw event dumping.

| Priority | Trigger Condition | Notification Behaviour |
|----------|-------------------|------------------------|
| 🔴 **CRITICAL** | Highest-severity anomaly + critical class detection | Email sent immediately |
| 🟠 **HIGH** | Significant anomaly score or critical class flag | Email sent (cooldown applies) |
| 🟡 **MEDIUM** | Lower anomaly score or motion-based escalation | Written to log only |
| 🟢 **LOW** | Normal or near-normal activity | Written to log only |

- All non-normal alerts are appended to `data/alerts/alert_log.json`
- HIGH and CRITICAL alerts trigger SendGrid email when `SENDGRID_API_KEY` is configured
- A cooldown window suppresses repeated notifications for the same ongoing threat pattern
- Manual notification can be triggered at any time directly from the Streamlit dashboard

---

## 📊 Dashboard

The dashboard uses Streamlit's multi-page layout. Run `streamlit run dashboard/app.py` — all pages appear in the sidebar automatically.

### Page 1: Command Centre (`app.py`)

| Panel | What You See |
|-------|-------------|
| 📋 **Recent Alerts** | Sortable feed of latest alerts with colour-coded priority badges |
| 🥧 **Priority Distribution** | Pie and bar chart breakdown of CRITICAL / HIGH / MEDIUM / LOW |
| 📈 **Anomaly Trend** | Score-over-time chart for the most recent session |
| 📦 **Session Summaries** | Per-run statistics read from `data/results/session_*.json` |
| 🔍 **Detection Activity** | Detection count and class distribution trends |
| 🌍 **Threat Heatmap** | Geographic threat map of Indian border zones (Plotly Mapbox) |
| 🔔 **Notification Status** | SendGrid readiness indicator + manual email trigger button |

### Page 2: Enhanced Analysis (`pages/1_🗺️_Enhanced_Analysis.py`)

| Panel | What You See |
|-------|-------------|
| 📈 **Risk Score Timeline** | Zone risk + temporal risk over time (dual chart) |
| 🗺️ **Zone Violations** | Bar chart of intrusions per zone (border / buffer / observation) |
| ⏱️ **Temporal Alert Types** | Donut chart of 5 temporal detector categories |
| 🗺️ **Zone Severity Heatmap** | Frame-by-frame zone severity scatter plot |
| 🎯 **Object Tracking** | Active tracked objects over time |
| 📊 **Detection Trend** | Rolling detection count slope (positive = increasing activity) |

**Primary data inputs:**

```
data/alerts/alert_log.json
data/results/session_*.json
data/results/enhanced_analysis.json     ← zone + temporal data from enhanced pipeline
data/detections/anomaly_summary.json    ← optional
```

> If no live pipeline output is present, both dashboard pages automatically fall back to demo data so the interface always remains fully functional and reviewable.

---

## 🧩 Key Modules

### `src/pipeline.py`
Main orchestration layer with zone + temporal intelligence. Connects preprocessing → detection → zone analysis → temporal analysis → anomaly scoring → alert management in a single runtime session. Adds zone intrusion detection and temporal video analysis as parallel processing stages alongside anomaly scoring. Accepts CLI arguments including `--temporal-window`, `--no-zones`, and `--no-temporal` flags. Saves session summaries and optionally annotated frames, then uploads to Azure when configured.

### `src/detector.py`
YOLOv8 wrapper. Loads the custom-trained `border_yolo.pt` model once per session and reuses it across all frames. Converts raw model output into structured `Detection` objects carrying class, confidence, bbox, threat tags, and frame-level `has_high` / `has_critical` flags.

### `src/anomaly.py`
Dual ML pipeline: **Isolation Forest** for unsupervised anomaly scoring and **Random Forest Classifier** for supervised threat classification. Baseline collected from the first 30 frames, then live scoring on 10-dimensional behavioural features. Produces interpretable anomaly reasons and severity classifications attached to each scored frame result.

### `src/zone_analyzer.py`
Spatial intelligence layer. Defines three configurable border surveillance zones (RESTRICTED, BUFFER, OBSERVATION) as normalised polygons. Every detection is tested against zone boundaries using ray-casting point-in-polygon. High-threat classes in restricted zones trigger immediate CRITICAL alerts. Supports night activity boosting and composite risk scoring.

### `src/temporal_analyzer.py`
Multi-frame video analysis engine. Maintains a sliding window of recent frames and runs five temporal detectors: sudden appearance, crowd buildup, loitering, approach trajectory, and coordinated movement. Includes a lightweight IoU-based object tracker that associates detections across frames without GPU dependency.

### `src/alert_manager.py`
Priority assignment from anomaly level, zone violations, and temporal alerts. Maintains the rolling JSON alert log with append-only writes. Sends SendGrid notifications for HIGH+ events within cooldown constraints. Writes alert records to Cosmos DB via `azure_client`.

### `src/azure_client.py`
Lazy initialisation — Azure clients are created only when valid credentials are present. Uploads session JSON files to Blob Storage. Writes alert documents to Cosmos DB. Falls back gracefully to a no-op if Azure is unavailable, with no pipeline interruption.

### `dashboard/app.py`
Main dashboard page. Reads `alert_log.json` and `session_*.json` from local `data/` directories. Auto-refreshes on a configurable interval. Renders alert feed, priority charts, session summaries, anomaly trend, geographic threat heatmap, and manual notification control.

### `dashboard/pages/1_🗺️_Enhanced_Analysis.py`
Enhanced analysis dashboard page (auto-detected by Streamlit multi-page system). Visualises zone intrusion data, temporal alert breakdowns, risk score timelines, object tracking counts, and detection trends. Falls back to demo data when no enhanced pipeline output is present.

---

## 🧪 Sample Output & Results

### Custom YOLOv8 Model Performance (v2 Balanced — 30 Epochs)

<div align="center">

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| **All** | **1685** | **30195** | **0.701** | **0.457** | **0.478** | **0.285** |
| person | 603 | 10387 | 0.463 | 0.283 | 0.293 | 0.096 |
| vehicle | 999 | 8170 | 0.327 | 0.398 | 0.270 | 0.155 |
| crowd | 24 | 149 | 0.713 | 0.367 | 0.483 | 0.197 |
| military_vehicle | 4 | 8 | 1.000 | 0.000 | 0.000 | 0.000 |
| aircraft | 425 | 2353 | 0.831 | 0.622 | 0.668 | 0.479 |
| ship | 527 | 8169 | 0.780 | 0.821 | 0.870 | 0.561 |
| suspicious_object | 116 | 959 | 0.791 | 0.706 | 0.763 | 0.505 |

</div>

> **Training config:** YOLOv8n · 30 epochs · 640×640 · CPU (12th Gen Intel i5-12350U) · Balanced v2 dataset (DOTA + xView + Visdrone)
> **Inference speed:** 1.4ms preprocess · 114.7ms inference · 1.4ms postprocess per image

---

**Example alert record written to `data/alerts/alert_log.json`:**

```json
{
  "alert_id": "alert_1712312345678",
  "frame_id": 42,
  "priority": "CRITICAL",
  "alert_level": "critical",
  "anomaly_score": -0.1042,
  "detection_count": 7,
  "motion_score": 10.4,
  "reasons": [
    "person (85%) detected in Border Zone (Restricted) [HIGH THREAT]",
    "Border approach detected: 2 object(s) moving toward border zone",
    "crowd gathering detected"
  ],
  "notified": true
}
```

**Example enhanced session summary:**

```json
{
  "source": "dota_aerial_test.mp4",
  "pipeline_type": "enhanced",
  "total_frames": 124,
  "frames_scored": 94,
  "total_detections": 2351,
  "alerts_raised": 38,
  "zone_analysis_enabled": true,
  "total_zone_violations": 34,
  "zone_critical_count": 8,
  "temporal_analysis_enabled": true,
  "total_temporal_alerts": 12,
  "temporal_approach_count": 2,
  "avg_inference_ms": 114.7,
  "avg_zone_ms": 0.8,
  "avg_temporal_ms": 1.2
}
```

**Typical output file set after an enhanced pipeline run:**

```
data/alerts/alert_log.json
data/results/enhanced_session_<source>_<timestamp>.json
data/results/enhanced_analysis.json      ← zone + temporal data for dashboard
data/detections/frame_000042.jpg         ← if --save-frames enabled
data/logs/pipeline.log
runs/detect/                             ← YOLO inference artefacts
```

---

## 🧾 Testing

The project includes an automated test suite covering all major runtime modules.

```bash
# Run full suite with verbose output
pytest tests -v

# Run with HTML coverage report
pytest tests --cov=src --cov-report=html
# Open: htmlcov/index.html in your browser

# Run individual test files
pytest tests/test_detector.py -v
pytest tests/test_anomaly_and_alert.py -v
pytest tests/test_pipeline.py -v
pytest tests/test_zone_analyzer.py -v
pytest tests/test_temporal_analyzer.py -v
```

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| `test_detector.py` | — | YOLOv8 model loading, inference execution, structured detection output format |
| `test_anomaly_and_alert.py` | — | Baseline learning, Isolation Forest scoring, priority assignment logic, alert log writes |
| `test_pipeline.py` | — | End-to-end orchestration, output file creation, inter-module integration |
| `test_zone_analyzer.py` | 23 | Point-in-polygon, zone violation detection, risk scoring, custom zone configs |
| `test_temporal_analyzer.py` | 28 | IoU tracking, sudden appearance, crowd buildup, loitering, approach trajectory |

---

## 📈 Future Improvements

- ~~Custom YOLOv8 fine-tuning on annotated border-specific datasets~~ ✅ **Completed** — custom model trained on balanced v2 dataset (DOTA + xView + Visdrone)
- ~~Zone-based intrusion detection~~ ✅ **Completed** — 3-zone spatial intelligence with night boost
- ~~Multi-frame temporal analysis~~ ✅ **Completed** — 5 temporal detectors + IoU object tracking
- ~~Random Forest threat classifier~~ ✅ **Completed** — dual ML pipeline alongside Isolation Forest
- ~~Geographic threat heatmap~~ ✅ **Completed** — Plotly Mapbox Indian border zone visualisation
- Stronger model and version tracking for reproducible production deployments
- REST API layer to expose pipeline controls programmatically
- Alert frame thumbnails with direct Azure Blob links embedded in dashboard views
- Automated CI/CD deployment profiles for staging and production environments
- Multi-camera ingestion with centralised alert aggregation across feeds
- Real-time streaming support via RTSP or WebRTC camera feeds
- DeepSORT or ByteTrack integration for production-grade multi-object tracking

---

## 🎓 Academic Context

<div align="center">

| Field | Detail |
|-------|--------|
| **Program** | Microsoft Elevate — GTU Internship 2026 |
| **Powered By** | Edunet Foundation & FICE Education |
| **College** | SAL Institute of Technology and Engineering Research |
| **Department** | Information & Communication Technology (ICT) |
| **Semester** | 8th Semester |
| **Duration** | January 2026 — April 2026 (12 weeks · 420 hours) |
| **Problem Domain** | Border Defence and Surveillance (GTU) |
</div>

**GTU domain requirements fulfilled by this project:**

- ✅ EDA on surveillance and sensor datasets
- ✅ Anomaly detection model to identify unusual activity patterns (Isolation Forest + Random Forest dual pipeline)
- ✅ ML/DL object classification of movement patterns using custom-trained YOLOv8 (7 border-specific classes)
- ✅ Alert prioritisation system with zone intrusion, temporal analysis, and anomaly scoring
- ✅ Cloud-based data integration using Microsoft Azure (Blob Storage + Cosmos DB)
- ✅ Spatial intelligence — zone-based intrusion detection with configurable border zones
- ✅ Video-level temporal intelligence — multi-frame tracking, loitering detection, approach trajectory analysis

---

## 👨‍💻 Author

<div align="center">

| Field | Detail |
|-------|--------|
| **Name** | Jainil Gupta (Jay Gupta) |
| **Role** | Solo Developer — ML Engineer · Cloud Architect · System Designer |
| **Enrollment** | 220670132018 |
| **Linkedin** | [@jainilgupta](https://linkedin.com/in/jainilgupta) |
| **Internal Guide** | Prof. Chintan Rana |
| **External Guide** | Adarsh Gupta |

</div>

---

## 📜 License

This project is distributed under the **MIT License** — see [`LICENSE`](LICENSE) for full details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d47a1,50:1a1f35,100:0d1117&height=120&section=footer" width="100%"/>

**⭐ If this project was useful, please consider starring the repository!**

*Built with ❤️ by Jainil Gupta · Microsoft Elevate Internship 2026 · SAL Institute of Technology and Engineering Research*

</div>
