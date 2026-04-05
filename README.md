<div align="center">
 
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1f35,100:0d47a1&height=220&section=header&text=Border%20Surveillance%20AI&fontSize=52&fontColor=00D9FF&fontAlignY=38&desc=Real-Time%20Detection%20%7C%20Anomaly%20Scoring%20%7C%20Smart%20Alerting%20%7C%20Azure%20Cloud%20%7C%20Streamlit&descAlignY=58&descSize=15&descColor=CADCFC" width="100%"/>
 
<br/>
 
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
  Live Demo Result (April 2026):  2 persons + 1 vehicle detected · 78% avg confidence · 18-second end-to-end workflow
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
- YOLOv8 inference on every extracted frame
- Detects: `person`, `vehicle`, `crowd`, `military_vehicle`, `aircraft`, `ship`, `suspicious_object`
- Structured per-detection output with class, confidence, bounding box, and threat tags
- `has_high` and `has_critical` flags per frame for downstream prioritisation
 
### 🧠 Anomaly Detection
- Baseline learned from first 30 frames (configurable)
- Isolation Forest scoring on low-dimensional behavioural features
- Features include detection count, class diversity, confidence stats, motion score, object location, and suspicious class presence
- Frame-level severity: `normal` → `high` → `critical`
 
</td>
<td width="50%" valign="top">
 
### 🚨 Smart Alert System
- Four priority levels: 🔴 `CRITICAL` / 🟠 `HIGH` / 🟡 `MEDIUM` / 🟢 `LOW`
- Rolling JSON alert log written locally on every run
- Cooldown logic to suppress duplicate notifications
- Email notifications via SendGrid for HIGH and CRITICAL alerts
- Azure Cosmos DB persistence when credentials are configured
 
### 📊 Operational Dashboard
- Streamlit command-centre interface
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
│   │   YOLOv8 Inference                                                 │ │
│   │   Detects → person │ vehicle │ crowd │ military_vehicle │ aircraft │ │ 
│   │             ship │ suspicious_object                               │ │
│   │   Output → class · confidence · bbox · threat_tag · flags          │ │
│   └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                     ANOMALY DETECTION                 [src/anomaly.py]   │
│                                                                          │
│   Phase 1: Baseline (frames 1–30) → learn normal behaviour               │
│   Phase 2: Live scoring → Isolation Forest on behavioural features       │
│   Output  → anomaly score │ severity │ human-readable reasons            │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                    ALERT MANAGEMENT              [src/alert_manager.py]  │
│                                                                          │
│   Score mapping →  🔴 CRITICAL  │  🟠 HIGH  │  🟡 MEDIUM  │  🟢 LOW    │
│   Rolling JSON log  │  Cooldown dedup  │  SendGrid email notifications   │
│   Azure Cosmos DB write (if configured)                                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────────────┐
│               OUTPUT LAYER + AZURE + DASHBOARD                           │
│                                                                          │
│  📄 data/alerts/alert_log.json       🗄️  Azure Cosmos DB (alerts)       │
│  📄 data/results/session_*.json      📦  Azure Blob Storage (sessions)  │
│  🖼️  data/detections/frame_*.jpg                                        |
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │             📊  Streamlit Dashboard  (dashboard/app.py)          │   |
│  │  Alert feed │ Priority chart │ Session summaries │ Trend lines   │    │
│  │  Anomaly overview │ Manual notify button │ Auto-refresh          │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```
 
---
 
## 🔄 How It Works
 
### 1️⃣ Input and Preprocessing
The pipeline accepts either a video file path or a live camera index. Frames are loaded through OpenCV, resized to 640×640, and optionally passed through an optical flow computation that estimates per-frame motion intensity. Each  frame becomes a structured `frame_item` dictionary passed to downstream modules.
 
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
 
Frames are classified as `normal`, `high`, or `critical` with human-readable anomaly reasons attached.
 
### 4️⃣ Alert Generation
`AlertManager` maps anomaly output to operational priority levels. HIGH and CRITICAL alerts trigger email notifications subject to a configurable cooldown window. Every non-normal alert is appended to the rolling JSON log. Alert records are optionally written to Azure Cosmos DB.
 
### 5️⃣ Storage and Monitoring
Session summaries are saved as timestamped JSON files. When Azure credentials are present, session results are uploaded to Blob Storage and alerts are written to Cosmos DB. The Streamlit dashboard reads local output files directly and auto-refreshes to show the latest operational state.
 
---
 
## ⚙️ Tech Stack
 
<div align="center">
 
| Layer | Technology | Role |
|-------|-----------|------|
| **Language** | Python 3.9+ | Core runtime |
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time frame inference |
| **Deep Learning** | PyTorch 2.x | Model backend |
| **Computer Vision** | OpenCV 4.x | Video I/O, frame processing, optical flow |
| **Anomaly Detection** | scikit-learn (Isolation Forest) | Behavioural scoring |
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
│   ├── pipeline.py                   # Main orchestration entry point
│   ├── detector.py                   # YOLOv8 wrapper + structured detections
│   ├── anomaly.py                    # Baseline learning + Isolation Forest scoring
│   ├── alert_manager.py              # Priority assignment, logging, email, cooldown
│   └── azure_client.py              # Blob Storage + Cosmos DB integration
│
├── 📁 dashboard/
│   ├── app.py                        # Streamlit command-centre dashboard
│   └── Border Defence AI logo.png    # Project branding asset
│
├── 📁 scripts/                       # Utility and dataset preparation scripts
│   ├── main.py                       # Minimal detector demo over a test video
│   ├── pilot.py                      # Manual integration checker across all modules
│   ├── smoke_test.py                 # Quick pipeline smoke test
│   ├── generate_test_video.py        # Synthetic test video generator
│   ├── preprocess_all_datasets.py    # Full dataset preprocessing
│   ├── preprocess_local.py           # Local dataset preprocessing
│   ├── convert_xview_to_yolo.py      # xView → YOLO format converter
│   ├── xview_geojson_to_yolo.py      # xView GeoJSON → YOLO labels
│   ├── fix_xview_patch.py            # xView patch correction utility
│   ├── fix_vedai_crowd.py            # VEDAI crowd class fix
│   └── smart_extract.py             # Intelligent frame extractor
│
├── 📁 data/                          # Runtime data (gitignored — not in repo)
│   ├── alerts/                       # alert_log.json — rolling alert output
│   ├── results/                      # session_*.json — per-run summaries
│   ├── test_videos/                  # Sample videos for local runs
│   ├── processed/                    # Preprocessed training-ready data
│   ├── annotations/                  # Dataset annotation files
│   ├── raw/                          # Source datasets
│   └── logs/                         # Pipeline runtime logs
│
├── 📁 tests/                         # Automated test suite
│   ├── test_detector.py
│   ├── test_anomaly_and_alert.py
│   └── test_pipeline.py
│
├── 📁 models/                        # YOLO weights + anomaly model artefacts
├── 📁 notebooks/                     # EDA and experimentation notebooks
├── 📁 docs/                          # Architecture diagrams + presentations
├── 📁 overview/                      # Implementation guide and references
│
├── yolov8n.pt                        # YOLOv8 nano weights (included)
├── yolov8s.pt                        # YOLOv8 small weights (included)
├── requirements.txt                  # All Python dependencies
├── pyproject.toml                    # Project metadata
├── pytest.ini                        # Test runner configuration
├── makefile                          # Common task shortcuts
├── .env                              # Local credentials (never committed)
├── .gitignore                        # Excludes data/, venv/, *.pt, .env
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
 
> 💡 **YOLOv8 weights** (`yolov8n.pt`) are already included in the repo root. No separate download is needed.
 
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
  --results-dir data/results \
  --annotated-dir data/detections
```
 
**Live camera mode:**
 
```bash
python src/pipeline.py --camera 0
```
 
**Expected terminal output:**
 
```
✅ Preprocessing complete   — frames extracted and resized to 640×640
✅ Detection complete       — structured detections logged per frame
✅ Anomaly scoring complete — Isolation Forest scored all frames
✅ Alerts generated         — priority levels assigned and logged
✅ Session saved            — data/results/session_<source>_<timestamp>.json
✅ Alert log written        — data/alerts/alert_log.json
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
 
`dashboard/app.py` provides a command-centre style operational view over all pipeline output.
 
| Panel | What You See |
|-------|-------------|
| 📋 **Recent Alerts** | Sortable feed of latest alerts with colour-coded priority badges |
| 🥧 **Priority Distribution** | Pie and bar chart breakdown of CRITICAL / HIGH / MEDIUM / LOW |
| 📈 **Anomaly Trend** | Score-over-time chart for the most recent session |
| 📦 **Session Summaries** | Per-run statistics read from `data/results/session_*.json` |
| 🔍 **Detection Activity** | Detection count and class distribution trends |
| 🔔 **Notification Status** | SendGrid readiness indicator + manual email trigger button |
 
**Primary data inputs:**
 
```
data/alerts/alert_log.json
data/results/session_*.json
data/detections/anomaly_summary.json    ← optional
```
 
> If no live pipeline output is present, the dashboard automatically falls back to demo data so the interface always remains fully functional and reviewable.
 
---
 
## 🧩 Key Modules
 
### `src/pipeline.py`
Main orchestration layer. Connects preprocessing → detection → anomaly scoring → alert management in a single runtime session. Saves session summaries and optionally annotated frames, then uploads to Azure when configured.
 
### `src/detector.py`
YOLOv8 wrapper. Loads the model once per session and reuses it across all frames. Converts raw model output into structured `Detection` objects carrying class, confidence, bbox, threat tags, and frame-level `has_high` / `has_critical` flags.
 
### `src/anomaly.py`
Two-phase runtime: baseline collection from the first 30 frames, then Isolation Forest live scoring for all subsequent frames. Produces interpretable anomaly reasons and severity classifications attached to each scored frame result.
 
### `src/alert_manager.py`
Priority assignment from anomaly level and motion score. Maintains the rolling JSON alert log with append-only writes. Sends SendGrid notifications for HIGH+ events within cooldown constraints. Writes alert records to Cosmos DB via `azure_client`.
 
### `src/azure_client.py`
Lazy initialisation — Azure clients are created only when valid credentials are present. Uploads session JSON files to Blob Storage. Writes alert documents to Cosmos DB. Falls back gracefully to a no-op if Azure is unavailable, with no pipeline interruption.
 
### `dashboard/app.py`
Reads `alert_log.json` and `session_*.json` from local `data/` directories. Auto-refreshes on a configurable interval. Renders alert feed, priority charts, session summaries, anomaly trend, and manual notification control.
 
---
 
## 🧪 Sample Output & Results
 
**Example alert record written to `data/alerts/alert_log.json`:**
 
```json
{
  "alert_id": "alert_1712312345678",
  "frame_id": 42,
  "priority": "HIGH",
  "alert_level": "high",
  "anomaly_score": -0.1042,
  "detection_count": 7,
  "motion_score": 10.4,
  "reasons": ["crowd gathering detected"],
  "notified": true
}
```
 
**Example session summary written to `data/results/session_dota_aerial_test_20260405_144229.json`:**
 
```json
{
  "source": "dota_aerial_test.mp4",
  "timestamp": "2026-04-05T14:42:29",
  "total_frames": 124,
  "frames_scored": 94,
  "critical_alerts": 1,
  "high_alerts": 3,
  "medium_alerts": 8,
  "low_alerts": 82,
  "avg_anomaly_score": -0.073,
  "avg_confidence": 0.78,
  "azure_uploaded": true
}
```
 
**Typical output file set after a single pipeline run:**
 
```
data/alerts/alert_log.json
data/results/session_<source>_<timestamp>.json
data/detections/frame_000042.jpg         ← if --save-frames enabled
data/logs/pipeline.log
runs/detect/                             ← YOLO inference artefacts
```
 
**Live demo results — Final presentation, April 2026:**
 
```
Upload video → Azure Blob trigger → YOLOv8 inference → Cosmos DB write → Email alert
Total end-to-end time   : 18 seconds
Objects detected        : 2 persons · 1 vehicle
Average confidence      : 78%
Dashboard refresh time  : < 5 seconds
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
```
 
| Test File | Coverage Area |
|-----------|--------------|
| `test_detector.py` | YOLOv8 model loading, inference execution, structured detection output format |
| `test_anomaly_and_alert.py` | Baseline learning, Isolation Forest scoring, priority assignment logic, alert log writes |
| `test_pipeline.py` | End-to-end orchestration, output file creation, inter-module integration |
 
---
 
## 📈 Future Improvements
 
- Stronger model and version tracking for reproducible production deployments
- REST API layer to expose pipeline controls programmatically
- Alert frame thumbnails with direct Azure Blob links embedded in dashboard views
- Automated CI/CD deployment profiles for staging and production environments
- Multi-camera ingestion with centralised alert aggregation across feeds
- Custom YOLOv8 fine-tuning on annotated border-specific datasets (VEDAI, xView, DOTA)
- Real-time streaming support via RTSP or WebRTC camera feeds
 
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
- ✅ Anomaly detection model to identify unusual activity patterns
- ✅ ML/DL object classification of movement patterns using YOLOv8
- ✅ Alert prioritisation system that reduces false positives via confidence filtering
- ✅ Cloud-based data integration using Microsoft Azure (Blob Storage + Cosmos DB)
 
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
