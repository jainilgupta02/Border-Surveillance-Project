<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1f35,100:0d47a1&height=200&section=header&text=Border%20Surveillance%20Project&fontSize=42&fontColor=00D9FF&fontAlignY=38&desc=AI-Powered%20Real-Time%20Detection%20%7C%20YOLOv8%20%7C%20Azure%20%7C%20Power%20BI&descAlignY=58&descSize=16&descColor=CADCFC" width="100%"/>

<!-- BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Microsoft%20Azure-Cloud-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/GTU-Internship%202026-00D9FF?style=flat-square"/>
  <img src="https://img.shields.io/badge/Microsoft%20Elevate-Program-0078D4?style=flat-square&logo=microsoft"/>
  <img src="https://img.shields.io/badge/SAL%20Institute-ICT%20Dept-4CAF50?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/jainilgupta02/Border-Surveillance-Project?style=flat-square&color=yellow"/>
  <img src="https://img.shields.io/github/license/jainilgupta02/Border-Surveillance-Project?style=flat-square&color=blue"/>
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square"/>
</p>

<br/>

> **🛡️ An end-to-end AI surveillance system** that detects objects, identifies anomalies, prioritizes alerts, and visualizes real-time threats — aligned with national border security requirements.

</div>

---

## 📌 Table of Contents

<details>
<summary>Click to expand</summary>

- [🎯 About the Project](#-about-the-project)
- [🚨 Problem Statement](#-problem-statement)
- [⚡ Key Features](#-key-features)
- [🏗️ System Architecture](#-system-architecture)
- [🛠️ Tech Stack](#-tech-stack)
- [📂 Repository Structure](#-repository-structure)
- [📊 Datasets](#-datasets)
- [🗓️ 12-Week Roadmap](#️-12-week-roadmap)
- [🚀 Getting Started](#-getting-started)
- [☁️ Azure Deployment](#-azure-deployment)
- [📈 Dashboard Preview](#-dashboard-preview)
- [📏 Evaluation Metrics](#-evaluation-metrics)
- [👥 Team](#-team)
- [🎓 Academic Context](#-academic-context)
- [📜 License](#-license)

</details>

---

## 🎯 About the Project

Traditional border surveillance relies heavily on **manual monitoring**, which leads to delayed threat detection, alert fatigue, and high operational costs. This project builds an intelligent, automated system that:

- 🔍 **Detects objects** (people, vehicles, weapons) in real-time using YOLOv8
- 🧠 **Identifies anomalies** (unusual movement, crowd spikes, suspicious patterns) using Ensemble ML
- 🚨 **Prioritizes alerts** using confidence scoring to reduce false positives
- ☁️ **Stores and syncs** alert logs on Microsoft Azure
- 📊 **Visualizes** incident trends and hotspots via a Power BI dashboard

<br/>

```
 Traditional System                 Our AI System
 ─────────────────                 ─────────────
 👁️  1 guard watches 8 cameras     🤖 AI watches all cameras 24/7
 ⏳  Delayed human reaction         ⚡ <1 sec anomaly detection
 ❌  High false alarm fatigue       ✅ Confidence-based filtering
 📁  Siloed, manual records         ☁️  Centralized Azure alert logs
 📉  No trend insights              📈  Power BI real-time dashboard
```

---

## 🚨 Problem Statement

> *Domain: Border Defence and Surveillance — GTU Internship 2026*

| Challenge | Impact |
|-----------|--------|
| **Large-scale monitoring** | Vast border areas exceed human monitoring capacity |
| **Delayed threat detection** | Manual analysis causes late identification of intrusions |
| **High false alarm rates** | Animals, weather, noise trigger unnecessary responses |
| **Resource constraints** | Limited manpower must cover extensive remote regions |
| **Poor data integration** | Sensor, camera, and historical data remain siloed |

**Our solution addresses all five challenges** through AI automation, cloud integration, and intelligent alerting.

---

## ⚡ Key Features

<table>
<tr>
<td width="50%">

### 🔍 Detection Module
- Real-time object detection (30+ FPS)
- Detects: persons, vehicles, weapons, suspicious objects
- Bounding box generation with confidence scores
- Powered by **YOLOv8** (state-of-the-art)

### 🧠 Anomaly Detection
- Optical flow-based motion analysis
- Ensemble model (Isolation Forest + Random Forest)
- Frame-level anomaly scoring (0.0 → 1.0)
- Unsupervised: works without labelled anomaly data

</td>
<td width="50%">

### 🚨 Smart Alert System
- Priority levels: 🔴 High / 🟡 Medium / 🟢 Low
- Confidence threshold filtering (reduces false positives)
- Email + SMS notifications for critical alerts
- Alert log stored in Azure Cosmos DB

### 📊 Monitoring Dashboard
- Real-time alert feed (last 24 hrs)
- Anomaly heatmap of high-risk zones
- Detection count trends over time
- Built with **Power BI** + optional **Streamlit**

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                    │
│   📹 Video Surveillance Feed  →  Frame Extraction (OpenCV)           │
└──────────────────────────┬───────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                     │
│   Resize (640×640) │ Normalize (0-1) │ Upload → Azure Blob Storage   │
└──────────────────────────┬───────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     DETECTION LAYER                      [Azure Fn]  │
│                                                                       │
│   ┌─────────────────────────┐    ┌────────────────────────────────┐  │
│   │    🎯 YOLOv8            │    │    🧠 Anomaly Detection        │  │
│   │    Object Detection     │───▶│    Ensemble (RF + IF)          │  │
│   │                         │    │                                │  │
│   │  Persons │ Vehicles      │    │  Score: 0.0 ──────────── 1.0  │  │
│   │  Weapons │ Objects       │    │         Normal       Anomaly  │  │
│   └─────────────────────────┘    └────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     ALERT PROCESSING                                  │
│   Score > 0.7 → 🔴 HIGH   │   0.4–0.7 → 🟡 MEDIUM  │  <0.4 → 🟢 LOW │
│   Log to Azure Cosmos DB  │   Attach frame thumbnail  │   Filter out  │
└──────────────────────────┬───────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│               NOTIFICATION + VISUALIZATION                            │
│   📧 Email Alert  │  📱 SMS (Critical)  │  🔔 Dashboard Push        │
│                                                                       │
│   ┌───────────────────────────────────────────────────────────────┐  │
│   │              📊 Power BI / Streamlit Dashboard                │  │
│   │   Real-time feed │ Heatmap │ Trend charts │ Alert table       │  │
│   └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      AZURE CLOUD STORAGE                              │
│   📦 Blob Storage (Frames)  │  🗄️ Cosmos DB (Alerts)  │  📁 GitHub   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.9+ | Core development |
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time detection |
| **Deep Learning** | TensorFlow, Keras | Model training |
| **Machine Learning** | Scikit-learn | Anomaly detection |
| **Computer Vision** | OpenCV | Video processing |
| **Data** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn, Plotly | Analysis charts |
| **Cloud Storage** | Azure Blob Storage | Frame & data storage |
| **Serverless** | Azure Functions | Event-driven processing |
| **Database** | Azure Cosmos DB | Alert log storage |
| **AI API** | Azure Computer Vision | Enhanced analysis |
| **Dashboard** | Power BI + DAX | Real-time monitoring |
| **Web App** | Streamlit | Python dashboard |
| **CI/CD** | GitHub Actions | Auto deployment |
| **Container** | Docker | Portable deployment |
| **Annotation** | Roboflow | Dataset labelling |

</div>

---

## 📂 Repository Structure

```
Border-Surveillance-Project/
│
├── 📁 data/
│   ├── processed/           # Cleaned & preprocessed frames
│   ├── annotations/         # Roboflow annotation exports
│   ├── sample/              # Small sample data for testing
│   └── README.md            # Dataset download instructions
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb          # Data preprocessing pipeline
│   ├── 03_Object_Detection.ipynb       # YOLOv8 training & eval
│   ├── 04_Anomaly_Detection.ipynb      # Ensemble model training
│   └── 05_End_to_End_Demo.ipynb        # Full pipeline demo
│
├── 📁 src/
│   ├── preprocessing.py        # Frame extraction & normalization
│   ├── detect_objects.py       # YOLOv8 inference wrapper
│   ├── anomaly_detector.py     # Ensemble anomaly scoring
│   ├── alert_manager.py        # Alert prioritization & logging
│   ├── azure_uploader.py       # Azure Blob/CosmosDB integration
│   └── run_pipeline.py         # Main end-to-end script
│
├── 📁 azure/
│   ├── functions/
│   │   └── anomaly_trigger/    # Azure Function (HTTP/Blob trigger)
│   ├── arm_templates/          # Infrastructure-as-code templates
│   └── deployment_guide.md     # Step-by-step Azure setup
│
├── 📁 dashboard/
│   ├── surveillance_dashboard.pbix    # Power BI dashboard file
│   ├── streamlit_app.py               # Python web dashboard
│   └── assets/                        # Dashboard screenshots
│
├── 📁 models/
│   ├── yolov8_border.pt               # Trained YOLOv8 weights
│   ├── anomaly_ensemble.pkl           # Anomaly detection model
│   └── model_card.md                  # Model info & metrics
│
├── 📁 results/
│   ├── metrics/                       # Precision, recall, F1 CSVs
│   ├── screenshots/                   # Demo screenshots
│   └── charts/                        # Performance visualizations
│
├── 📁 tests/
│   ├── test_detection.py
│   ├── test_anomaly.py
│   └── test_azure_upload.py
│
├── 📁 docs/
│   ├── project_report.pdf             # Full internship report
│   ├── presentation.pptx              # Internship presentation
│   └── weekly_reports/                # GTU weekly report docs
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml                  # GitHub Actions pipeline
│
├── config.example.yaml                # Template config (safe to commit)
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container setup
├── docker-compose.yml                 # Multi-service setup
├── .gitignore                         # Ignored files
└── README.md                          # This file
```

---

## 📊 Datasets

| Dataset | Size | Type | Use | Link |
|---------|------|------|-----|------|
| **UCF-Crime** | 128 hrs / 1,900 videos | Surveillance CCTV | Anomaly detection training | [🔗 UCF](https://www.crcv.ucf.edu/projects/real-world/) |
| **xView** | 1M+ objects, 0.3m res | Satellite imagery | Aerial object detection | [🔗 xView](https://xviewdataset.org/) |
| **DOTA v2** | 280K objects | Aerial images | Vehicle/infra detection | [🔗 DOTA](https://captain-whu.github.io/DOTA/) |
| **COCO** | 330K images | General | YOLOv8 pre-training base | [🔗 COCO](https://cocodataset.org/) |
| **Roboflow Custom** | TBD | Annotated frames | Border-specific fine-tuning | Private |

> ⚠️ **Note:** Large datasets are not committed to this repo. See [`data/README.md`](data/README.md) for download and setup instructions.

---

## 🗓️ 12-Week Roadmap

```
Jan 2026                                                      Apr 2026
  │                                                              │
  ●──────●──────●──────●──────●──────●──────●──────●──────●────●
  W1     W2     W3     W4     W5     W6     W7     W8     W9   W10  W11  W12
  │      │      │      │      │      │      │      │      │      │
  ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
 [Plan] [EDA] [Det.] [Det.] [Ano.] [Ano.] [Azure][Azure][Dash] [Dash][Test][Final]
```

| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 🔵 **Foundation** | 1-2 | Planning, EDA, environment setup | ✅ Done |
| 🟡 **Detection** | 3-4 | YOLOv8 training & evaluation | 🔄 In Progress |
| 🟠 **Anomaly** | 5-6 | Ensemble ML, alert prioritization | ⏳ Upcoming |
| 🔴 **Cloud** | 7-8 | Azure deployment, CI/CD | ⏳ Upcoming |
| 🟣 **Dashboard** | 9-10 | Power BI, Streamlit, notifications | ⏳ Upcoming |
| 🟢 **Finalize** | 11-12 | Testing, docs, demo, report | ⏳ Upcoming |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.9 or higher
python --version

# Git
git --version
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jainilgupta02/Border-Surveillance-Project.git
cd Border-Surveillance-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# OR
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy config template and fill credentials
cp config.example.yaml config.yaml
# Edit config.yaml with your Azure credentials
```

### Run the Demo Pipeline

```bash
# Run on a sample video
python src/run_pipeline.py --video data/sample/test_video.mp4

# Output:
# ✅ Frame extraction complete   (frames saved to data/processed/)
# ✅ Object detection complete   (detections logged)
# ✅ Anomaly scoring complete    (anomaly scores computed)
# ✅ Alerts generated            (2 HIGH, 3 MEDIUM alerts)
# ✅ Results saved to            results/metrics/run_001.csv
# ✅ Azure upload complete       (if credentials configured)
```

### Launch Dashboard

```bash
# Option A: Streamlit web dashboard
streamlit run dashboard/streamlit_app.py

# Option B: Open Power BI file
# Open: dashboard/surveillance_dashboard.pbix
# Refresh data source pointing to results/metrics/
```

---

## ☁️ Azure Deployment

Minimal Azure services used in this project:

```
┌─────────────────────────────────────────────────────────────┐
│  Azure Services Used                                         │
│                                                              │
│  📦 Blob Storage   → Store processed frames & alert CSVs    │
│  ⚡ Functions      → Serverless anomaly trigger              │
│  🗄️ Cosmos DB      → Alert log database (NoSQL)             │
│  🔍 Computer Vision → Optional enhanced image analysis      │
│  📊 App Insights   → Performance monitoring                  │
└─────────────────────────────────────────────────────────────┘
```

> 📖 See [`azure/deployment_guide.md`](azure/deployment_guide.md) for full step-by-step Azure setup.

---

## 📈 Dashboard Preview

> 🚧 Dashboard screenshots will be added after Week 9 implementation.

| Component | Description |
|-----------|-------------|
| 📊 Alert Timeline | Bar chart of alerts per hour |
| 🗺️ Heatmap | High-risk zones visualized on grid |
| 📋 Alert Table | Sortable list: time, type, score, priority |
| 📈 Trend Line | Anomaly score over time |
| 🔢 KPI Cards | Total alerts, false positive rate, uptime |

---

## 📏 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Detection Accuracy** | > 90% | YOLOv8 mAP@0.5 on test set |
| **False Positive Rate** | < 20% | Alert reliability |
| **Processing Speed** | > 20 FPS | Real-time capability |
| **Alert Response Time** | < 1 second | From detection to log |
| **System Uptime** | > 99% | Azure deployment reliability |

---

## 👥 Team

<div align="center">

| Name | Role | GitHub |
|------|------|--------|
| **Jainil Gupta** | Team Lead · ML Engineer · Cloud Architect | [@jainilgupta02](https://github.com/jainilgupta02) |
| *(Team Member 2)* | *(Role — e.g., Frontend / Dashboard)* | *(GitHub link)* |
| *(Team Member 3)* | *(Role — e.g., Data / Model Training)* | *(GitHub link)* |
| *(Team Member 4)* | *(Role — e.g., Azure / DevOps)* | *(GitHub link)* |

</div>

---

## 🎓 Academic Context

<div align="center">

| | |
|--|--|
| **Program** | Microsoft Elevate — GTU Internship 2026 |
| **Powered By** | Edunet Foundation & FICE Education |
| **College** | SAL Institute of Technology and Engineering Research |
| **Department** | Information & Communication Technology (ICT) |
| **Semester** | 8th Semester |
| **Duration** | January 2026 — April 2026 |
| **Problem Domain** | Border Defence and Surveillance (GTU) |

</div>

This project fulfills all five GTU domain requirements:

- ✅ EDA on surveillance and sensor datasets
- ✅ Anomaly detection model to identify unusual activities
- ✅ ML/DL classification of objects and movement patterns
- ✅ Alert prioritization system to reduce false positives
- ✅ Cloud-based data integration using Microsoft Azure

---

## 📜 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d47a1,50:1a1f35,100:0d1117&height=120&section=footer" width="100%"/>

**⭐ If you find this project useful, please star the repository!**

*Built with ❤️ by Jainil Gupta & Team | Microsoft Elevate Internship 2026*

</div>