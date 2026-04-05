# Border Surveillance AI

AI-powered border monitoring pipeline for video-based threat detection, anomaly analysis, alert generation, cloud logging, and operational visualization.

## Overview

Border surveillance environments generate large volumes of visual data that are difficult to monitor continuously by hand. This project automates that workflow by combining object detection, behavioral anomaly scoring, alert prioritization, optional email notifications, Azure integration, and a Streamlit command dashboard.

The system is designed to process surveillance footage, identify suspicious activity, log operational events, and surface those events in a dashboard that is easier for evaluators, reviewers, and operators to understand.

## Key Features

- YOLO-based object detection for classes such as `person`, `vehicle`, `crowd`, `military_vehicle`, `aircraft`, `ship`, and `suspicious_object`
- Optical-flow-assisted motion analysis during preprocessing
- Isolation Forest based anomaly detection using interpretable frame-level features
- Priority-based alerting with `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW` levels
- Rolling JSON alert log for local monitoring
- Email notification support for high-severity alerts through SendGrid
- Azure Blob Storage support for session result uploads
- Azure Cosmos DB support for alert persistence
- Streamlit dashboard for alert monitoring, session summaries, anomaly trends, and manual notification workflows
- Automated test coverage for preprocessing, detector, anomaly/alert logic, and pipeline orchestration

## System Architecture

```text
Video / Camera Input
        |
        v
Preprocessing
- OpenCV video loading
- Frame resize to 640x640
- Optional optical flow / motion score
        |
        v
Object Detection
- YOLOv8 inference
- Threat-tagged detections
        |
        v
Anomaly Detection
- Baseline collection (first 30 frames by default)
- Feature extraction
- Isolation Forest scoring
        |
        v
Alert Management
- Priority assignment
- Cooldown-based notification control
- JSON alert logging
- Optional SendGrid email
- Optional Azure Cosmos DB write
        |
        v
Outputs
- `data/alerts/alert_log.json`
- `data/results/session_*.json`
- `data/detections/frame_*.jpg` (if enabled)
- Azure Blob session upload
- Streamlit dashboard visualization
```

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Streamlit
- Plotly
- Azure Blob Storage SDK
- Azure Cosmos DB SDK
- SendGrid
- pytest

## Project Structure

```text
Border Surveillance Project/
+-- dashboard/
�   +-- app.py
�   +-- Border Defence and Surveillance AI logo.png
+-- data/
�   +-- alerts/
�   +-- annotations/
�   +-- logs/
�   +-- processed/
�   +-- raw/
�   +-- results/
�   +-- test_videos/
+-- docs/
+-- models/
+-- notebooks/
+-- overview/
+-- runs/
+-- scripts/
+-- src/
+-- tests/
+-- requirements.txt
+-- pyproject.toml
+-- pytest.ini
+-- makefile
```

### Folder Guide

- `src/`: core application modules
- `src/preprocessing.py`: video loading, frame resizing, normalization, optical flow, and frame extraction
- `src/detector.py`: YOLO wrapper and structured frame-level detection results
- `src/anomaly.py`: feature extraction, baseline learning, anomaly scoring, and anomaly summaries
- `src/alert_manager.py`: alert priority assignment, alert logging, cooldown logic, and email notifications
- `src/azure_client.py`: Azure Blob and Cosmos DB integration with graceful fallback
- `src/pipeline.py`: main end-to-end runtime entry point
- `dashboard/app.py`: Streamlit monitoring dashboard
- `scripts/main.py`: minimal detector demo over a test video
- `scripts/pilot.py`: manual integration checker across preprocessing, detection, anomaly, alerts, and output files
- `scripts/`: dataset preparation, preprocessing, conversion, and smoke-test utilities
- `data/raw/`: source datasets
- `data/processed/`: processed training or pipeline-ready data
- `data/alerts/`: rolling alert logs
- `data/results/`: saved pipeline session summaries
- `data/test_videos/`: sample videos used for local execution
- `models/`: trained model artifacts such as YOLO weights and anomaly model pickle
- `tests/`: automated validation for major runtime modules
- `overview/` and `docs/`: implementation guide, architecture diagrams, and presentation materials

## How It Works

### 1. Input and Preprocessing

- The pipeline accepts either a video file or a live camera index
- Frames are read through OpenCV
- Each frame is resized to `640x640`
- Optional optical flow is computed to estimate frame motion intensity
- The preprocessing stage yields structured `frame_item` dictionaries to downstream modules

### 2. Detection

- `BorderDetector` loads the YOLO model once and reuses it for the full session
- Each frame is passed through the detector
- Detections are converted into structured objects containing:
  - class name
  - confidence
  - bounding box
  - normalized spatial features
  - per-class threat tags

### 3. Anomaly Analysis

- `AnomalyDetector` operates in two phases:
  - baseline collection
  - live scoring
- The first `30` frames are used by default to learn a normal baseline
- For each scored frame, the model uses low-dimensional behavioral features such as:
  - detection count
  - class diversity
  - confidence statistics
  - critical-class count
  - object location distribution
  - object size
  - optical-flow motion score
  - suspicious class presence
- Frames are classified into `normal`, `high`, or `critical`

### 4. Alert Generation

- `AlertManager` converts anomaly output into operational alert priorities
- Priority mapping is based on anomaly level and motion score
- High and critical alerts are logged and can trigger email notifications
- Cooldown logic helps reduce repeated notifications for the same threat pattern

### 5. Storage and Monitoring

- Alerts are written to `data/alerts/alert_log.json`
- Session summaries are saved to `data/results/session_*.json`
- If enabled, annotated frames are saved to `data/detections/`
- If Azure is configured:
  - session summaries are uploaded to Azure Blob Storage
  - alerts are written to Azure Cosmos DB
- The Streamlit dashboard reads local output files to visualize operational activity

## Key Modules and Interactions

### `src/pipeline.py`

- Main orchestration layer
- Connects preprocessing, detection, anomaly scoring, and alert management
- Saves session summaries and optionally annotated frames
- Uploads session summaries to Azure when configured

### `src/detector.py`

- Runs YOLO inference
- Converts model output into structured detections
- Tracks frame-level `has_high` and `has_critical` flags

### `src/anomaly.py`

- Learns a baseline from early frames
- Applies Isolation Forest scoring to later frames
- Produces interpretable anomaly reasons and severity levels

### `src/alert_manager.py`

- Assigns alert priority
- Maintains a rolling alert log
- Sends notifications through SendGrid when credentials are available
- Saves alert records to Azure Cosmos DB through the Azure client

### `src/azure_client.py`

- Initializes Azure clients only when credentials are present
- Uploads session result JSON files to Blob Storage
- Writes alert documents to Cosmos DB
- Falls back safely to local-only mode if Azure is unavailable

### `dashboard/app.py`

- Reads local alert and session data
- Refreshes automatically
- Displays summaries, charts, recent alerts, and notification status
- Includes manual email alert functionality using configured credentials

## Data Flow

```text
video / camera
-> preprocessing.extract_frames()
-> detector.BorderDetector.detect()
-> anomaly.AnomalyDetector.score()
-> alert_manager.AlertManager.process()
-> local JSON outputs
-> optional Azure storage
-> dashboard/app.py
```

## Entry Points

- Main pipeline:
  - `python src/pipeline.py --video data/test_videos/dota_aerial_test.mp4`
- Live camera mode:
  - `python src/pipeline.py --camera 0`
- Dashboard:
  - `streamlit run dashboard/app.py`
- Detector demo:
  - `python scripts/main.py`
- Manual integration check:
  - `python scripts/pilot.py data/test_videos/dota_aerial_test.mp4`

## Configuration

The project uses `.env` values for cloud and notification integrations.

### Environment Variables Used

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_COSMOS_ENDPOINT`
- `AZURE_COSMOS_KEY`
- `AZURE_COSMOS_DATABASE`
- `AZURE_COSMOS_CONTAINER`
- `AZURE_STORAGE_CONTAINER_ALERTS`
- `AZURE_STORAGE_CONTAINER_RESULTS`
- `SENDGRID_API_KEY`
- `ALERT_FROM_EMAIL`
- `ALERT_TO_EMAIL`
- `SMTP_USER`
- `SMTP_APP_PASSWORD`
- `DATA_ROOT` for dashboard execution outside the project root

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "Border Surveillance Project"
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create a `.env` File

Use the variables below and fill in your own values:

```env
AZURE_STORAGE_CONNECTION_STRING=
AZURE_COSMOS_ENDPOINT=
AZURE_COSMOS_KEY=
AZURE_COSMOS_DATABASE=SurveillanceDB
AZURE_COSMOS_CONTAINER=Alerts
AZURE_STORAGE_CONTAINER_ALERTS=alert-frames
AZURE_STORAGE_CONTAINER_RESULTS=session-results
SENDGRID_API_KEY=
ALERT_FROM_EMAIL=
ALERT_TO_EMAIL=
SMTP_USER=
SMTP_APP_PASSWORD=
```

### 5. Run the Pipeline

Recommended 5-minute evaluation command:

```bash
python src/pipeline.py --video data/test_videos/dota_aerial_test.mp4 --save-frames
```

Useful optional flags:

```bash
python src/pipeline.py \
  --video data/test_videos/dota_aerial_test.mp4 \
  --frame-skip 3 \
  --save-frames \
  --results-dir data/results \
  --annotated-dir data/detections
```

### 6. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

## Azure Integration

When Azure credentials are configured, the system supports two cloud paths:

- Azure Blob Storage
  - uploads session result JSON files
  - uses containers such as `session-results`
- Azure Cosmos DB
  - stores alert documents for operational access
  - uses the configured database and container values

If Azure credentials are missing or invalid, the local pipeline still runs normally.

## Alert System

The alerting layer is designed for operational triage rather than raw event dumping.

- `CRITICAL`: highest-severity anomalies
- `HIGH`: significant suspicious activity
- `MEDIUM`: lower-severity events, including motion-based escalation
- `LOW`: normal or low-risk logged activity

### Notification Behavior

- All non-normal alerts are written to the rolling JSON log
- `HIGH` and `CRITICAL` alerts can trigger email notifications
- Notification frequency is controlled by a cooldown window
- Manual notification can also be triggered from the dashboard

## Dashboard

The Streamlit dashboard provides a command-center style operational view over pipeline outputs.

### What the User Can See

- recent alerts
- alert priority distribution
- recent session summaries
- anomaly overview
- detection and activity trends
- notification readiness and status indicators

### Primary Dashboard Inputs

- `data/alerts/alert_log.json`
- `data/results/session_*.json`
- optional `data/detections/anomaly_summary.json` when available

If live pipeline data is unavailable, the dashboard falls back to demo data so the interface remains usable.

## Sample Outputs

After a typical run, the repository can contain outputs like:

- `data/alerts/alert_log.json`
- `data/results/session_<source>_<timestamp>.json`
- `data/detections/frame_000001.jpg`
- `data/logs/pipeline.log`
- `runs/detect/...` inference artifacts from model experimentation

### Example Alert Content

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

## Testing

Run the full test suite:

```bash
pytest tests -v
```

Run with coverage:

```bash
pytest tests --cov=src --cov-report=html
```

## Future Improvements

- add stronger model/version tracking for production runs
- expose pipeline controls through an API layer
- add alert thumbnails and direct blob links in dashboard views
- add automated deployment and environment profiles
- expand support for multi-camera ingestion and centralized monitoring

## Author

**Jay Gupta**  
Solo Developer  
Border Surveillance AI Project

## License

This project is distributed under the [MIT License](LICENSE).
