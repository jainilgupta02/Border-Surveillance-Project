# BORDER SURVEILLANCE AI - COMPLETE TECHNICAL IMPLEMENTATION GUIDE

## YOUR PROJECT AT A GLANCE

**Timeline:** 20 days  
**Team Size:** You + 4-5 members (flexible)  
**Your Role:** Technical Lead & Core Developer  
**Deployment:** Microsoft Azure (Serverless)  

---

## CRITICAL SUCCESS FACTORS

### ✅ MUST BUILD (Non-Negotiable):
1. **Video Processing** - Frame extraction pipeline
2. **Object Detection** - YOLOv8 integration
3. **Anomaly Detection** - ML model training
4. **Cloud Integration** - Azure Blob + Cosmos DB
5. **Azure Functions** - Serverless deployment

###
1. **Alert System** - Email notifications (simple)
2. **Dashboard** - Power BI or Streamlit (visual)
3. **Testing** - Test dataset + metrics (methodical)
4. **Documentation** - README, guides (writing)
5. **Demo & Presentation** - Video + slides (creative)

---

## DAY-BY-DAY TASK LIST (YOUR PERSONAL ROADMAP)

### WEEK 1: BUILD CORE AI PIPELINE

**Day 1: Setup** (4-6 hours)
- [ ] Run `day1_setup.sh` script
- [ ] Create GitHub repo
- [ ] Set up Python environment
- [ ] Install all dependencies
- [ ] Test YOLOv8 installation
- **Evening:** Assign learning tasks to team

**Day 2: Video Processing** (6-8 hours)
```python
# Your goal: Complete video_processor.py
- [ ] Implement extract_frames()
- [ ] Test with 3 sample videos
- [ ] Handle different video formats
- [ ] Save metadata to JSON
```
**Checkpoint:** Can extract frames at 1 FPS ✅

**Day 3: Object Detection - Part 1** (6-8 hours)
```python
# Your goal: Get YOLOv8 working
- [ ] Load YOLOv8n model
- [ ] Run detection on single image
- [ ] Parse results (bbox, confidence, class)
- [ ] Test on 10-20 images
```
**Checkpoint:** Can detect objects in images ✅

**Day 4: Object Detection - Part 2** (6-8 hours)
```python
# Your goal: Integrate with video pipeline
- [ ] Batch process all frames from video
- [ ] Save detections to JSON
- [ ] Save annotated images
- [ ] Calculate processing speed
```
**Checkpoint:** End-to-end video → detections ✅

**Day 5: Anomaly Detection - Data Prep** (4-6 hours)
```python
# Your goal: Prepare training data
- [ ] Collect 50-100 "normal" detections
- [ ] Extract features: x, y, width, height, time
- [ ] Create CSV dataset
- [ ] Visualize data distribution
```
**Checkpoint:** Have training dataset ready ✅

**Day 6: Anomaly Detection - Model** (6-8 hours)
```python
# Your goal: Train Isolation Forest
- [ ] Train model on normal data
- [ ] Test on anomaly examples
- [ ] Tune contamination parameter
- [ ] Save model to file
```
**Checkpoint:** Can detect anomalies ✅

**Day 7: Integration & Testing** (6-8 hours)
- [ ] Connect all modules
- [ ] Test end-to-end pipeline
- [ ] Fix bugs
- [ ] Optimize performance
- [ ] Document code
**Checkpoint:** Working local prototype ✅

---

### WEEK 2: CLOUD DEPLOYMENT

**Day 8: Azure Setup** (4-6 hours)
- [ ] Create Azure account
- [ ] Create Resource Group
- [ ] Set up Blob Storage
- [ ] Create Cosmos DB
- [ ] Configure Key Vault
**Evening:** Share Azure credentials with team

**Day 9: Blob Storage Integration** (6-8 hours)
```python
# Your goal: Upload/download from cloud
- [ ] Implement cloud_uploader.py
- [ ] Upload test video
- [ ] Download and verify
- [ ] Test with 5-10 videos
```

**Day 10: Cosmos DB Integration** (6-8 hours)
```python
# Your goal: Store detections in database
- [ ] Design schema
- [ ] Implement database.py CRUD
- [ ] Store detection results
- [ ] Query and display results
```

**Day 11: Azure Functions - Development** (6-8 hours)
- [ ] Create Function App project
- [ ] Convert pipeline to function
- [ ] Set up Blob trigger
- [ ] Test locally with emulator

**Day 12: Azure Functions - Deployment** (6-8 hours)
- [ ] Deploy to Azure
- [ ] Configure environment variables
- [ ] Test blob-triggered execution
- [ ] Monitor in Application Insights

**Day 13: Testing & Bug Fixes** (6-8 hours)
- [ ] Upload 10 test videos
- [ ] Verify all process correctly
- [ ] Check Cosmos DB data
- [ ] Fix any errors

**Day 14: Integration Day** (4-6 hours)
- [ ] Review team member PRs
- [ ] Integrate alert system
- [ ] Test complete flow
- [ ] Update documentation

---

### WEEK 3: POLISH & DELIVERY

**Day 15-16: Dashboard Integration**
- [ ] Review dashboard PR from team
- [ ] Test Cosmos DB connection
- [ ] Verify visualizations
- [ ] Deploy dashboard

**Day 17: Testing & Metrics**
- [ ] Run evaluation script
- [ ] Calculate accuracy metrics
- [ ] Document results
- [ ] Create test report

**Day 18: Demo Preparation**
- [ ] Record demo video (with team)
- [ ] Create presentation slides
- [ ] Practice demo flow
- [ ] Prepare Q&A answers

**Day 19: Final Polish**
- [ ] Fix remaining bugs
- [ ] Optimize performance
- [ ] Clean up code
- [ ] Update all documentation

**Day 20: Submission**
- [ ] Final testing
- [ ] GitHub repository cleanup
- [ ] Prepare submission materials
- [ ] Submit project!

---

## CRITICAL CODE SNIPPETS (Copy-Paste Ready)

### 1. Video Frame Extraction
```python
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames from video at specified FPS."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Resize to 640x640
            frame = cv2.resize(frame, (640, 640))
            
            # Save frame
            output_path = Path(output_dir) / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_count
```

### 2. YOLOv8 Detection
```python
from ultralytics import YOLO
import cv2

def detect_objects(image_path, model_path="yolov8n.pt", conf=0.5):
    """Detect objects in image using YOLOv8."""
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detection = {
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            }
            detections.append(detection)
    
    return detections
```

### 3. Anomaly Detection
```python
from sklearn.ensemble import IsolationForest
import numpy as np

def train_anomaly_detector(normal_data):
    """Train Isolation Forest on normal detections."""
    # Extract features: center_x, center_y, width, height
    features = []
    for det in normal_data:
        x1, y1, x2, y2 = det['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        features.append([center_x, center_y, width, height])
    
    X = np.array(features)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    return model

def predict_anomaly(model, detection):
    """Predict if detection is anomalous."""
    x1, y1, x2, y2 = detection['bbox']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    X = np.array([[center_x, center_y, width, height]])
    score = model.decision_function(X)[0]
    
    # Convert to 0-1 scale (lower = more anomalous)
    anomaly_score = 1 / (1 + np.exp(score))
    
    return anomaly_score
```

### 4. Azure Blob Upload
```python
from azure.storage.blob import BlobServiceClient
from pathlib import Path

def upload_to_blob(file_path, connection_string, container_name):
    """Upload file to Azure Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=Path(file_path).name
    )
    
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    
    return blob_client.url
```

### 5. Cosmos DB Insert
```python
from azure.cosmos import CosmosClient
import datetime

def store_detection(detection_data, endpoint, key, database, container):
    """Store detection result in Cosmos DB."""
    client = CosmosClient(endpoint, key)
    database_client = client.get_database_client(database)
    container_client = database_client.get_container_client(container)
    
    document = {
        "id": f"det_{datetime.datetime.now().timestamp()}",
        "timestamp": datetime.datetime.now().isoformat(),
        **detection_data
    }
    
    container_client.create_item(body=document)
```

### 6. Email Alert
```python
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_alert(api_key, from_email, to_email, detection):
    """Send email alert for high-confidence detection."""
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=f"⚠️ Alert: {detection['class']} detected",
        html_content=f"""
        <h2>Surveillance Alert</h2>
        <p><strong>Object:</strong> {detection['class']}</p>
        <p><strong>Confidence:</strong> {detection['confidence']:.2%}</p>
        <p><strong>Time:</strong> {detection['timestamp']}</p>
        <p><strong>Anomaly Score:</strong> {detection['anomaly_score']:.2f}</p>
        """
    )
    
    sg = SendGridAPIClient(api_key)
    response = sg.send(message)
    return response.status_code == 202
```

---

## TROUBLESHOOTING GUIDE

### Problem: YOLOv8 installation fails
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

### Problem: Azure authentication error
**Solution:**
1. Check connection string in .env
2. Verify Key Vault access
3. Use Azure CLI: `az login`

### Problem: Cosmos DB query slow
**Solution:**
1. Add partition key to queries
2. Create index on frequently queried fields
3. Use SELECT with specific fields

### Problem: Function timeout (>5 min)
**Solution:**
1. Process smaller video chunks
2. Use durable functions for long tasks
3. Increase timeout in host.json

---

## TEAM COMMUNICATION TEMPLATE

### Daily Standup (15 min):
**You ask each member:**
1. What did you complete yesterday?
2. What will you work on today?
3. Any blockers?

### Code Review Comments:
**Be constructive:**
✅ "Great work! Consider adding error handling here."  
✅ "This function could be optimized using NumPy."  
❌ "This code is wrong."

### Task Assignment Example:
```
@member1 - Alert System

Task: Implement email notifications using SendGrid

Requirements:
- Read detection results from Cosmos DB
- Filter for anomaly_score > 0.8
- Send email with template (see alert_template.html)
- Log all sent alerts

Deadline: Day 11 (Friday)
Priority: High
Estimated effort: 2-3 days

Resources:
- SendGrid docs: https://docs.sendgrid.com
- Example code: src/alert_manager.py (stub)

Let me know if you need help!
```

---

## SUCCESS METRICS

### Technical Metrics:
- [ ] Detection Accuracy: >70% mAP
- [ ] Processing Speed: >10 FPS
- [ ] Anomaly Detection Rate: >80%
- [ ] System Uptime: >95%
- [ ] Alert Response Time: <30 seconds

### Project Metrics:
- [ ] All modules completed
- [ ] Zero critical bugs
- [ ] 100% code documented
- [ ] All tests passing
- [ ] Demo video recorded
- [ ] Presentation ready

---

## FINAL CHECKLIST (Day 20)

### Code Quality:
- [ ] All functions have docstrings
- [ ] No hardcoded secrets
- [ ] Error handling everywhere
- [ ] Code follows PEP 8
- [ ] All files < 300 lines

### Deployment:
- [ ] Azure Functions working
- [ ] Blob Storage accessible
- [ ] Cosmos DB populated
- [ ] Dashboard live
- [ ] Alerts sending

### Documentation:
- [ ] README complete
- [ ] Architecture diagram
- [ ] Deployment guide
- [ ] API reference
- [ ] Team credits

### Presentation:
- [ ] Demo video (3-5 min)
- [ ] Slides (15-20)
- [ ] Live demo ready
- [ ] Q&A prepared

---

## REMEMBER:

**You are building a PROTOTYPE, not production software.**

✅ Focus on: Working end-to-end pipeline  
✅ Prioritize: Core AI functionality  
✅ Deliver: Demonstrable results  

❌ Don't worry about: Perfect optimization  
❌ Don't waste time on: Edge cases  
❌ Don't build: Unnecessary features  

**Your goal:** Show that AI can detect and alert on suspicious activity.

**That's it!**

---

Good luck! You've got this! 🚀

**Questions?** Check docs/ folder or create GitHub issue.
