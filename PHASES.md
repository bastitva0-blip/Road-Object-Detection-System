# 🚗 Road Object Detection System - Development Phases

## Overview
This document outlines all 8 development phases with clear goals, technical tasks, dependencies, and deployment strategies.

---

## Phase 1: Webcam Feed + Basic Vehicle & Pedestrian Detection
**Timeline**: Week 1-2  
**Status**: ✅ CODE COMPLETE (pending physical hardware validation)

### Goals
- [x] Real-time video capture from USB webcam
- [x] Multi-class object detection (vehicles, pedestrians)
- [x] Frame annotation with bounding boxes
- [x] Configuration management system
- [x] Live display with OpenCV
- [x] Performance optimization

### Technical Tasks

#### 1.1 Webcam Integration
- **Status**: ✅ Code Ready
- **Files**: `src/core/video_capture.py`
- **Tasks**:
  - [x] OpenCV VideoCapture wrapper
  - [x] Camera property configuration (resolution, FPS)
  - [x] Auto-focus support
  - [ ] Test with Zebronics Sharp PRO camera (needs physical hardware)
  - [x] Handle camera disconnection gracefully

#### 1.2 YOLOv8 Object Detection
- **Status**: ✅ Code Ready
- **Files**: `src/detection/yolo_detector.py`, `src/detection/object_detector.py`
- **Tasks**:
  - [x] YOLOv8 model loading
  - [x] Detection inference
  - [x] Confidence & IOU filtering
  - [x] Optimize for CPU inference
  - [ ] Benchmark FPS on target hardware (needs physical hardware)
  - [x] Cache model weights

#### 1.3 Frame Processing
- **Status**: ✅ Code Ready
- **Files**: `src/core/frame_processor.py`
- **Tasks**:
  - [x] CLAHE histogram equalization
  - [x] Denoising (fastNlMeansDenoising)
  - [x] Low-light enhancement
  - [x] Benchmarking preprocessing pipeline
  - [x] Profile memory usage

#### 1.4 Main Application Loop
- **Status**: ✅ Code Ready
- **Files**: `src/main.py`
- **Tasks**:
  - [x] Initialize all components
  - [x] Main detection loop
  - [x] Frame annotation with bounding boxes
  - [x] OpenCV display
  - [x] Add keyboard controls (pause, save, record)
  - [x] FPS counter and statistics
  - [x] Error handling and logging

### Dependencies

#### Python Packages
```
opencv-python==4.8.1.78    # Video capture & image processing
ultralytics==8.0.196        # YOLOv8 detection
torch==2.0.1                # Deep learning framework
torchvision==0.15.2         # Computer vision utilities
numpy==1.24.3               # Numerical computing
```

#### Hardware
- USB Webcam (Zebronics Sharp PRO - 2048×1536, 30FPS)
- NVIDIA GPU (optional, for faster inference)
- Minimum 2GB RAM

#### Deployment
- Local machine (desktop/laptop)
- No external services needed yet

### Testing Strategy
```bash
# Test camera detection
python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'FAILED')"

# Test detection on static image
python -c "from src.detection.object_detector import ObjectDetector; import numpy as np; print(ObjectDetector().detect(np.zeros((720,1280,3))))"

# Run full system
python src/main.py
```

### Success Criteria
- ✅ Webcam video displays in real-time
- ✅ Detections appear with bounding boxes
- ✅ Detection accuracy > 70%
- ✅ FPS > 15 on CPU / > 25 on GPU
- ✅ System runs for 1+ hour without crashes

### Performance Targets
| Metric | CPU | GPU |
|--------|-----|-----|
| FPS | 10-15 | 30+ |
| Latency | 60-100ms | 20-30ms |
| Memory | ~2GB | ~4GB |

---

## Phase 2: Lane Detection + Road Markings
**Timeline**: Week 3-4  
**Status**: ✅ CORE COMPLETE (classical CV; LaneNet backend and dataset training deferred)

### Goals
- [x] Detect lane lines (solid, dashed, double)
- [x] Identify road markings (arrows, stop lines, speed bumps)
- [x] Zebra crossing detection
- [ ] Lane occupancy analysis (needs Phase 3 tracking + lane geometry)
- [x] Pothole/waterlogging detection prep (architecture decision documented)

### Technical Tasks

#### 2.1 Lane Line Detection
- **Files**: `src/road_analysis/lane_detector.py`
- **Approach 1: Classical (Hough Transform)**
  - [x] Grayscale conversion
  - [x] Canny edge detection
  - [x] Region of interest (ROI) masking
  - [x] Hough Line Transform
  - [x] Line clustering and fitting
  - [ ] Polynomial curve fitting (linear fit implemented; degree-2 fit for sharp curves deferred)
  
- **Approach 2: Deep Learning (LaneNet)**
  - [ ] Download pretrained LaneNet model
  - [ ] Semantic segmentation inference
  - [ ] Lane pixel extraction
  - [ ] Curve fitting on detected pixels

- **Tasks**:
  - [ ] Implement both approaches (Hough done; LaneNet deferred, not needed to hit FPS target)
  - [x] Benchmark performance (`scripts/benchmark_lane_detection.py`)
  - [x] Choose based on FPS/accuracy tradeoff (Hough chosen, see module docstring)
  - [x] Add lane departure warning logic

#### 2.2 Road Marking Recognition
- **Files**: `src/road_analysis/lane_detector.py`
- **Tasks**:
  - [x] Arrow detection (direction & location)
  - [x] Stop line detection
  - [x] Speed bump detection (morphological analysis)
  - [x] Zebra crossing detection (white stripe pattern)
  - [x] Road edge detection (for rural roads)

#### 2.3 Pothole/Anomaly Detection Prep
- **Files**: `src/road_analysis/pothole_detector.py`
- **Tasks**:
  - [ ] Dataset preparation (Roboflow pothole dataset) (needs external account/network access)
  - [x] Model architecture research (U-Net, SegFormer) — U-Net selected, see Phase 5 notes
  - [ ] Data annotation pipeline (Phase 5 scope)

### Dependencies

#### Python Packages
```
scikit-image==0.21.0        # Image processing utilities
scipy==1.11.2               # Scientific computing
opencv-python-contrib==4.8.1.78  # Advanced OpenCV modules
```

#### Pre-trained Models
- LaneNet (optional): [GitHub - tuSimple/LaneNet](https://github.com/tuSimple/LaneNet)
- Dataset: BDD100K (lane annotations)

#### Deployment
- Local processing (CPU-capable)
- Optional: ONNX export for faster inference

### Testing Strategy
```bash
# Test on sample road images
python scripts/test_lane_detection.py --input data/test_images/

# Benchmark Hough vs LaneNet
python scripts/benchmark_lane_models.py
```

### Success Criteria
- ✅ Lane detection accuracy > 80% on BDD100K
- ✅ FPS impact < 5 (total FPS still > 10)
- ✅ Handles multiple lane types (solid, dashed, double)
- ✅ Road marking detection > 75% accuracy

---

## Phase 3: Multi-Object Tracking + Speed Estimation
**Timeline**: Week 5-6  
**Status**: ✅ CORE COMPLETE (validated on synthetic data; MOTA/ground-truth accuracy needs real dataset)

### Goals
- [x] Persistent object ID assignment across frames
- [x] Trajectory visualization (object trails)
- [x] Vehicle speed estimation
- [x] Object dwell time tracking
- [x] Re-identification after occlusion

### Technical Tasks

#### 3.1 Multi-Object Tracking (MOT)
- **Files**: `src/tracking/tracker.py`
- **Algorithm: DeepSORT**
  - [ ] Feature extraction (deep learning backbone) — not used, see decision below
  - [ ] Kalman filter for motion prediction — implemented under ByteTrack instead
  - [ ] Hungarian algorithm for assignment — implemented under ByteTrack instead
  - [ ] Track management (birth, death, re-id) — implemented under ByteTrack instead

- **Algorithm: ByteTrack (Alternative)** — chosen (see module docstring for rationale)
  - [x] High-confidence & low-confidence matching
  - [x] Faster inference than DeepSORT (no appearance embedder, CPU-only)
  - [x] Better handling of occlusions

- **Tasks**:
  - [x] Choose between DeepSORT/ByteTrack — ByteTrack: no extra model download, runs on numpy/scipy already in requirements.txt
  - [x] Integrate with Phase 1 detections
  - [x] Assign unique track IDs
  - [x] Handle 30+ objects simultaneously (tested with 35-40, no ID switches, ~140 FPS tracker-only)
  - [x] Test occlusion handling (`tests/test_tracking.py::test_no_id_switch_across_brief_occlusion`)

#### 3.2 Speed Estimation
- **Files**: `src/analytics/speed_estimator.py`
- **Approach: Optical Flow + Calibration** — implemented
  - [x] Calculate frame-to-frame motion (optical flow)
  - [x] Convert pixels to meters using calibration
  - [ ] Account for perspective distortion (documented limitation: scene-average pixels_per_meter, no homography correction)
  - [x] Implement camera calibration script (`scripts/calibrate_camera.py`)
  
- **Approach: 3D Bounding Box** (not implemented — optical flow met the phase goal without a depth/3D model)
  - [ ] Estimate object dimensions
  - [ ] Calculate 3D position in scene
  - [ ] Track 3D center across frames
  - [ ] Derive velocity from 3D trajectory

- **Tasks**:
  - [x] Implement optical flow method first
  - [x] Create camera calibration utility
  - [ ] Validate against ground truth (needs a real annotated video + radar/GPS reference; verified formula correctness on synthetic known-velocity input instead, see `tests/test_tracking.py::test_known_pixel_velocity_converts_correctly`)
  - [ ] Accuracy target: ±10% error (untestable without ground truth)

#### 3.3 Trajectory & Analytics
- **Files**: `src/analytics/zone_logic.py`, `src/tracking/tracker.py`
- **Tasks**:
  - [x] Store object position history (`Track.trajectory`, capped rolling window)
  - [x] Calculate dwell time in zones (`ZoneLogic.get_dwell_time`)
  - [x] Trajectory visualization (draw trails) (`main.py::_draw_tracks`)
  - [x] Predict future positions (`Track.predict_future_position`, linear extrapolation from Kalman velocity)

### Dependencies

#### Python Packages
```
scipy==1.11.2               # Hungarian algorithm (linear_sum_assignment)
numpy==1.24.3                # Kalman filter math
opencv-python==4.8.1.78     # optical flow (calcOpticalFlowPyrLK), calibration UI
```
No new dependencies needed — `deep-sort-realtime` removed from requirements.txt since
ByteTrack (custom Kalman filter + IOU, no appearance embedder) was chosen instead.

#### External Tools
- Camera calibration tool (`scripts/calibrate_camera.py`, built on OpenCV)
- Dataset: UA-DETRAC or CityFlow for validation (not fetched — network/account required)

### Testing Strategy
```bash
# Test tracking on video
python scripts/test_tracking.py --video data/sample_video.mp4

# Validate speed estimation
python scripts/validate_speed_estimation.py --ground_truth data/speed_gt.csv

# Benchmark MOT metrics
python scripts/evaluate_mot.py --dataset ua_detrac
```

### Success Criteria
- ✅ MOTA (Multiple Object Tracking Accuracy) > 70%
- ✅ Track ID switches < 10% of total tracks
- ✅ Speed estimation MAE < 2 km/h
- ✅ FPS impact < 5 (total FPS still > 10)

---

## Phase 4: Traffic Sign Recognition + Red Light Violations
**Timeline**: Week 7-8  
**Status**: 🚧 SCAFFOLDING READY

### Goals
- [ ] Detect traffic signs (stop, yield, speed limit, etc.)
- [ ] Recognize sign types with confidence
- [ ] OCR for speed limit numbers
- [ ] License plate detection & reading
- [ ] Red light violation detection
- [ ] Jaywalking detection

### Technical Tasks

#### 4.1 Traffic Sign Recognition
- **Files**: `src/road_analysis/traffic_sign_ocr.py`
- **Approach: YOLOv8 for Detection**
  - [ ] Fine-tune YOLOv8 on traffic sign dataset
  - [ ] Detect: stop, yield, speed limit, no-entry, one-way
  - [ ] Confidence thresholding
  
- **Dataset Preparation**:
  - [ ] Roboflow traffic sign dataset
  - [ ] OIDDS (Open Images Dataset)
  - [ ] Custom Indian sign dataset (if available)

- **Tasks**:
  - [ ] Collect/prepare training dataset
  - [ ] Fine-tune model (2-3 epochs)
  - [ ] Validate on test set
  - [ ] Accuracy target: > 85%

#### 4.2 OCR for Speed Limits & License Plates
- **Files**: `src/road_analysis/traffic_sign_ocr.py`
- **Tools**:
  - Tesseract OCR (open-source)
  - EasyOCR (PyTorch-based, better accuracy)
  - Paddle OCR (faster inference)

- **Tasks**:
  - [ ] Crop sign region from detection
  - [ ] Preprocess image (contrast, threshold)
  - [ ] Run OCR inference
  - [ ] Post-process results (filter invalid numbers)
  - [ ] License plate OCR (angle correction, character recognition)
  - [ ] Store detected plates in event log

#### 4.3 Red Light Violation Detection
- **Files**: `src/analytics/zone_logic.py`
- **Tasks**:
  - [ ] Detect traffic light state (Red, Yellow, Green)
  - [ ] Define stop line zone
  - [ ] Check if vehicle crosses stop line on Red
  - [ ] Log violation with timestamp & vehicle ID
  - [ ] Alert generation

#### 4.4 Jaywalking Detection
- **Files**: `src/analytics/zone_logic.py`
- **Tasks**:
  - [ ] Define zebra crossing zone
  - [ ] Detect pedestrians crossing outside zone
  - [ ] Check if crossing against signal
  - [ ] Log jaywalking event
  - [ ] Alert generation

### Dependencies

#### Python Packages
```
pytesseract==0.3.10         # Tesseract wrapper
easyocr==1.6.2              # Deep learning OCR
paddleocr==2.7.0.2          # Faster OCR alternative
pillow==10.0.0              # Image manipulation
```

#### External Services/Tools
```
Tesseract OCR (system package)
  - Linux: sudo apt install tesseract-ocr
  - macOS: brew install tesseract
  - Windows: Download from GitHub
```

#### Pre-trained Models
- YOLOv8 fine-tuned on traffic signs (from Phase 4 training)
- EasyOCR pretrained models (auto-downloaded)

#### Datasets
- Roboflow Traffic Signs: [roboflow.com/datasets](https://roboflow.com/datasets)
- OIDDS Traffic Signs subset
- IDD (India Driving Dataset) - Indian signs

### Testing Strategy
```bash
# Test sign detection
python scripts/test_sign_detection.py --input data/test_images/

# Test OCR accuracy
python scripts/validate_ocr.py --dataset data/ocr_test_set/

# Test violation detection
python scripts/test_violation_detection.py --video data/traffic_test.mp4
```

### Success Criteria
- ✅ Sign detection accuracy > 85%
- ✅ OCR accuracy > 90% for numbers
- ✅ License plate reading > 80% accuracy
- ✅ Red light violation detection 100% precision (no false positives)
- ✅ FPS impact < 5

---

## Phase 5: Pothole & Road Damage Detection
**Timeline**: Week 9-10  
**Status**: 🚧 SCAFFOLDING READY

### Goals
- [ ] Detect potholes in real-time
- [ ] Identify debris on road
- [ ] Waterlogging detection
- [ ] Speed bump detection
- [ ] Severity classification

### Technical Tasks

#### 5.1 Pothole Detection CNN
- **Files**: `src/road_analysis/pothole_detector.py`
- **Model Architecture Options**:
  - U-Net (segmentation-based)
  - YOLOv8 (detection-based)
  - ResNet + FCN (semantic segmentation)
  
- **Tasks**:
  - [ ] Download Roboflow pothole dataset
  - [ ] Data augmentation (rotation, brightness, noise)
  - [ ] Train CNN model (50-100 epochs)
  - [ ] Evaluate on test set
  - [ ] Export to ONNX for faster inference
  - [ ] Implement real-time inference

#### 5.2 Debris Detection
- **Tasks**:
  - [ ] Use YOLOv8 generic object detection
  - [ ] Define debris classes (trash, stones, branches)
  - [ ] Fine-tune on custom debris dataset
  - [ ] Confidence filtering

#### 5.3 Waterlogging Detection
- **Tasks**:
  - [ ] Image reflectivity analysis (flooded areas are reflective)
  - [ ] Color-based detection (water typically blue/gray)
  - [ ] Morphological operations for blob detection
  - [ ] Seasonal/weather context awareness

#### 5.4 Severity Classification
- **Tasks**:
  - [ ] Classify pothole severity: low, medium, high
  - [ ] Based on size, shape, and darkness
  - [ ] Assign priority for road maintenance

### Dependencies

#### Python Packages
```
torch==2.0.1                # Deep learning
torchvision==0.15.2         # Vision models
albumentations==1.3.0       # Data augmentation
segmentation-models-pytorch==0.3.3  # Pretrained models
```

#### Pre-trained Models
- Roboflow Pothole Dataset (labeled)
- U-Net encoder-decoder architecture

#### Training Infrastructure
- GPU recommended (4-8GB VRAM)
- Cloud option: Google Colab, AWS SageMaker, Paperspace

### Testing Strategy
```bash
# Test pothole detection
python scripts/test_pothole_detection.py --video data/pothole_test.mp4

# Validate on Roboflow dataset
python scripts/validate_pothole_model.py --dataset data/pothole_val/

# Benchmark segmentation metrics
python scripts/evaluate_segmentation.py
```

### Success Criteria
- ✅ Pothole detection mAP > 75%
- ✅ False positive rate < 5%
- ✅ Real-time inference (> 10 FPS)
- ✅ Severity classification accuracy > 80%

---

## Phase 6: Analytics - Heatmap, Flow Map, Queue Length
**Timeline**: Week 11-12  
**Status**: 🚧 SCAFFOLDING READY

### Goals
- [ ] Traffic density heatmap
- [ ] Vehicle flow direction map
- [ ] Queue length estimation
- [ ] Peak hour detection
- [ ] Near-miss event detection
- [ ] Accident detection

### Technical Tasks

#### 6.1 Traffic Density Heatmap
- **Files**: `src/analytics/event_detector.py`
- **Tasks**:
  - [ ] Create grid overlay on frame
  - [ ] Count objects in each grid cell
  - [ ] Color-code by density (green=low, red=high)
  - [ ] Smooth heatmap with Gaussian blur
  - [ ] Animate over time

#### 6.2 Vehicle Flow Direction Map
- **Tasks**:
  - [ ] Track vehicle centers across time
  - [ ] Calculate movement vectors
  - [ ] Aggregate vectors by region
  - [ ] Draw flow arrows on overlay
  - [ ] Identify traffic direction anomalies

#### 6.3 Queue Length Estimation
- **Tasks**:
  - [ ] Detect vehicle clusters (consecutive vehicles)
  - [ ] Measure pixel distance along lane
  - [ ] Convert to meters using calibration
  - [ ] Estimate queue length in meters
  - [ ] Track queue growth/shrinkage over time

#### 6.4 Near-Miss & Accident Detection
- **Files**: `src/analytics/event_detector.py`
- **Near-Miss Detection**:
  - [ ] Calculate distance between all object pairs
  - [ ] Threshold for "near-miss" (e.g., < 2 meters)
  - [ ] Log event with involved vehicle IDs
  - [ ] Alert system

- **Accident Detection**:
  - [ ] Detect sudden stops (velocity drop)
  - [ ] Check for overlapping bounding boxes
  - [ ] Combine multiple signals for confidence
  - [ ] Log with timestamp and location

#### 6.5 Peak Hour Detection
- **Tasks**:
  - [ ] Maintain rolling window of vehicle counts (e.g., 30 min)
  - [ ] Detect sharp increases in traffic
  - [ ] Compare to historical averages
  - [ ] Flag peak hours

### Dependencies

#### Python Packages
```
pandas==2.0.3               # Data manipulation
numpy==1.24.3               # Array operations
scipy==1.11.2               # Gaussian blur, filtering
matplotlib==3.7.3           # Visualization
seaborn==0.12.2             # Statistical visualization
```

#### Storage (for historical data)
- CSV files (simple option)
- **PostgreSQL** (recommended for scaling)
  - Store: vehicle counts, queue lengths, events
  - Query: historical trends, peak hours
  
- **Supabase** (PostgreSQL + auth + real-time)
  - Managed PostgreSQL database
  - Authentication built-in
  - Real-time subscriptions
  - Free tier: 500MB storage, good for prototyping

### Testing Strategy
```bash
# Test analytics on recorded video
python scripts/test_analytics.py --video data/test_video.mp4

# Visualize heatmap
python scripts/visualize_heatmap.py --output heatmap.png

# Validate queue length estimation
python scripts/validate_queue_estimation.py --ground_truth data/queue_gt.csv
```

### Success Criteria
- ✅ Heatmap visualization runs in real-time
- ✅ Queue length MAE < 5 meters
- ✅ Near-miss detection precision > 90%
- ✅ Peak hour detection accuracy > 85%

---

## Phase 7: Alert System + Event Logging
**Timeline**: Week 13-14  
**Status**: 🚧 SCAFFOLDING READY

### Goals
- [ ] Alert dispatch system with severity levels
- [ ] Event logging to CSV/JSON
- [ ] Audio alerts (beeps with different tones)
- [ ] Optional Telegram notifications
- [ ] Alert cooldown to prevent spam

### Technical Tasks

#### 7.1 Alert Manager
- **Files**: `src/alerts/alert_manager.py`
- **Tasks**:
  - [x] Alert type registry (red light violation, pothole, accident, etc.)
  - [x] Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
  - [x] Cooldown mechanism (prevent duplicate alerts)
  - [ ] Callback registration system
  - [ ] Alert queuing

#### 7.2 Event Logging
- **Files**: `src/alerts/event_logger.py`
- **Tasks**:
  - [x] Log to CSV with headers (timestamp, event_type, details)
  - [x] Log to JSON for structured data
  - [ ] Include metadata (location, involved vehicles, severity)
  - [ ] Create new log file on app restart
  - [ ] Automatic backup of logs

#### 7.3 Audio Alerts
- **Tasks**:
  - [ ] Install audio library (Pygame or pydub)
  - [ ] Define tone frequencies:
    - 440 Hz: Low priority (yellow lines)
    - 800 Hz: Medium priority (pedestrian detected)
    - 1200 Hz: High priority (red light violation)
    - 1600 Hz: Critical (accident, collision)
  - [ ] Generate beep sounds dynamically
  - [ ] Play beep based on alert severity

#### 7.4 Telegram Bot Integration (Optional)
- **Tasks**:
  - [ ] Create Telegram bot via @BotFather
  - [ ] Store bot token in environment variables
  - [ ] Send alerts via Telegram API
  - [ ] Include image with alert (violation screenshot)
  - [ ] Rate limiting to avoid spam

### Dependencies

#### Python Packages
```
pygame==2.2.0               # Audio playback
pydub==0.25.1               # Audio generation
python-telegram-bot==20.1   # Telegram integration
```

#### External Services
- **Telegram** (optional, for notifications)
  - Free tier: unlimited messages
  - Setup: Create bot via @BotFather
  - Configuration: Store token in .env

- **Supabase** (for event storage/backup)
  - Real-time database access
  - REST API for querying
  - Free tier includes 500MB

### Testing Strategy
```bash
# Test alert system
python scripts/test_alerts.py

# Test Telegram integration
python scripts/test_telegram.py --token YOUR_BOT_TOKEN --chat YOUR_CHAT_ID

# Validate event logging
python scripts/validate_event_log.py --log data/logs/events_*.csv
```

### Success Criteria
- ✅ Alert system fires with < 100ms latency
- ✅ Event log complete and accurate
- ✅ Audio beeps audible and distinguishable
- ✅ No duplicate alerts (cooldown working)
- ✅ Telegram integration (if enabled) sends messages reliably

---

## Phase 8: Web Dashboard + Webinar Demo Features
**Timeline**: Week 15-16  
**Status**: 🚧 SCAFFOLDING READY

### Goals
- [ ] Real-time video stream in browser
- [ ] Live statistics panel
- [ ] Detection class toggles
- [ ] Confidence threshold slider
- [ ] Event log viewer
- [ ] Recording controls
- [ ] Webinar demo features (pause & annotate, slow-motion, side-by-side)

### Technical Tasks

#### 8.1 FastAPI Backend
- **Files**: `src/dashboard/backend.py`
- **Endpoints**:
  - [x] `GET /` - Dashboard HTML
  - [x] `GET /api/status` - System status (camera, model, GPU)
  - [x] `GET /api/stats` - Live statistics
  - [ ] `GET /api/events` - Event log paginated
  - [ ] `GET /video_feed` - MJPEG video stream
  - [ ] `WS /ws/video` - WebSocket for live video
  - [ ] `POST /api/config/update` - Update settings
  - [ ] `POST /api/recording/start` - Start recording
  - [ ] `POST /api/recording/stop` - Stop recording

- **Tasks**:
  - [ ] Implement all endpoints
  - [ ] Add CORS for cross-origin requests
  - [ ] Request validation with Pydantic
  - [ ] Error handling with proper HTTP status codes

#### 8.2 Frontend Dashboard
- **Files**: `src/dashboard/static/index.html`, `styles.css`, `app.js`
- **Components**:
  - [x] HTML structure (header, video, stats, controls, alerts)
  - [x] CSS styling (dark theme, responsive grid)
  - [x] JavaScript for interactivity (template)
  - [ ] Real-time stats updates (fetch every 2s)
  - [ ] WebSocket connection for video
  - [ ] Control panel:
    - [ ] Recording toggle (start/stop)
    - [ ] Heatmap toggle
    - [ ] Confidence slider (0-1 range)
    - [ ] Detection class filters (checkboxes)
  - [ ] Event log table with pagination
  - [ ] Download log button

#### 8.3 Video Streaming
- **MJPEG Streaming**:
  - [ ] Encode frames to JPEG
  - [ ] Stream with multipart/x-mixed-replace
  - [ ] Throttle to target FPS (e.g., 15 FPS for web)

- **WebSocket Streaming** (alternative):
  - [ ] Convert frames to base64
  - [ ] Send over WebSocket every N frames
  - [ ] Lower latency than MJPEG

#### 8.4 Webinar Demo Features
- **Pause & Annotate**:
  - [ ] Pause video on user click
  - [ ] Allow drawing on paused frame
  - [ ] Resume/step through frames
  
- **Slow-Motion Playback**:
  - [ ] Replay recent detections at 0.25x speed
  - [ ] Store frame buffer (last 10 seconds)
  - [ ] Allow scrubbing through timeline

- **Side-by-Side View**:
  - [ ] Show raw feed vs annotated feed
  - [ ] Slider to blend between them
  - [ ] Highlight specific detections

#### 8.5 Deployment Options

##### Option 1: Local Network (Simple)
- Run FastAPI on local machine
- Access via `http://192.168.x.x:8000`
- Perfect for webinar demo on campus network

##### Option 2: Cloud Deployment (Recommended)
- **Vercel** (frontend only - static dashboard)
  - Deploy HTML/CSS/JS files
  - $0/month (free tier)
  - `vercel deploy`

- **Heroku** (FastAPI backend)
  - Deploy Python app
  - $7/month (paid, free tier phased out)
  - `heroku login` → `git push heroku main`
  - Alternative: Railway.app ($5/month credit/month)

- **AWS** (scalable option)
  - EC2 (compute) + S3 (storage) + RDS (database)
  - On-demand pricing, often expensive
  - Good for production with high traffic

- **DigitalOcean** (recommended for balance)
  - $5-12/month droplet
  - App Platform for easy deployment
  - PostgreSQL database included
  - SSH access for full control

- **Docker + Cloud Run** (Google Cloud)
  - Containerized FastAPI
  - $0.00002/second (minimal for low traffic)
  - Auto-scaling
  - Simple: `gcloud run deploy`

### Dependencies

#### Python Packages
```
fastapi==0.104.1            # Web framework
uvicorn==0.24.0             # ASGI server
pydantic==2.4.2             # Data validation
python-multipart==0.0.6     # Form data parsing
```

#### Frontend Libraries (CDN)
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.0.0/dist/chart.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
```

#### Deployment Platforms
- **Vercel**: Static file hosting (free)
- **Heroku/Railway**: Backend deployment ($5-7/month)
- **DigitalOcean**: VPS ($5-12/month)
- **AWS/Google Cloud**: Pay-as-you-go

#### Database (Optional)
- **Supabase** (PostgreSQL + Auth)
  - Free tier: 500MB storage
  - Real-time subscriptions
  - $25/month for production
  - Setup: `pip install supabase`

```python
# Example: Store events in Supabase
from supabase import create_client, Client

url = "https://your-project.supabase.co"
key = "your-anon-key"
supabase: Client = create_client(url, key)

# Insert event
data = supabase.table("events").insert({
    "timestamp": datetime.now().isoformat(),
    "event_type": "red_light_violation",
    "vehicle_id": 123,
    "severity": "HIGH"
}).execute()
```

### Testing Strategy
```bash
# Start backend
uvicorn src.dashboard.backend:app --reload --port 8000

# In browser: http://localhost:8000

# Test API endpoints
curl http://localhost:8000/api/status
curl http://localhost:8000/api/stats

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/video
```

### Deployment Checklist

#### Local Network (Webinar Demo)
- [ ] Dashboard accessible on `http://192.168.x.x:8000`
- [ ] Video stream loads
- [ ] Stats update in real-time
- [ ] Controls responsive
- [ ] Event log populated

#### Cloud Deployment
- [ ] Environment variables configured (.env)
- [ ] Database connected (Supabase)
- [ ] Deployment platform chosen
- [ ] Docker image built and tested
- [ ] DNS configured (optional, for domain)
- [ ] HTTPS enabled (important for production)
- [ ] Monitoring and alerting set up

### Success Criteria
- ✅ Dashboard loads in < 2 seconds
- ✅ Video streams with < 1 second latency
- ✅ Stats update every 2 seconds
- ✅ Controls responsive (< 100ms)
- ✅ Supports 50+ concurrent users (if deployed)
- ✅ Event log queryable and downloadable
- ✅ Webinar demo features work smoothly

### Optional Enhancements
- [ ] Historical data visualization (charts)
- [ ] Email digest of daily events
- [ ] Mobile app (React Native)
- [ ] Multi-camera view
- [ ] Map integration (show detection locations)

---

## Cross-Phase Dependencies

### Data Flow
```
Phase 1 (Detection)
    ↓
Phase 2 (Lane Analysis) + Phase 3 (Tracking)
    ↓
Phase 4 (Traffic Intelligence)
    ↓
Phase 5 (Road Anomalies)
    ↓
Phase 6 (Analytics)
    ↓
Phase 7 (Alerts)
    ↓
Phase 8 (Dashboard)
```

### Shared Infrastructure
- **Configuration**: `config/default_config.yaml` (used in all phases)
- **Logging**: `src/alerts/event_logger.py` (Phase 7, used by all phases)
- **Database**: Supabase (Phase 6 onwards)
- **Models**: `data/models/` (shared model storage)

### Dependency Matrix

| Phase | Depends On | Required For |
|-------|-----------|--------------|
| 1 | None | 2,3,4,5,6,7,8 |
| 2 | 1 | 4,6 |
| 3 | 1 | 4,6,7,8 |
| 4 | 1,3 | 6,7,8 |
| 5 | 1 | 6,7,8 |
| 6 | 1,2,3,4,5 | 7,8 |
| 7 | 1,6 | 8 |
| 8 | 1,6,7 | None |

---

## Infrastructure & Deployment Strategy

### Development Environment
```bash
# Local setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/setup_models.py

# Run locally
python src/main.py
uvicorn src.dashboard.backend:app --reload --port 8000
```

### Recommended Tech Stack

#### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Database**: PostgreSQL (Supabase)
- **ORM**: SQLAlchemy (optional)

#### Frontend
- **Static Files**: HTML/CSS/JavaScript
- **Real-time**: WebSocket (built into FastAPI)
- **Charting**: Chart.js (CDN)
- **HTTP Client**: Axios (CDN)

#### Deployment Options

**Option A: Local Network (Webinar)**
```
Laptop with GPU
    ↓
FastAPI (localhost:8000)
    ↓
Browser on same network (http://192.168.x.x:8000)
```

**Option B: Cloud - DigitalOcean (Recommended)**
```
GitHub repo
    ↓ (push)
DigitalOcean App Platform
    ↓
Dockerfile → Container
    ↓
Auto-deployed on git push
    ↓ (HTTPS)
Public URL (my-app.ondigitalocean.app)
    ↓
PostgreSQL Database (managed)
```

**Option C: Cloud - AWS (Enterprise)**
```
GitHub repo
    ↓
AWS CodePipeline
    ↓
ECR (Docker Registry)
    ↓
ECS (Elastic Container Service)
    ↓
Application Load Balancer
    ↓
RDS (PostgreSQL Database)
    ↓
Public URL (myapp.com)
```

### Environment Variables (.env)
```bash
# Core settings
DEVICE=cpu          # or '0' for GPU
MODEL_NAME=yolov8n
CONFIDENCE=0.5

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000

# Database (Supabase)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-anon-key

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your-token
TELEGRAM_CHAT_ID=your-chat-id
```

### Cost Estimation

| Component | Cost | Notes |
|-----------|------|-------|
| Hardware (webcam, laptop) | $3,500 | One-time |
| Cloud Deployment (DigitalOcean) | $120/year | $10/month |
| Database (Supabase free tier) | $0 | 500MB free |
| Telegram Bot | $0 | Free API |
| Total Monthly | ~$10 | Minimal for production |

---

## Timeline & Roadmap

```
Week 1-2   : Phase 1 ✅
Week 3-4   : Phase 2 🚧
Week 5-6   : Phase 3 🚧
Week 7-8   : Phase 4 🚧
Week 9-10  : Phase 5 🚧
Week 11-12 : Phase 6 🚧
Week 13-14 : Phase 7 🚧
Week 15-16 : Phase 8 + Webinar Demo 🚧

Post-Webinar: Enhancements & Scaling
```

---

## Success Metrics

### Accuracy Metrics
- Phase 1: Detection accuracy > 70%
- Phase 2: Lane detection > 80%
- Phase 3: MOTA > 70%, Speed MAE < 2 km/h
- Phase 4: Sign detection > 85%, OCR > 90%
- Phase 5: Pothole mAP > 75%
- Phase 6: Queue length MAE < 5m
- Phase 7: Alert precision > 95%
- Phase 8: Dashboard latency < 1s

### Performance Metrics
- Overall system FPS: 15+ (CPU), 30+ (GPU)
- Memory usage: < 2GB (CPU), < 4GB (GPU)
- Dashboard response: < 100ms
- Event logging: < 10ms per event

### Deployment Metrics
- Dashboard availability: > 99%
- API uptime: > 99%
- Database response time: < 100ms

---

## References & Resources

### Documentation
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Supabase Docs](https://supabase.com/docs)

### Libraries & Tools
- [DeepSORT PyTorch](https://github.com/ZQQ1997/deepsort_pytorch)
- [LaneNet](https://github.com/tuSimple/LaneNet)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [MediaPipe](https://mediapipe.dev/)

### Datasets
- [BDD100K](https://bdd-data.berkeley.edu/) - Driving scenes
- [Cityscapes](https://www.cityscapes-dataset.com/) - Urban segmentation
- [IDD](https://idd.insaan.iiit.ac.in/) - India Driving Dataset
- [Roboflow](https://roboflow.com/) - Annotated datasets

### Cloud Platforms
- [Supabase](https://supabase.com/) - PostgreSQL + Auth + Real-time
- [DigitalOcean](https://www.digitalocean.com/) - App Platform
- [Vercel](https://vercel.com/) - Frontend deployment
- [Railway.app](https://railway.app/) - Backend deployment

---

**Last Updated**: 2026-09-14  
**Version**: 1.0  
**Maintained By**: Robotics Club
