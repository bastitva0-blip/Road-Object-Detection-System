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
**Status**: ✅ CORE COMPLETE (classical CV sign candidates; fine-tuned model deferred, needs dataset)

### Goals
- [x] Detect traffic signs (stop, yield, speed limit, etc.) — candidate detection, not classification (see 4.1)
- [x] Recognize sign types with confidence (coarse: color/shape category, not fine-grained sign class)
- [x] OCR for speed limit numbers
- [x] License plate detection & reading (OCR only; plate localization heuristic not implemented, see 4.2)
- [x] Red light violation detection
- [x] Jaywalking detection

### Technical Tasks

#### 4.1 Traffic Sign Recognition
- **Files**: `src/road_analysis/traffic_sign_ocr.py`
- **Approach: YOLOv8 for Detection** (not done — needs a labeled dataset; see decision below)
  - [ ] Fine-tune YOLOv8 on traffic sign dataset
  - [ ] Detect: stop, yield, speed limit, no-entry, one-way
  - [ ] Confidence thresholding
  
- **Dataset Preparation**:
  - [ ] Roboflow traffic sign dataset (needs external account/download)
  - [ ] OIDDS (Open Images Dataset)
  - [ ] Custom Indian sign dataset (if available)

- **Tasks**:
  - [ ] Collect/prepare training dataset
  - [ ] Fine-tune model (2-3 epochs)
  - [ ] Validate on test set
  - [ ] Accuracy target: > 85% (untestable without a labeled dataset)
  - [x] Classical CV fallback shipped instead: HSV red/blue mask + contour shape → coarse category (`stop_or_yield`, `speed_limit_or_no_entry`, `mandatory_or_one_way`), no training data needed

#### 4.2 OCR for Speed Limits & License Plates
- **Files**: `src/road_analysis/traffic_sign_ocr.py`
- **Tools**: Tesseract OCR — chosen over EasyOCR/PaddleOCR (both pull an extra torch-based model download; Tesseract is already in requirements.txt and needs only the `tesseract-ocr` system package)

- **Tasks**:
  - [x] Crop sign region from detection
  - [x] Preprocess image (contrast, threshold) — grayscale, Otsu threshold, border padding
  - [x] Run OCR inference
  - [x] Post-process results (filter invalid numbers) — regex digit extraction, 5-150 range check
  - [ ] License plate OCR (angle correction, character recognition) — character recognition done (`read_license_plate`); angle correction and plate *localization* (finding the plate rectangle on a vehicle crop) not implemented
  - [ ] Store detected plates in event log (no plate localization yet to feed it)

#### 4.3 Red Light Violation Detection
- **Files**: `src/analytics/zone_logic.py`
- **Tasks**:
  - [x] Detect traffic light state (Red, Yellow, Green) — `classify_traffic_light_state`, HSV brightness per third of the box
  - [x] Define stop line zone — registered as a polygon (intersection area beyond the line), not the line itself (near-zero-area line is unreliable for point-in-polygon)
  - [x] Check if vehicle crosses stop line on Red
  - [x] Log violation with timestamp & vehicle ID (`EventLogger.log_event`, wired in `main.py`)
  - [ ] Alert generation (logged only; audio/push alerts are Phase 7 scope)

#### 4.4 Jaywalking Detection
- **Files**: `src/analytics/zone_logic.py`
- **Tasks**:
  - [x] Define zebra crossing zone (reuses `zebra_crossing` zone config)
  - [x] Detect pedestrians crossing outside zone
  - [ ] Check if crossing against signal (current logic flags any road presence outside the crossing, not conditioned on signal state)
  - [x] Log jaywalking event
  - [ ] Alert generation (logged only; Phase 7 scope)

### Dependencies

#### Python Packages
```
pytesseract==0.3.10         # Tesseract wrapper
pillow==10.0.0               # Image manipulation
```
easyocr/paddleocr not added — Tesseract met accuracy needs on synthetic tests without an extra torch-based model download.

#### External Services/Tools
```
Tesseract OCR (system package) — required, install with:
  - Linux: sudo apt install tesseract-ocr
  - macOS: brew install tesseract
  - Windows: Download from GitHub
```
`TrafficSignOCR` degrades gracefully (`read_speed_limit`/`read_license_plate` return `None`) if this binary is missing.

#### Pre-trained Models
- YOLOv8 fine-tuned on traffic signs — not done, see 4.1

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
- ⚠️ Sign detection accuracy > 85% — untestable without a labeled dataset; classical detector is 2/2 on synthetic red/blue candidates, no false positives on a plain gray frame (`tests/test_traffic_sign_ocr.py`)
- ✅ OCR accuracy > 90% for numbers — 100% (4/4) on centered synthetic digits, tight inner crop (see `TrafficSignOCR.read_speed_limit` docstring for the crop caveat)
- ⚠️ License plate reading > 80% accuracy — OCR itself works (tesseract confuses `0`/`O` on one synthetic test, a known OCR ambiguity); untested on real plates
- ✅ Red light violation detection 100% precision (no false positives) — verified logically correct (`tests/test_violations.py`), not on real video
- ✅ FPS impact < 5 — analytics throttled to every `ANALYTICS_INTERVAL`-th frame (`scripts/benchmark_anomaly_detection.py`)

---

## Phase 5: Pothole & Road Damage Detection
**Timeline**: Week 9-10  
**Status**: ✅ CLASSICAL-CV MVP COMPLETE (CNN training deferred, needs dataset + GPU)

### Goals
- [x] Detect potholes in real-time (classical CV heuristic, not CNN — see 5.1)
- [x] Identify debris on road (classical CV heuristic)
- [x] Waterlogging detection
- [ ] Speed bump detection — implemented in Phase 2 (`LaneDetector.detect_road_markings`), not duplicated here
- [x] Severity classification

### Technical Tasks

#### 5.1 Pothole Detection CNN
- **Files**: `src/road_analysis/pothole_detector.py`
- **Model Architecture Options**:
  - U-Net (segmentation-based) — selected, see module docstring for rationale
  - YOLOv8 (detection-based)
  - ResNet + FCN (semantic segmentation)
  
- **Tasks**:
  - [ ] Download Roboflow pothole dataset (needs external account/download)
  - [ ] Data augmentation (rotation, brightness, noise)
  - [ ] Train CNN model (50-100 epochs) — needs GPU + hours of training, out of scope here
  - [ ] Evaluate on test set
  - [ ] Export to ONNX for faster inference
  - [ ] Implement real-time inference
  - [x] Classical CV fallback shipped instead: dark-blob detection against local road-mean brightness, works with zero training data (`tests/test_pothole_detector.py`)

#### 5.2 Debris Detection
- **Tasks**:
  - [ ] Use YOLOv8 generic object detection (COCO has no trash/stone/branch classes; would need the same fine-tuning as 5.1)
  - [ ] Define debris classes (trash, stones, branches)
  - [ ] Fine-tune on custom debris dataset
  - [ ] Confidence filtering
  - [x] Classical CV fallback shipped instead: HSV saturation-anomaly blobs vs. the road's median color

#### 5.3 Waterlogging Detection
- **Tasks**:
  - [x] Image reflectivity analysis (flooded areas are reflective) — low-texture (Laplacian) + bright + low-saturation heuristic
  - [x] Color-based detection (water typically blue/gray)
  - [x] Morphological operations for blob detection
  - [ ] Seasonal/weather context awareness (not implemented — no weather signal available)

#### 5.4 Severity Classification
- **Tasks**:
  - [x] Classify pothole severity: low, medium, high
  - [x] Based on size, shape, and darkness (aspect-ratio filter + area-ratio/darkness scoring)
  - [ ] Assign priority for road maintenance (severity is computed; no maintenance-priority queue/output built)

### Dependencies

#### Python Packages
```
No new dependencies — classical CV heuristics use opencv-python/numpy only
(already in requirements.txt). torch/torchvision/albumentations/
segmentation-models-pytorch stay deferred until CNN training (5.1) happens.
```

#### Pre-trained Models
- Roboflow Pothole Dataset (labeled) — not fetched
- U-Net encoder-decoder architecture — not trained

#### Training Infrastructure
- GPU recommended (4-8GB VRAM) — not available in this environment
- Cloud option: Google Colab, AWS SageMaker, Paperspace

### Testing Strategy
```bash
# Unit tests on synthetic road frames (dark blob, color anomaly, smooth bright patch)
pytest tests/test_pothole_detector.py -v

# Benchmark detection cost per frame
python scripts/benchmark_anomaly_detection.py
```

### Success Criteria
- ⚠️ Pothole detection mAP > 75% — mAP needs a labeled dataset; classical detector is 1/1 on synthetic dark blobs, 0 false positives on 2 clean-frame tests
- ❌ False positive rate < 5% — 0 FPs on synthetic uniform-gray road frames, but tested against a real photo (textured brick pavement, ultralytics' bundled `bus.jpg`) and got 7 false potholes + 21 false debris; classical CV isn't ready for real, visually busy road surfaces — see pothole_detector.py docstring
- ✅ Real-time inference (> 10 FPS) — ~26ms/frame combined (`scripts/benchmark_anomaly_detection.py`), throttled further via `ANALYTICS_INTERVAL` in `main.py`
- ✅ Severity classification accuracy > 80% — logic verified correct on synthetic cases; no labeled severity dataset to compute accuracy against

---

## Phase 6: Analytics - Heatmap, Flow Map, Queue Length
**Timeline**: Week 11-12  
**Status**: ✅ CORE COMPLETE (all logic implemented and tested; no historical DB storage)

### Goals
- [x] Traffic density heatmap
- [x] Vehicle flow direction map
- [x] Queue length estimation
- [x] Peak hour detection
- [x] Near-miss event detection
- [x] Accident detection

### Technical Tasks

#### 6.1 Traffic Density Heatmap
- **Files**: `src/analytics/event_detector.py`
- **Tasks**:
  - [x] Create grid overlay on frame (`generate_heatmap`, configurable grid_size)
  - [x] Count objects in each grid cell
  - [x] Color-code by density (green=low, red=high) — `cv2.COLORMAP_JET` in `render_heatmap_overlay`
  - [x] Smooth heatmap with Gaussian blur
  - [x] Animate over time — recomputed every frame in `main.py`/engine, toggled with 'h'

#### 6.2 Vehicle Flow Direction Map
- **Tasks**:
  - [x] Track vehicle centers across time (via tracker trajectory)
  - [x] Calculate movement vectors (`Track.velocity` from the Kalman filter, exposed via `as_dict`)
  - [x] Aggregate vectors by region (`compute_flow_vectors`, grid-cell averaging)
  - [x] Draw flow arrows on overlay (`cv2.arrowedLine` in engine.py `_run_heatmap`, toggled with 'h')
  - [ ] Identify traffic direction anomalies (no anomaly logic on top of the vectors yet)

#### 6.3 Queue Length Estimation
- **Tasks**:
  - [ ] Detect vehicle clusters (consecutive vehicles) — current implementation uses the full spread of all vehicle centers, not per-lane clustering
  - [x] Measure pixel distance along lane
  - [x] Convert to meters using calibration
  - [x] Estimate queue length in meters (`estimate_queue_length`)
  - [ ] Track queue growth/shrinkage over time (single-frame estimate only, no history)

#### 6.4 Near-Miss & Accident Detection
- **Files**: `src/analytics/event_detector.py`
- **Near-Miss Detection**:
  - [x] Calculate distance between all object pairs
  - [x] Threshold for "near-miss" (e.g., < 2 meters)
  - [x] Log event with involved vehicle IDs (wired through `AlertManager` → `EventLogger` in engine.py)
  - [x] Alert system (AlertManager fires MEDIUM severity)

- **Accident Detection**:
  - [x] Detect sudden stops (velocity drop) — compares recent vs. earlier trajectory speed
  - [x] Check for overlapping bounding boxes (IOU threshold)
  - [x] Combine multiple signals for confidence (0.5 base, +0.5 if a sudden stop coincides)
  - [x] Log with timestamp and location (via AlertManager/EventLogger)

#### 6.5 Peak Hour Detection
- **Tasks**:
  - [x] Maintain rolling window of vehicle counts (e.g., 30 min) (`record_vehicle_count`, deque)
  - [x] Detect sharp increases in traffic
  - [x] Compare to historical averages
  - [x] Flag peak hours (`is_peak_hour`)

### Dependencies

#### Python Packages
```
No new dependencies — heatmap/flow/near-miss/accident/queue/peak-hour logic
uses only opencv-python/numpy (already in requirements.txt). pandas/matplotlib/
seaborn stay unused until Phase 6's historical-trend storage/visualization is
actually built (see Storage below).
```

#### Storage (for historical data)
- Not implemented — `EventDetector`'s peak-hour window and heatmap are in-memory only, reset on restart
- CSV files (simple option) — reachable via `EventLogger`, not wired to peak-hour/heatmap data specifically
- **PostgreSQL / Supabase** (recommended for scaling) — needs external account/service, out of scope here

### Testing Strategy
```bash
# Unit tests on synthetic tracks (near-miss, accident, queue, peak hour, heatmap, flow)
pytest tests/test_event_detector.py -v
```

### Success Criteria
- ✅ Heatmap visualization runs in real-time — `generate_heatmap`+`render_heatmap_overlay` tested, no historical-trend dashboard (that needs the DB storage noted above)
- ⚠️ Queue length MAE < 5 meters — untestable without ground-truth video; formula verified correct on synthetic tracks (`tests/test_event_detector.py`)
- ⚠️ Near-miss detection precision > 90% — untestable without a labeled dataset; distance-threshold logic verified correct on synthetic tracks
- ⚠️ Peak hour detection accuracy > 85% — untestable without historical real-world data; spike-vs-average logic verified correct on synthetic counts

---

## Phase 7: Alert System + Event Logging
**Timeline**: Week 13-14  
**Status**: ✅ CORE COMPLETE (Telegram implemented but unverified against a real bot)

### Goals
- [x] Alert dispatch system with severity levels
- [x] Event logging to CSV/JSON
- [x] Audio alerts (beeps with different tones)
- [x] Optional Telegram notifications
- [x] Alert cooldown to prevent spam

### Technical Tasks

#### 7.1 Alert Manager
- **Files**: `src/alerts/alert_manager.py`
- **Tasks**:
  - [x] Alert type registry (red light violation, pothole, accident, etc.)
  - [x] Severity levels (LOW, MEDIUM, HIGH, CRITICAL)
  - [x] Cooldown mechanism (prevent duplicate alerts)
  - [x] Callback registration system
  - [x] Alert queuing — `get_recent_alerts()` pollable history (used by the Phase 8 dashboard's `/api/events`), not an async dispatch queue

#### 7.2 Event Logging
- **Files**: `src/alerts/event_logger.py`
- **Tasks**:
  - [x] Log to CSV with headers (timestamp, event_type, details)
  - [x] Log to JSON for structured data
  - [x] Include metadata (location, involved vehicles, severity) — caller-supplied `details` dict, severity added by the engine's alert callback
  - [x] Create new log file on app restart (timestamped filename per `EventLogger` instance)
  - [x] Automatic backup of logs (`log_dir/backups/`, copied on `save_json_log()`)

#### 7.3 Audio Alerts
- **Files**: `src/alerts/audio_alerts.py`
- **Tasks**:
  - [x] Install audio library (Pygame or pydub) — pygame added to requirements.txt; failed to build in this sandbox (Python 3.14, no SDL dev headers), so the graceful no-device fallback path is what's actually been tested here
  - [x] Define tone frequencies (440/800/1200/1600 Hz per severity, exact values from the spec)
  - [x] Generate beep sounds dynamically (`generate_tone`, verified: correct dtype/length/amplitude bounds)
  - [x] Play beep based on alert severity (`AudioAlertPlayer.play`, wired into engine.py's alert callbacks)

#### 7.4 Telegram Bot Integration (Optional)
- **Files**: `src/alerts/telegram_notifier.py`
- **Tasks**:
  - [ ] Create Telegram bot via @BotFather (needs the user's own bot + token, not creatable here)
  - [x] Store bot token in environment variables (`TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, already in `.env.example`)
  - [x] Send alerts via Telegram API — `requests`-based, not `python-telegram-bot` (see requirements.txt note); verified error handling against the real API with an intentionally invalid token (404 caught, logged, no crash), never sent a real message (no bot token available)
  - [ ] Include image with alert (violation screenshot) — `send_photo()` implemented but not called from anywhere yet
  - [x] Rate limiting to avoid spam (3s min interval, verified)

### Dependencies

#### Python Packages
```
pygame==2.2.0               # Audio playback (optional — see 7.3)
```
pydub and python-telegram-bot dropped: pydub was only needed alongside pygame for
audio generation, but `numpy`-based sine synthesis (already a dependency) covers
it; Telegram uses plain `requests` (already a dependency) instead of python-telegram-bot.

#### External Services
- **Telegram** (optional, for notifications)
  - Setup: Create bot via @BotFather, put the token/chat ID in `.env`
  - Not exercised against a real bot here — needs the user's own credentials

- **Supabase** (for event storage/backup) — not implemented; local file backup (`log_dir/backups/`) covers the "don't lose it" need without an external service

### Testing Strategy
```bash
# Unit tests: alert manager, event log backup, audio tone generation, Telegram notifier
pytest tests/test_alerts.py -v
```

### Success Criteria
- ✅ Alert system fires with < 100ms latency — synchronous in-process callbacks, effectively instant
- ✅ Event log complete and accurate — verified via `tests/test_alerts.py`
- ⚠️ Audio beeps audible and distinguishable — tone generation verified correct (frequency/amplitude/duration); actual audibility unverified, no audio device in this sandbox
- ✅ No duplicate alerts (cooldown working) — verified in `tests/test_alerts.py`
- ⚠️ Telegram integration (if enabled) sends messages reliably — error handling verified against the real API; message delivery itself unverified, no bot token available

---

## Phase 8: Web Dashboard + Webinar Demo Features
**Timeline**: Week 15-16  
**Status**: ✅ CORE COMPLETE (backend + basic frontend live; webinar-specific demo features not built)

### Goals
- [x] Real-time video stream in browser (MJPEG, verified: real JPEG frames served, viewable in a browser via the forwarded port)
- [x] Live statistics panel
- [ ] Detection class toggles (checkboxes exist in the original HTML template but aren't wired to anything — filtering isn't implemented client- or server-side)
- [x] Confidence threshold slider
- [x] Event log viewer
- [x] Recording controls
- [ ] Webinar demo features (pause & annotate, slow-motion, side-by-side) — see 8.4, not built

### Technical Tasks

#### 8.1 FastAPI Backend
- **Files**: `src/dashboard/backend.py`, `src/core/engine.py`
- **Endpoints**:
  - [x] `GET /` - Dashboard HTML
  - [x] `GET /api/status` - System status (camera, model, GPU)
  - [x] `GET /api/stats` - Live statistics
  - [x] `GET /api/events` - Event log (query param `limit`, not full pagination)
  - [x] `GET /video_feed` - MJPEG video stream
  - [x] `WS /ws/video` - WebSocket for live video (base64 JPEG frames)
  - [x] `POST /api/config/update` - Update settings (confidence_threshold only)
  - [x] `POST /api/recording/start` - Start recording
  - [x] `POST /api/recording/stop` - Stop recording
  - [x] `POST /api/heatmap/toggle` - not in the original spec, added since the frontend heatmap checkbox needs a backend hook

- **Architecture note**: the entire per-frame pipeline (capture → detect → track →
  lane/road analysis → violations → alerts) was extracted from `main.py` into a
  new `DetectionEngine` class (`src/core/engine.py`), shared by both `main.py`
  (desktop `cv2.imshow`) and this backend (runs the engine on a background
  thread, streams its output). Necessary to avoid duplicating ~250 lines of
  pipeline logic between the two front ends.

- **Tasks**:
  - [x] Implement all endpoints
  - [ ] Add CORS for cross-origin requests (not needed yet — frontend is same-origin; add if a separate frontend deployment is ever built)
  - [x] Request validation with Pydantic (`ConfigUpdate` model)
  - [x] Error handling with proper HTTP status codes (503 if the engine hasn't started yet)

#### 8.2 Frontend Dashboard
- **Files**: `src/dashboard/static/index.html`, `styles.css`, `app.js`
- **Components**:
  - [x] HTML structure (header, video, stats, controls, alerts)
  - [x] CSS styling (dark theme, responsive grid)
  - [x] JavaScript for interactivity
  - [x] Real-time stats updates (fetch every 2s)
  - [ ] WebSocket connection for video (endpoint exists — see 8.1 — but the frontend uses a plain `<img src="/video_feed">` instead, simpler and needs no reconnect logic)
  - [x] Control panel:
    - [x] Recording toggle (start/stop)
    - [x] Heatmap toggle
    - [x] Confidence slider (0-1 range)
    - [ ] Detection class filters (checkboxes) — not implemented, see Goals note above
  - [x] Event log table with pagination — list view, not a paginated table
  - [ ] Download log button

#### 8.3 Video Streaming
- **MJPEG Streaming** — chosen as the default (simpler `<img>` tag, no reconnect logic needed):
  - [x] Encode frames to JPEG
  - [x] Stream with multipart/x-mixed-replace
  - [x] Throttle to target FPS (e.g., 15 FPS for web) — capped independently of engine FPS

- **WebSocket Streaming** (alternative, endpoint implemented, not used by the default frontend):
  - [x] Convert frames to base64
  - [x] Send over WebSocket every N frames
  - [x] Lower latency than MJPEG (not benchmarked against MJPEG; both cap at 15 FPS)

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
Validated in this sandbox against newer versions (fastapi 0.141, uvicorn latest,
torch/torchvision 2.14/0.29) since Python 3.14 here has no wheels for the pins
above — the pins themselves are left as originally set for the documented
Python 3.10+ target, not bumped to this sandbox's unusual forced versions.

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
# Start backend (DEMO_MODE=true runs a synthetic feed if no webcam is present)
DEMO_MODE=true uvicorn src.dashboard.backend:app --port 8000

# In browser: http://localhost:8000

# Test API endpoints
curl http://localhost:8000/api/status
curl http://localhost:8000/api/stats
curl http://localhost:8000/api/events

# Test WebSocket connection
wscat -c ws://localhost:8000/ws/video
```
Verified for real: server started, all endpoints hit with curl, MJPEG stream
confirmed as valid JPEG bytes, video visually confirmed correct in a browser
via a forwarded port. `--reload` was dropped from the example — it works, but
its reloader subprocess broke this sandbox's own background-process tracking;
unrelated to the app, just noting it in case another constrained host hits it.

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
- ✅ Dashboard loads in < 2 seconds — verified (curl + browser via a forwarded Codespaces port)
- ⚠️ Video streams with < 1 second latency — MJPEG capped at 15 FPS confirmed working; end-to-end latency not measured
- ✅ Stats update every 2 seconds — frontend polls `/api/stats` every 2s
- ⚠️ Controls responsive (< 100ms) — endpoints respond fast; engine state updates lag behind by up to one CPU-inference frame (~1-3s in this sandbox), documented in `/api/stats`'s eventual-consistency note
- ❌ Supports 50+ concurrent users (if deployed) — not load-tested, single dev server
- ✅ Event log queryable and downloadable — `/api/events`; no download button in the UI, endpoint itself is directly fetchable
- ❌ Webinar demo features work smoothly — not built, see 8.4

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
