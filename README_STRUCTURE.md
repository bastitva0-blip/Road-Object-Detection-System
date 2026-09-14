# Road Object Detection System - Project Structure

## Overview
This is a comprehensive, scalable project structure for a multi-phase road object detection system built with OpenCV, YOLOv8, and AI-powered analytics.

## Directory Structure

```
Road-Object-Detection-System/
│
├── src/                                 # Main source code
│   ├── __init__.py
│   ├── main.py                         # Entry point for the system
│   │
│   ├── core/                           # Core modules (webcam, preprocessing)
│   │   ├── __init__.py
│   │   ├── config.py                   # Configuration management
│   │   ├── video_capture.py            # Webcam capture wrapper
│   │   └── frame_processor.py          # Frame preprocessing (CLAHE, denoise)
│   │
│   ├── detection/                      # Object detection
│   │   ├── __init__.py
│   │   ├── yolo_detector.py            # YOLOv8 wrapper
│   │   └── object_detector.py          # High-level detection interface
│   │
│   ├── tracking/                       # Multi-object tracking (Phase 3)
│   │   ├── __init__.py
│   │   └── tracker.py                  # DeepSORT/ByteTrack wrapper
│   │
│   ├── road_analysis/                  # Road/lane detection (Phase 2+)
│   │   ├── __init__.py
│   │   ├── lane_detector.py            # Hough Transform + LaneNet
│   │   ├── traffic_sign_ocr.py         # Sign recognition + Tesseract OCR
│   │   └── pothole_detector.py         # Pothole/debris detection (Phase 5)
│   │
│   ├── analytics/                      # Scene intelligence (Phase 3-6)
│   │   ├── __init__.py
│   │   ├── speed_estimator.py          # Optical flow speed estimation
│   │   ├── zone_logic.py               # Zone-based events (jaywalking, red light)
│   │   └── event_detector.py           # Near-miss, accident, queue length
│   │
│   ├── alerts/                         # Alerts and logging (Phase 7)
│   │   ├── __init__.py
│   │   ├── alert_manager.py            # Alert orchestration
│   │   └── event_logger.py             # CSV/JSON event logging
│   │
│   └── dashboard/                      # Web dashboard (Phase 8)
│       ├── __init__.py
│       ├── backend.py                  # FastAPI server
│       ├── routes.py                   # API endpoints
│       └── static/
│           ├── index.html              # Main dashboard page
│           ├── styles.css              # Dashboard styling
│           └── app.js                  # Frontend logic
│
├── data/                               # Data management
│   ├── datasets/                       # Downloaded/prepared datasets
│   │   ├── coco/
│   │   ├── bdd100k/
│   │   ├── idd/
│   │   └── custom/                     # Custom training data
│   │
│   ├── models/                         # Trained/pretrained models
│   │   ├── yolov8/
│   │   ├── lane_detector/
│   │   ├── pose_estimator/
│   │   └── pothole_detector/
│   │
│   ├── logs/                           # Event logs (created at runtime)
│   │   └── events_*.csv
│   │
│   ├── recordings/                     # Video recordings (created at runtime)
│   │
│   └── screenshots/                    # Auto-saved screenshots (created at runtime)
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_tracking_eval.ipynb
│   └── 04_demo.ipynb
│
├── tests/                              # Unit and integration tests
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_analytics.py
│   └── test_dashboard.py
│
├── config/                             # Configuration files
│   ├── default_config.yaml             # Main configuration
│   ├── webinar_config.yaml             # Webinar demo settings
│   └── detection_classes.yaml          # Class definitions
│
├── scripts/                            # Utility scripts
│   ├── setup_models.py                 # Download models
│   ├── calibrate_camera.py             # Camera calibration
│   ├── prepare_dataset.py              # Dataset preparation
│   └── demo.py                         # Webinar demo script
│
├── docs/                               # Documentation
│   ├── ARCHITECTURE.md                 # System architecture
│   ├── SETUP_GUIDE.md                  # Installation guide
│   ├── API_REFERENCE.md                # API documentation
│   └── WEBINAR_DEMO.md                 # Webinar demo guide
│
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── Dockerfile                          # Docker container
├── docker-compose.yml                  # Multi-container setup
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore rules
├── README.md                           # Main project README
├── idea.md                             # Project specification
├── README_STRUCTURE.md                 # This file
└── CHANGELOG.md                        # Version history

```

## Development Phases

### Phase 1: Webcam + Basic Detection (Week 1-2)
- ✅ Webcam capture setup
- ✅ Basic vehicle & pedestrian detection
- **Files**: `src/core/*`, `src/detection/*`

### Phase 2: Lane & Road Analysis (Week 3-4)
- Lane line detection (Hough + LaneNet)
- Road marking recognition
- **Files**: `src/road_analysis/lane_detector.py`

### Phase 3: Tracking & Speed Estimation (Week 5-6)
- Multi-object tracking (DeepSORT/ByteTrack)
- Vehicle speed estimation
- **Files**: `src/tracking/*`, `src/analytics/speed_estimator.py`

### Phase 4: Traffic Signs & Violations (Week 7-8)
- Traffic sign recognition
- License plate OCR
- Red light violation detection
- **Files**: `src/road_analysis/traffic_sign_ocr.py`, `src/analytics/zone_logic.py`

### Phase 5: Road Anomalies (Week 9-10)
- Pothole detection
- Debris detection
- Waterlogging detection
- **Files**: `src/road_analysis/pothole_detector.py`

### Phase 6: Analytics & Heatmaps (Week 11-12)
- Traffic density heatmap
- Vehicle flow map
- Queue length estimation
- Near-miss detection
- **Files**: `src/analytics/event_detector.py`

### Phase 7: Alerts & Logging (Week 13-14)
- Audio alerts
- Event logging (CSV/JSON)
- Telegram notifications (optional)
- **Files**: `src/alerts/*`

### Phase 8: Web Dashboard (Week 15-16)
- FastAPI backend
- Live video stream
- Interactive controls
- Demo features
- **Files**: `src/dashboard/*`

## How to Use This Structure

### 1. **Initial Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Download models
python scripts/setup_models.py

# Create environment file
cp .env.example .env
```

### 2. **Run the System**
```bash
# Basic detection
python src/main.py

# With custom config
python src/main.py --config config/webinar_config.yaml

# Demo mode
python src/main.py --demo
```

### 3. **Start Dashboard**
```bash
# Terminal 1: Start main detection
python src/main.py

# Terminal 2: Start FastAPI dashboard
uvicorn src.dashboard.backend:app --reload --port 8000
# Access at http://localhost:8000
```

### 4. **Run Tests**
```bash
pytest tests/ -v
```

### 5. **Jupyter Notebooks**
```bash
jupyter notebook notebooks/
```

## Configuration

All system parameters are configurable via YAML files in `config/`:
- `default_config.yaml` - Main configuration
- `detection_classes.yaml` - Detection class definitions
- `webinar_config.yaml` - Demo-specific settings

**Key Config Parameters:**
- `capture_width`, `capture_height` - Video dimensions
- `model_name` - YOLOv8 model size (nano/small/medium)
- `confidence_threshold` - Detection confidence
- `enable_tracking`, `enable_lane_detection` - Feature toggles

## Data Management

**Logs**: All event logs are saved to `data/logs/` as CSV/JSON
**Recordings**: Video recordings saved to `data/recordings/`
**Screenshots**: Auto-saved on events to `data/screenshots/`
**Models**: Pre-trained models cached in `data/models/`

## Adding New Features

1. **Create new module** in appropriate `src/` subdirectory
2. **Add `__init__.py`** for module imports
3. **Write tests** in `tests/`
4. **Update config** in `config/*.yaml`
5. **Document** in `docs/`

## Performance Optimization

- Use `yolov8n` (nano) for CPU-only systems
- Use `yolov8s` (small) or larger for GPU systems
- Enable frame preprocessing for low-light scenarios
- Configure detection classes to reduce inference load

## Scaling Considerations

This structure supports:
- ✅ Multi-camera setups
- ✅ Multiple detection models
- ✅ Distributed processing
- ✅ Cloud deployment (Docker included)
- ✅ Real-time web streaming

## Notes

- All Phase 1 components are **fully implemented**
- Phases 2-8 have **scaffolding code ready for implementation**
- Use `# Phase N` comments to track implementation progress
- Configuration is **centralized** and **environment-aware**
- Testing infrastructure is **ready to expand**

---

**Last Updated**: 2026-09-14
**Version**: 0.1.0
