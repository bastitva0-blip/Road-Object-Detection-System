# Road Object Detection System - Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   WEBCAM INPUT (1280×720)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   Frame Preprocessor       │
        │ (Resize, Denoise, CLAHE)   │
        └────────────┬───────────────┘
                     │
        ┌────────────┴─────────────────────────────┐
        ▼                                          ▼
    ┌─────────────────┐              ┌──────────────────────┐
    │  YOLOv8 Object  │              │  Lane & Road Analysis │
    │    Detector     │              │  (Hough, Pothole CNN)│
    │ (Vehicles,      │              │  (Phase 2+)          │
    │  Pedestrians,   │              └──────────────────────┘
    │  Animals)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Multi-Object Tracker        │
    │ (DeepSORT/ByteTrack)        │
    │ (Phase 3+)                  │
    └────────┬────────────────────┘
             │
    ┌────────┴──────────────────────────────────────┐
    │                                               │
    ▼                                               ▼
┌────────────────────────┐            ┌─────────────────────┐
│ Scene Intelligence     │            │  Pose Estimation    │
│ Layer                  │            │  (Fall Detection)   │
│ (Speed, Zone Logic,    │            │  (Phase 3+)         │
│  Near-Miss, Queue      │            └─────────────────────┘
│  Length, Analytics)    │
│ (Phase 3-6)            │
└────────┬───────────────┘
         │
    ┌────┴─────────────────────┬─────────────┬─────────────┐
    ▼                          ▼             ▼             ▼
┌──────────┐          ┌──────────────┐  ┌───────────┐ ┌────────────┐
│ Annotated│          │  Event Log   │  │ Heatmap & │ │ Alert      │
│ Video    │          │  (CSV/JSON)  │  │ Flow Maps │ │ System     │
│ Feed     │          │              │  │           │ │ (Phase 7+) │
└────┬─────┘          └──────────────┘  └───────────┘ └────────────┘
     │
     ▼
┌──────────────────────────────┐
│   Web Dashboard (Phase 8)    │
│ - Live video stream          │
│ - Live statistics            │
│ - Detection controls         │
│ - Event log viewer           │
│ - Recording toggle           │
└──────────────────────────────┘
```

## Module Architecture

### Core Layer (`src/core/`)
- **VideoCapture**: Handles webcam input with property configuration
- **FrameProcessor**: Image enhancement (CLAHE, denoising)
- **SystemConfig**: YAML-based configuration management

### Detection Layer (`src/detection/`)
- **YOLODetector**: YOLOv8 inference wrapper
- **ObjectDetector**: High-level detection interface with class categorization

### Tracking Layer (`src/tracking/`)
- **MultiObjectTracker**: Persistent ID assignment and trajectory tracking
- Supports: DeepSORT, ByteTrack (to be integrated)

### Road Analysis Layer (`src/road_analysis/`)
- **LaneDetector**: Hough Line Transform + LaneNet (Phase 2)
- **TrafficSignOCR**: Sign recognition + Tesseract OCR (Phase 4)
- **PotholeDetector**: Road damage detection CNN (Phase 5)

### Analytics Layer (`src/analytics/`)
- **SpeedEstimator**: Optical flow-based speed calculation (Phase 3)
- **ZoneLogic**: Zone-based event detection - jaywalking, red light violations (Phase 4)
- **EventDetector**: Near-miss, accidents, queue length (Phase 6)

### Alert Layer (`src/alerts/`)
- **AlertManager**: Centralized alert dispatch with cooldown
- **EventLogger**: CSV/JSON event persistence

### Dashboard Layer (`src/dashboard/`)
- **FastAPI Backend**: RESTful API + WebSocket streaming
- **Frontend**: HTML/CSS/JS responsive dashboard

## Data Flow

```
Input Frame (BGR)
    ↓
[Frame Processor]
    ↓
Preprocessed Frame
    ├──→ [YOLOv8 Detector]
    │        ↓
    │    Detection Results
    │        ↓
    ├──→ [Tracker]
    │        ↓
    │    Tracked Objects + IDs
    │        ↓
    └──→ [Analytics]
             ├─→ Speed Estimation
             ├─→ Zone Logic
             ├─→ Event Detection
             └─→ Statistics
    ↓
Processed Results
    ├──→ [Annotated Output]
    ├──→ [Alert Manager]
    ├──→ [Event Logger]
    └──→ [Dashboard]
```

## Configuration Hierarchy

```
Environment Variables (.env)
    ↓
Command Line Arguments
    ↓
YAML Configuration (config/*.yaml)
    ↓
SystemConfig (default values)
```

## Model Specifications

| Component | Model | Framework | Purpose |
|-----------|-------|-----------|---------|
| Object Detection | YOLOv8 (n/s/m/l) | PyTorch | Vehicles, pedestrians, animals |
| Tracking | DeepSORT | PyTorch | Multi-object tracking |
| Pose Estimation | MediaPipe Pose | TFLite | Fall detection |
| Lane Detection | Hough/LaneNet | OpenCV/PyTorch | Road lane identification |
| Pothole Detection | Custom CNN | PyTorch | Road damage detection |
| OCR | Tesseract | EasyOCR | License plate, sign text |

## Performance Targets

| Metric | CPU | GPU (NVIDIA) |
|--------|-----|-------------|
| FPS (YOLOv8n) | 10-15 fps | 30+ fps |
| Latency | 60-100ms | 20-30ms |
| Memory | ~2GB | ~4GB |

## Scalability Considerations

### Multi-Camera Setup
```python
cameras = [
    VideoCapture(camera_index=0),
    VideoCapture(camera_index=1),
    VideoCapture(camera_index=2)
]
for cap in cameras:
    frame = cap.read()
    # Process each camera independently
```

### Distributed Processing
- Event queue for async processing
- Redis pub/sub for inter-process communication
- Load balancing for multiple inference engines

### Cloud Deployment
- Dockerized application (Dockerfile + docker-compose.yml)
- Kubernetes-ready
- S3/cloud storage for logs and recordings

## Memory Management

- Frame buffering: ~50MB per stream
- Model weights: ~150MB (YOLOv8n)
- Tracking data: ~5MB (per 100 objects)
- Event logs: ~1MB per hour

## API Endpoints

```
GET  /                      → Dashboard page
GET  /api/status            → System status
GET  /api/stats             → Live statistics
GET  /api/events            → Event log
GET  /video_feed            → MJPEG stream
WS   /ws/video              → WebSocket video feed
POST /api/config/update     → Update configuration
POST /api/recording/start   → Start recording
POST /api/recording/stop    → Stop recording
```

## Error Handling

- Graceful degradation if model fails
- Fallback to CPU if GPU unavailable
- Reconnect logic for camera disconnection
- Exception logging to event log

---

**Architecture Version**: 1.0
**Last Updated**: 2026-09-14
