# 🚗 Road Object Detection System — Idea Doc
> Robotics Club Project | Webinar Edition

> A comprehensive OpenCV + AI-powered real-time road awareness system that detects, tracks, and analyzes everything on the road — built for a live webinar demonstration.

---

## Problem Statement

Roads are dynamic, high-risk environments where split-second awareness is critical. Existing solutions are either too expensive (LiDAR-based ADAS systems), too narrowly scoped (just lane detection or just pedestrian detection), or locked behind proprietary hardware. There's a clear need for an **open, camera-first, multi-class road awareness system** that works on commodity hardware like a USB webcam.

---

## Hardware Setup

| Component | Model | Cost |
|---|---|---|
| Webcam | Zebronics Sharp PRO (2048×1536 QXGA, 30FPS, Autofocus) | ~₹2,000–3,500 |
| Laptop / PC | NVIDIA GPU preferred (CPU mode available) | Already owned |
| Tripod | Use webcam's built-in tripod support | — |

> Frame is downsampled to 1280×720 for inference; full QXGA resolution preserved for recording.

---

## Complete Feature List

### 🧍 People & Vulnerable Road User Detection
- **Pedestrian detection** — bounding box + confidence score on every person visible
- **Pedestrian on road alert** — triggers when a person crosses into the road zone
- **Cyclist detection** — separate class from pedestrians, tracked independently
- **Motorcyclist detection** — including pillion rider count
- **Child vs adult classification** — size-based heuristic to flag children near roads
- **Crowd density estimation** — count people in a zone, flag overcrowding
- **Person direction prediction** — is this pedestrian walking toward the road or away?
- **Fall detection** — detect if a person has fallen on the road (pose estimation)
- **Jaywalking detection** — person crossing outside zebra crossing zone

### 🚗 Vehicle Detection & Classification
- **Multi-class vehicle detection** — car, truck, bus, motorcycle, auto-rickshaw, bicycle, tempo, tractor
- **Vehicle counting** — total count per class, updated live
- **Vehicle color detection** — dominant color extraction per vehicle bounding box
- **Vehicle size classification** — small / medium / large / heavy
- **Emergency vehicle detection** — ambulance, police car, fire truck (shape + color cues)
- **Wrong-way vehicle detection** — vehicle moving against expected traffic flow direction
- **Stopped vehicle detection** — vehicle stationary for >N seconds flagged as stalled
- **Overspeeding detection** — estimated speed vs zone speed limit comparison

### 🐄 Animal Detection (India-specific)
- Cattle on road (cow, buffalo)
- Stray dogs
- General animal class as fallback

### 🛣️ Lane & Road Analysis
- **Lane line detection** — solid, dashed, double lines via Hough Transform / LaneNet
- **Lane departure warning** — vehicle crossing lane without indicator
- **Lane occupancy** — which lanes are busy vs free
- **Zebra crossing detection** — identify pedestrian crossing zones
- **Road marking recognition** — arrows, stop lines, speed bumps marked on road
- **Pothole detection** — surface anomaly CNN detecting depressions and cracks
- **Waterlogging detection** — reflective surface analysis to detect flooded patches
- **Debris detection** — foreign objects on road (rocks, fallen branches, garbage)
- **Road edge detection** — identify where the road ends (useful for rural roads)

### 🚦 Traffic Infrastructure Detection
- **Traffic light detection** — red / yellow / green state classification
- **Red light violation detection** — vehicle crossing stop line on red
- **Traffic sign recognition** — stop, yield, speed limit, no-entry, one-way
- **Speed limit sign OCR** — read the number off the sign using Tesseract
- **License plate detection** — locate plate region on vehicle
- **License plate OCR** — read plate text (useful for violation logging)
- **Road divider / median detection** — physical barriers and islands

### 📊 Traffic Analytics
- **Vehicle speed estimation** — using optical flow + frame rate + perspective calibration
- **Traffic density heatmap** — live heatmap overlay showing congestion zones
- **Vehicle flow direction map** — directional arrows showing movement patterns
- **Intersection occupancy** — how full is the intersection at any moment
- **Traffic queue length estimation** — measure tailback length in pixels / meters
- **Peak hour detection** — rolling average vehicle count to identify busy periods
- **Near-miss event detection** — two objects getting dangerously close, flagged as incident
- **Accident detection** — sudden stop + overlap of bounding boxes as a proxy signal

### 🌦️ Environmental & Lighting Adaptation
- **Low light / night mode** — histogram equalization + CLAHE preprocessing
- **Fog / rain detection** — image clarity score; auto-switch to fog-enhanced model
- **Glare handling** — adaptive thresholding for high-contrast sunlight scenarios
- **Shadow removal** — morphological operations to reduce shadow-induced false positives

### 🎯 Tracking & Behavior Analysis
- **Multi-object tracking** — persistent IDs for every object across frames (DeepSORT / ByteTrack)
- **Object trajectory visualization** — draw trails showing where each object has been
- **Re-identification across occlusion** — maintain ID when object passes behind another
- **Dwell time tracking** — how long has this object been in this zone
- **Entry / exit zone counting** — count objects entering or leaving a defined region
- **Loitering detection** — person/vehicle stationary in a zone for unusual duration

### 🔔 Alert & Event System
- **Audio beep alerts** — different tones for different event severity
- **On-screen alert banners** — overlaid warning text on the video feed
- **Event logger** — timestamped CSV/JSON log of every detection event
- **Screenshot on event** — auto-save a frame when a critical event fires
- **Push notification** (optional) — send alerts to phone via Telegram Bot API
- **Alert cooldown** — prevent repeated alerts for the same ongoing event

### 📺 Web Dashboard (Simple Website)
Built with **FastAPI (backend) + plain HTML/CSS/JS (frontend)** — no heavy frameworks, opens in any browser on the same network.

- **Live annotated video stream** — MJPEG stream embedded in browser, bounding boxes + labels visible in real time
- **Live stats panel** — vehicle count, pedestrian count, FPS counter, uptime
- **Active alerts feed** — scrolling list of recent events (e.g. "Pedestrian on road — 14:32:05")
- **Detection class toggles** — checkboxes to enable/disable specific classes live
- **Confidence threshold slider** — tune sensitivity without restarting the system
- **Heatmap view toggle** — switch between raw annotated feed and traffic heatmap overlay
- **License plate log table** — live table of detected plates with timestamp and cropped image
- **Event log download** — button to download the session's CSV log
- **Recording toggle** — start/stop saving annotated video to disk from the browser
- **System status bar** — shows camera connected, model loaded, GPU/CPU mode

### 🧪 Webinar-Specific Demo Features
- **Pause & annotate mode** — freeze frame, highlight detections for explanation
- **Confidence threshold slider** — live tune detection sensitivity on screen
- **Class filter toggle** — turn on/off specific detection classes live
- **Slow-motion playback** — replay a detection event at 0.25x speed
- **Side-by-side view** — raw feed vs annotated feed split screen
- **Stats summary card** — end-of-session summary (total objects detected, events fired, etc.)

---

## Detection Classes Summary

| Category | Classes |
|---|---|
| People | Person, Cyclist, Motorcyclist, Child |
| Vehicles | Car, Bus, Truck, Auto, Motorcycle, Bicycle, Tempo, Tractor, Ambulance |
| Animals | Cow, Buffalo, Dog, Generic Animal |
| Road Objects | Pothole, Debris, Speed Bump, Waterlogging |
| Infrastructure | Traffic Light, Stop Sign, Speed Limit Sign, No Entry, One Way |
| Road Markings | Lane Line, Zebra Crossing, Stop Line, Direction Arrow |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Capture | OpenCV `VideoCapture` (Zebronics webcam @ 1280×720) |
| Detection | YOLOv8 (Ultralytics) — nano for CPU, small/medium for GPU |
| Tracking | DeepSORT / ByteTrack |
| Lane Detection | Hough Line Transform + polynomial fit / LaneNet |
| Pose Estimation | MediaPipe Pose (for fall detection) |
| OCR | Tesseract (license plates, speed signs) |
| Analytics | NumPy, SciPy, OpenCV optical flow |
| Dashboard | FastAPI backend + plain HTML/CSS/JS frontend |
| Video stream | MJPEG stream via FastAPI `/video_feed` endpoint |
| Alerts | On-screen audio beep (Pygame) + web dashboard alert feed |
| Logging | CSV / JSON event log |

---

## Architecture Overview

```
Zebronics Sharp PRO Webcam (QXGA)
        │
        ▼
  Frame Preprocessor
  (resize to 720p, denoise, CLAHE for low light)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  YOLOv8 Object Detector               Lane & Road Analyzer
  (vehicles, people, animals,          (Hough lines, pothole CNN,
   signs, traffic lights)               zebra crossing, markings)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
          Multi-Object Tracker (DeepSORT)
          + Pose Estimator (MediaPipe)
                       │
                       ▼
            Scene Intelligence Layer
        (speed est., zone logic, near-miss,
         red light violation, crowd density)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Annotated     Event Log    Alert System
     Video Feed    (CSV/JSON)   (audio + push)
          │
          ▼
     Live Dashboard
     (local + web browser)
```

---

## Milestones

| Phase | Goal | Timeline |
|---|---|---|
| Phase 1 | Webcam feed + basic vehicle & pedestrian detection | Week 1–2 |
| Phase 2 | Lane detection + road markings | Week 3–4 |
| Phase 3 | Multi-object tracking + speed estimation | Week 5–6 |
| Phase 4 | Traffic sign recognition + red light violation | Week 7–8 |
| Phase 5 | Pothole, debris, waterlogging detection | Week 9–10 |
| Phase 6 | Analytics — heatmap, flow map, queue length | Week 11–12 |
| Phase 7 | Alert system + event logger + Telegram bot | Week 13–14 |
| Phase 8 | Web dashboard + webinar demo features | Week 15–16 |

---

## Dataset Sources

| Dataset | What It Covers |
|---|---|
| COCO | General objects — vehicles, people, animals |
| BDD100K | Diverse driving scenes, lane annotations, weather |
| Cityscapes | Urban road segmentation |
| IDD (India Driving Dataset) | India-specific — autos, cattle, chaotic intersections |
| OIDDS / Roboflow | Traffic signs, potholes, road damage |
| UA-DETRAC | Vehicle detection and tracking benchmark |
| CityFlow | Multi-camera vehicle tracking |

> Fine-tune on IDD for India-specific deployment. Roboflow has pre-labelled pothole datasets ready to use.

---

## Webinar Demo Plan

1. **Intro (5 min)** — Problem statement, what the system does, hardware shown on screen
2. **Live feed demo (10 min)** — Point webcam at road footage / window; show detections live
3. **Feature walkthrough (15 min)** — Toggle classes, show heatmap, trigger alerts, show event log
4. **Architecture deep-dive (10 min)** — Explain the pipeline, models used, how tracking works
5. **India-specific angle (5 min)** — IDD dataset, cattle detection, auto-rickshaw class
6. **Q&A (15 min)**

---

## Decisions Made

| Decision | Choice | Reason |
|---|---|---|
| License plate detection | ✅ In scope | Campus testing — plates visible and relevant |
| Privacy / face blur | ❌ Not needed | Closed campus environment, no public privacy concern |
| Telegram alerts | ❌ Skipped | Web dashboard covers alerting needs |
| Alert delivery | ✅ Web dashboard | Simple, demo-friendly, no external dependencies |

## Open Questions

- [ ] Lane detection: classical Hough vs learned LaneNet — test both and compare FPS hit
- [ ] Night / rain mode: single model with augmentation or separate model per condition?
- [ ] Fall detection accuracy: MediaPipe pose reliable enough or need dedicated model?

---

## References

- [OpenCV Docs](https://docs.opencv.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [India Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/)
- [DeepSORT Tracking](https://github.com/nwojke/deep_sort)
- [MediaPipe Pose](https://mediapipe.dev/)
- [Roboflow Pothole Dataset](https://roboflow.com/datasets)
- [LaneNet Paper](https://arxiv.org/abs/1802.05591)

---

*Robotics Club — Road Detection Project | Updated Sep 2026*
