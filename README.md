# 🚗 Road Object Detection System

A comprehensive **AI-powered road monitoring and traffic intelligence system** using YOLOv8, computer vision, and deep learning. Designed for real-time object detection, traffic analysis, and road condition monitoring with a 16-week development roadmap spanning 8 phases.

**Status**: ✅ Complete project structure & scaffolding | 🚧 Implementation phase

---

## 🎯 Project Overview

This system provides **end-to-end road monitoring capabilities** including:
- Real-time vehicle, pedestrian, and animal detection
- Lane detection and road marking recognition
- Multi-object tracking with persistent IDs
- Traffic sign recognition and OCR
- Pothole and road damage detection
- Analytics (traffic flow, density, speed estimation)
- Real-time alerts and event logging
- Web dashboard for monitoring and control

**Target Deployment**: Production-ready on DigitalOcean ($10-15/month) with Supabase database

---

## ✨ Key Features

| Feature | Phase | Status | Tech Stack |
|---------|-------|--------|-----------|
| **Webcam Feed + Detection** | 1 | ✅ Scaffolded | OpenCV, YOLOv8, NumPy |
| **Lane Detection** | 2 | 🚧 Template | Hough Transform, LaneNet |
| **Multi-Object Tracking** | 3 | 🚧 Template | DeepSORT, ByteTrack |
| **Traffic Signs & OCR** | 4 | 🚧 Template | YOLOv8, Tesseract, EasyOCR |
| **Pothole Detection** | 5 | 🚧 Template | U-Net, YOLOv8, CNN |
| **Analytics Engine** | 6 | 🚧 Template | Pandas, Matplotlib, Seaborn |
| **Alert System** | 7 | 🚧 Template | PyGame, Telegram Bot |
| **Web Dashboard** | 8 | ✅ Scaffolded | FastAPI, HTML/CSS/JS |

---

## 📊 8-Phase Development Roadmap

### Phase 1: Webcam + Basic Detection (Week 1-2)
- Real-time video capture
- YOLOv8 multi-class detection (23 classes)
- Frame preprocessing (CLAHE, denoising)
- FPS monitoring & statistics
- **Performance Target**: 15+ FPS (CPU), 30+ FPS (GPU)
- **Accuracy Target**: >70%

### Phase 2: Lane Detection (Week 3-4)
- Hough Transform for lane line detection
- LaneNet deep learning approach
- Road marking recognition
- Zebra crossing detection

### Phase 3: Tracking + Speed Estimation (Week 5-6)
- DeepSORT multi-object tracking
- Kalman filter trajectory prediction
- Optical flow speed estimation
- Vehicle trajectory visualization

### Phase 4: Traffic Intelligence (Week 7-8)
- YOLOv8 fine-tuning for traffic signs
- Tesseract/EasyOCR for text recognition
- License plate reading
- Red light violation detection
- Jaywalking detection

### Phase 5: Pothole & Road Damage (Week 9-10)
- CNN-based pothole detection
- Debris detection
- Waterlogging detection
- Severity classification

### Phase 6: Analytics Engine (Week 11-12)
- Traffic density heatmap
- Vehicle flow analysis
- Queue length estimation
- Peak hour detection
- Near-miss detection
- Database: CSV → PostgreSQL/Supabase

### Phase 7: Alert System (Week 13-14)
- Alert manager with cooldown
- CSV/JSON event logging
- Audio alerts (different frequencies)
- Optional Telegram bot integration
- Real-time notifications

### Phase 8: Web Dashboard + Deployment (Week 15-16)
- FastAPI REST API
- MJPEG + WebSocket video streaming
- Live statistics dashboard
- Recording controls
- Event log viewer
- Webinar demo features

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM (8GB+ recommended)
- GPU optional (NVIDIA CUDA for acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/bastitva0-blip/Road-Object-Detection-System.git
cd Road-Object-Detection-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model
python scripts/setup_models.py

# Run the system
python src/main.py
```

### Configuration

Edit `config/default_config.yaml`:
```yaml
camera_index: 0              # Webcam index
capture_width: 1280
capture_height: 720
fps: 30
model_name: yolov8n          # nano, small, medium, large
confidence_threshold: 0.5
enable_tracking: false       # Enable after Phase 3
enable_lane_detection: false # Enable after Phase 2
enable_analytics: false      # Enable after Phase 6
```

### Run with Arguments

```bash
# Run with default config
python src/main.py

# Use custom config
python src/main.py --config config/custom_config.yaml

# Demo mode (no camera required)
python src/main.py --demo

# Webcam feed only
python src/main.py --webcam
```

### Keyboard Controls
- `q` - Quit
- `s` - Save screenshot
- `p` - Pause/Resume
- `r` - Start/Stop recording

---

## 📁 Project Structure

```
Road-Object-Detection-System/
├── src/                          # Main source code
│   ├── main.py                   # Entry point
│   ├── core/                     # Core modules
│   │   ├── config.py            # Configuration management
│   │   ├── video_capture.py     # Webcam wrapper
│   │   └── frame_processor.py   # Preprocessing
│   ├── detection/                # Detection modules
│   │   ├── yolo_detector.py     # YOLOv8 wrapper
│   │   └── object_detector.py   # High-level interface
│   ├── tracking/                 # Multi-object tracking
│   │   └── tracker.py
│   ├── road_analysis/            # Road analysis modules
│   │   ├── lane_detector.py
│   │   ├── traffic_sign_ocr.py
│   │   └── pothole_detector.py
│   ├── analytics/                # Analytics engine
│   │   ├── speed_estimator.py
│   │   ├── zone_logic.py
│   │   └── event_detector.py
│   ├── alerts/                   # Alert system
│   │   ├── alert_manager.py
│   │   └── event_logger.py
│   └── dashboard/                # Web dashboard
│       ├── backend.py            # FastAPI server
│       └── static/               # Frontend assets
│           ├── index.html
│           ├── styles.css
│           └── app.js
│
├── config/                       # Configuration files
│   ├── default_config.yaml      # Main config
│   └── detection_classes.yaml   # Class definitions
│
├── data/                         # Data directories
│   ├── logs/                    # Event logs
│   ├── recordings/              # Video recordings
│   └── screenshots/             # Captured frames
│
├── scripts/                      # Utility scripts
│   └── setup_models.py          # Model downloader
│
├── tests/                        # Test suite
│   └── test_detection.py
│
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md
│   └── SETUP_GUIDE.md
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Dockerfile                    # Docker container
├── docker-compose.yml            # Multi-container setup
├── .env.example                  # Environment variables
├── .gitignore                    # Git ignore rules
│
├── PHASES.md                     # Detailed phase roadmap (📖 START HERE)
├── QUICK_START.md               # Quick start guide
├── README_STRUCTURE.md           # Folder structure details
├── PROJECT_COMPLETE.md           # Completion summary
└── CHANGELOG.md                  # Version history
```

---

## 📦 Dependencies

**Core Libraries** (40+ packages):
- `opencv-python` - Computer vision
- `ultralytics` - YOLOv8 detection
- `torch`, `torchvision` - Deep learning
- `numpy`, `scipy` - Numerical computing
- `fastapi`, `uvicorn` - Web framework
- `pydantic` - Data validation
- `pandas` - Data analysis
- `pyyaml` - Configuration

**Specialized Libraries**:
- `deep-sort-realtime` - Multi-object tracking
- `mediapipe` - Pose & hand detection
- `pytesseract`, `easyocr`, `paddleocr` - OCR
- `scikit-learn`, `scikit-image` - ML utilities
- `matplotlib`, `seaborn` - Visualization
- `pytest` - Testing

See [requirements.txt](requirements.txt) for complete list with pinned versions.

---

## 🌐 Deployment Options

| Option | Cost | Pros | Cons | Use Case |
|--------|------|------|------|----------|
| **Local Network** | $0 | No cost, fast development | Limited access | Development, webinars |
| **DigitalOcean** ⭐ | $10-15/mo | Full app, managed DB, SSH access | Requires management | **Recommended for production** |
| **Vercel** | $0 | Free frontend hosting | No backend support | Frontend only |
| **Heroku/Railway** | $5-7/mo | Easy deployment, git push | Limited resources | Small deployments |
| **AWS** | $$$ | Scalable, enterprise-grade | Complex setup, expensive | Enterprise scale |
| **Google Cloud Run** | $0.00002/sec | Serverless, pay-per-use | Cold starts, vendor lock-in | Sporadic workloads |

**Recommended Stack**: DigitalOcean + Supabase ($10-25/month total)

---

## 🗄️ Database Options

| Option | Cost | Pros | Cons | Best For |
|--------|------|------|------|----------|
| **CSV** | $0 | Simple, no setup | Limited scale, no queries | Phase 1-5 prototyping |
| **PostgreSQL** | $0-50/mo | Powerful, open-source | Requires management | Production, analytics |
| **Supabase** ⭐ | $0-25/mo | PostgreSQL + Auth + Real-time + REST API | Vendor lock-in | **Recommended, easy setup** |
| **AWS RDS** | $15+/mo | Managed, scalable | Expensive, complex | Enterprise applications |

**Recommended**: Supabase free tier (500MB storage, unlimited API calls)
- PostgreSQL database
- Real-time subscriptions
- Built-in authentication
- REST API + WebSocket support

---

## 🛠️ Usage Examples

### Basic Detection Only
```bash
python src/main.py --config config/default_config.yaml
```

### With Dashboard
```bash
# Terminal 1: Start detection system
python src/main.py

# Terminal 2: Start web dashboard
python src/dashboard/backend.py

# Open browser: http://localhost:8000
```

### With Docker
```bash
# Build image
docker build -t road-detection .

# Run container
docker run --device /dev/video0 -p 8000:8000 road-detection
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_detection.py -v
```

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [**PHASES.md**](PHASES.md) | 📋 Complete 8-phase development roadmap (**START HERE**) |
| [QUICK_START.md](QUICK_START.md) | ⚡ 5-minute getting started guide |
| [README_STRUCTURE.md](README_STRUCTURE.md) | 📁 Detailed folder structure explanation |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 🏗️ System design and data flow |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | 🔧 Installation and troubleshooting |
| [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md) | ✅ Project completion summary |
| [CHANGELOG.md](CHANGELOG.md) | 📝 Version history |

---

## 🎓 Learning Path

1. **Read** [PHASES.md](PHASES.md) - Understand the full roadmap
2. **Setup** - Follow installation steps above
3. **Phase 1** - Run basic detection: `python src/main.py`
4. **Expand** - Follow PHASES.md for each phase sequentially
5. **Deploy** - Use DigitalOcean + Supabase for production

---

## 🔗 Technology Stack

**Detection & Computer Vision**:
- YOLOv8 (Ultralytics)
- OpenCV
- MediaPipe
- Scikit-Image

**Tracking & Analytics**:
- DeepSORT / ByteTrack
- Kalman Filters
- Optical Flow

**Web & API**:
- FastAPI
- Uvicorn
- Pydantic
- WebSocket

**Database**:
- PostgreSQL
- Supabase
- CSV (initial)

**ML & Data Science**:
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

---

## 🤝 Contributing

This is an active development project. Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

Follow the code style and testing conventions outlined in documentation.

---

## 📊 Project Statistics

- **Total Files**: 48
- **Python Modules**: 25+
- **Lines of Code**: ~3,000 (scaffolded)
- **Documentation**: ~3,000 lines
- **Development Timeline**: 16 weeks (8 phases)
- **Deployment Cost**: $10-25/month recommended

---

## 🐛 Known Issues & Limitations

- Phase 1 scaffolding complete; phases 2-8 ready for implementation
- GPU acceleration requires NVIDIA CUDA toolkit
- Dashboard WebSocket streaming requires low-latency network
- OCR accuracy depends on image quality and traffic sign condition

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OpenCV](https://opencv.org/) - Computer vision library
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## 📞 Support & Contact

For issues, questions, or suggestions:
1. Check [PHASES.md](PHASES.md) for detailed documentation
2. Review [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for troubleshooting
3. Open an issue on GitHub
4. Submit a pull request with improvements

---

## 🚀 Getting Started Now

```bash
# 1. Clone
git clone https://github.com/bastitva0-blip/Road-Object-Detection-System.git && cd Road-Object-Detection-System

# 2. Install
pip install -r requirements.txt

# 3. Setup models
python scripts/setup_models.py

# 4. Run
python src/main.py

# 5. Read the roadmap
cat PHASES.md
```

**Ready to build? Start with Phase 1!** 🎯

---

**Last Updated**: September 2026  
**Repository**: https://github.com/bastitva0-blip/Road-Object-Detection-System