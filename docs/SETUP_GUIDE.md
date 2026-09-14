# Installation & Setup Guide

## System Requirements

### Minimum (CPU-only)
- Python 3.9+
- 4GB RAM
- 10GB Storage
- Any USB webcam

### Recommended (GPU)
- Python 3.9+
- 8GB+ RAM
- NVIDIA GPU (CUDA 11.8+)
- 15GB Storage

## Pre-Installation Steps

### 1. Clone/Setup Repository
```bash
cd /workspaces/Road-Object-Detection-System
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip setuptools wheel
```

## Installation

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Pre-trained Models
```bash
python scripts/setup_models.py
```

This will:
- Download YOLOv8 (nano, small, medium)
- Create `data/models/` directory
- Cache models for offline use

### Step 3: Configure Environment
```bash
cp .env.example .env
# Edit .env with your camera index and settings
```

### Step 4: Test Webcam
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

## Running the System

### Basic Run (Default Settings)
```bash
python src/main.py
```

### With Custom Configuration
```bash
python src/main.py --config config/default_config.yaml
```

### Demo Mode
```bash
python src/main.py --demo
```

### Specific Webcam
```bash
python src/main.py --webcam 1  # Use camera index 1
```

### Controls While Running
- **q** - Quit application
- **s** - Save screenshot
- **p** - Pause/resume
- **r** - Start/stop recording

## Dashboard Setup

### Terminal 1: Run Detection System
```bash
python src/main.py
```

### Terminal 2: Start Web Dashboard
```bash
uvicorn src.dashboard.backend:app --reload --port 8000
```

### Access Dashboard
Open browser: `http://localhost:8000`

## Docker Setup

### Build Docker Image
```bash
docker build -t road-detection:latest .
```

### Run with Docker
```bash
docker run --rm \
  --device /dev/video0 \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  road-detection:latest
```

### Docker Compose
```bash
docker-compose up
```

## Troubleshooting

### Issue: Camera Not Detected
```bash
# List available cameras
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print(frame.shape if ret else 'Failed')"
```

### Issue: CUDA/GPU Not Working
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU mode
export DEVICE=cpu
python src/main.py
```

### Issue: Out of Memory
```bash
# Use smaller model
python src/main.py --model yolov8n

# Reduce frame size in config
# Modify: capture_width and capture_height in default_config.yaml
```

### Issue: Low FPS
- Use YOLOv8n (nano) instead of larger models
- Enable GPU acceleration
- Reduce frame resolution
- Disable unnecessary features in config

## Optional: Camera Calibration

For accurate speed estimation and perspective analysis:
```bash
python scripts/calibrate_camera.py
```

This will create `config/camera_calibration.yaml` with:
- Intrinsic camera matrix
- Distortion coefficients
- Pixels-to-meters conversion factor

## Running Tests
```bash
pytest tests/ -v
```

## Development Setup

For development contributions:
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

## Performance Tuning

### CPU Optimization
```yaml
# config/default_config.yaml
model_name: yolov8n
confidence_threshold: 0.6  # Higher = faster but fewer detections
iou_threshold: 0.5         # Higher = faster
```

### GPU Optimization
```yaml
model_name: yolov8s        # Small or medium model
batch_size: 4              # Process multiple frames
```

### Memory Optimization
```yaml
capture_width: 640         # Smaller input size
capture_height: 480
save_recordings: false     # Disable recording to save disk space
```

## Deployment Checklist

- [ ] Environment file (.env) configured
- [ ] Camera detected and working
- [ ] Models downloaded successfully
- [ ] Configuration file validated
- [ ] Test run successful
- [ ] Dashboard accessible
- [ ] Event logs being created
- [ ] Performance acceptable

## Next Steps

1. **Phase 1**: Verify basic detection works
2. **Phase 2**: Test lane detection (if enabled)
3. **Phase 3**: Monitor tracking accuracy
4. **Phase 4**: Validate sign recognition
5. **Phase 8**: Use web dashboard for monitoring

---

**Guide Version**: 1.0
**Last Updated**: 2026-09-14
