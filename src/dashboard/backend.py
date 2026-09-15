"""
FastAPI backend for the web dashboard.

Runs a DetectionEngine (the same pipeline main.py uses) on a background
thread, and serves its output over HTTP: MJPEG video, JSON stats/events, and
POST endpoints to control it at runtime.
"""

import asyncio
import base64
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Make the sibling `core`, `detection`, etc. packages importable, same as main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import SystemConfig
from core.engine import DetectionEngine

logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    confidence_threshold: Optional[float] = None


class EngineWorker:
    """Runs DetectionEngine.process_frame() in a loop on a background thread"""

    def __init__(self, engine: DetectionEngine):
        self.engine = engine
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                if not self.engine.process_frame():
                    time.sleep(0.1)
            except Exception:
                logger.exception("Engine worker frame processing failed")
                time.sleep(0.1)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.engine.shutdown()


def create_app(config: SystemConfig = None, webcam_index: int = 0, demo_mode: bool = False) -> FastAPI:
    """
    Create and configure the FastAPI application

    Args:
        config: System configuration (defaults loaded if None)
        webcam_index: Camera device index
        demo_mode: Use a synthetic video feed instead of a real camera

    Returns:
        Configured FastAPI app; the engine starts on the "startup" event
    """
    config = config or SystemConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = DetectionEngine(config, webcam_index=webcam_index, demo_mode=demo_mode)
        worker = EngineWorker(engine)
        worker.start()
        app.state.worker = worker
        logger.info("Detection engine started" + (" (demo mode)" if demo_mode else ""))
        yield
        worker.stop()

    app = FastAPI(title="Road Object Detection Dashboard", lifespan=lifespan)
    app.state.worker = None

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _engine() -> DetectionEngine:
        if app.state.worker is None:
            raise HTTPException(status_code=503, detail="Engine not started yet")
        return app.state.worker.engine

    @app.get("/")
    async def root():
        """Serve main dashboard page"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Road Object Detection Dashboard"}

    @app.get("/api/status")
    async def get_status():
        """Get system status"""
        engine = _engine()
        stats = engine.get_latest_stats()
        return {
            "status": "running",
            "fps": stats.get("fps", 0),
            "uptime": stats.get("uptime", 0),
            "camera_connected": engine.cap.is_opened(),
            "model_loaded": True,
            "gpu_mode": False,
            "demo_mode": engine.demo_mode,
        }

    @app.get("/api/stats")
    async def get_stats():
        """Get live detection statistics"""
        return _engine().get_latest_stats()

    @app.get("/api/events")
    async def get_events(limit: int = 50):
        """Paginated recent alert/event log"""
        return {"events": _engine().alert_manager.get_recent_alerts(limit=limit)}

    @app.post("/api/config/update")
    async def update_config(update: ConfigUpdate):
        """Update runtime-adjustable settings (currently: confidence_threshold)"""
        engine = _engine()
        if update.confidence_threshold is not None:
            engine.update_confidence_threshold(update.confidence_threshold)
        return {"confidence_threshold": engine.config.confidence_threshold}

    @app.post("/api/recording/start")
    async def start_recording():
        engine = _engine()
        if engine.recorder is None:
            recording = engine.toggle_recording()
        else:
            recording = True
        return {"recording": recording}

    @app.post("/api/recording/stop")
    async def stop_recording():
        engine = _engine()
        if engine.recorder is not None:
            engine.toggle_recording()
        return {"recording": False}

    @app.post("/api/heatmap/toggle")
    async def toggle_heatmap():
        return {"heatmap": _engine().toggle_heatmap()}

    def _mjpeg_generator():
        engine = _engine()
        boundary = b"--frame"
        while True:
            frame = engine.get_latest_frame()
            if frame is not None:
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                           + buffer.tobytes() + b"\r\n")
            time.sleep(1 / 15)  # cap the stream at 15 FPS regardless of engine FPS

    @app.get("/video_feed")
    async def video_feed():
        """MJPEG live video stream (used by the dashboard's <img> tag)"""
        return StreamingResponse(
            _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/ws/video")
    async def websocket_video(websocket: WebSocket):
        """Alternative to /video_feed: base64 JPEG frames over a WebSocket"""
        await websocket.accept()
        engine = _engine()
        try:
            while True:
                frame = engine.get_latest_frame()
                if frame is not None:
                    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        await websocket.send_text(base64.b64encode(buffer.tobytes()).decode("ascii"))
                await asyncio.sleep(1 / 15)
        except WebSocketDisconnect:
            logger.info("Video WebSocket client disconnected")

    return app


# Module-level app for `uvicorn src.dashboard.backend:app` (see PHASES.md testing strategy).
# Reads settings from the environment since the uvicorn CLI can't pass Python kwargs.
import os  # noqa: E402

app = create_app(
    webcam_index=int(os.environ.get("WEBCAM_INDEX", "0")),
    demo_mode=os.environ.get("DEMO_MODE", "false").lower() == "true",
)
