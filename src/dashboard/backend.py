"""
FastAPI backend for the web dashboard
"""

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import asyncio
from typing import Dict, Any


def create_app(config: Dict[str, Any] = None) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        config: Application configuration
    
    Returns:
        Configured FastAPI app
    """
    app = FastAPI(title="Road Object Detection Dashboard")
    
    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
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
        return {
            "status": "running",
            "fps": 0,
            "uptime": 0,
            "camera_connected": True,
            "model_loaded": True,
            "gpu_mode": False
        }
    
    @app.get("/api/stats")
    async def get_stats():
        """Get detection statistics"""
        return {
            "total_detections": 0,
            "vehicle_count": 0,
            "pedestrian_count": 0,
            "animal_count": 0,
            "recent_events": []
        }
    
    @app.websocket("/ws/video")
    async def websocket_video(websocket: WebSocket):
        """WebSocket endpoint for video streaming"""
        await websocket.accept()
        try:
            while True:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    return app
