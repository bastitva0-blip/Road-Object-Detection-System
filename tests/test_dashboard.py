"""
Integration test for the FastAPI dashboard backend, using a real DetectionEngine
in demo mode (synthetic video feed) — no mocking of the engine or YOLO model.
Slow (loads the real model once per session) but exercises the real code path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient

from src.core.config import SystemConfig
from src.dashboard.backend import create_app


@pytest.fixture(scope="module")
def client():
    config = SystemConfig()
    app = create_app(config, demo_mode=True)
    with TestClient(app) as c:
        yield c


class TestDashboardEndpoints:
    def test_root_serves_dashboard_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Road Object Detection" in response.content

    def test_static_files_served(self, client):
        response = client.get("/static/app.js")
        assert response.status_code == 200

    def test_status_endpoint(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["demo_mode"] is True

    def test_stats_endpoint_shape(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        # May be {} if no frame processed yet, or a full stats dict once the
        # background worker has run at least once -- both are valid states.
        assert isinstance(response.json(), dict)

    def test_events_endpoint(self, client):
        response = client.get("/api/events")
        assert response.status_code == 200
        assert "events" in response.json()

    def test_config_update(self, client):
        response = client.post("/api/config/update", json={"confidence_threshold": 0.65})
        assert response.status_code == 200
        assert response.json()["confidence_threshold"] == pytest.approx(0.65)

    def test_recording_start_stop(self, client):
        start = client.post("/api/recording/start")
        assert start.status_code == 200
        stop = client.post("/api/recording/stop")
        assert stop.status_code == 200
        assert stop.json()["recording"] is False

    def test_heatmap_toggle(self, client):
        first = client.post("/api/heatmap/toggle").json()["heatmap"]
        second = client.post("/api/heatmap/toggle").json()["heatmap"]
        assert first != second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
