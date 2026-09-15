"""Unit tests for classical-CV pothole/debris/waterlogging detection"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.road_analysis.pothole_detector import PotholeDetector


def _flat_road_frame(seed: int, noise: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
    noise_arr = rng.integers(-noise, noise, frame.shape)
    return np.clip(frame.astype(int) + noise_arr, 0, 255).astype(np.uint8)


class TestPotholeDetector:
    def test_no_false_positives_on_clean_road(self):
        import cv2
        frame = _flat_road_frame(seed=1, noise=8)
        detector = PotholeDetector()
        result = detector.detect_anomalies(frame)
        assert result["potholes"] == []
        assert result["debris"] == []
        assert result["waterlogging"] == []

    def test_detects_dark_blob_as_pothole(self):
        import cv2
        frame = _flat_road_frame(seed=0)
        cv2.circle(frame, (400, 600), 30, (25, 25, 25), -1)

        detector = PotholeDetector()
        result = detector.detect_anomalies(frame)
        assert len(result["potholes"]) == 1
        x, y, w, h = result["potholes"][0]["bbox"]
        assert x < 400 < x + w
        assert y < 600 < y + h
        assert result["potholes"][0]["severity"] in ("low", "medium", "high")

    def test_detects_color_anomaly_as_debris(self):
        import cv2
        frame = _flat_road_frame(seed=0)
        cv2.rectangle(frame, (700, 550), (715, 565), (30, 30, 200), -1)

        detector = PotholeDetector()
        result = detector.detect_anomalies(frame)
        assert len(result["debris"]) >= 1

    def test_detects_smooth_bright_patch_as_waterlogging(self):
        import cv2
        frame = _flat_road_frame(seed=0)
        cv2.ellipse(frame, (900, 620), (80, 40), 0, 0, 360, (190, 185, 175), -1)

        detector = PotholeDetector()
        result = detector.detect_anomalies(frame)
        assert len(result["waterlogging"]) == 1

    def test_ignores_sky_region_above_roi(self):
        import cv2
        detector = PotholeDetector(roi_top_fraction=0.55)
        frame = _flat_road_frame(seed=0)
        # dark blob above the ROI line (sky/horizon) must not be reported
        cv2.circle(frame, (400, 100), 30, (25, 25, 25), -1)
        result = detector.detect_anomalies(frame)
        assert result["potholes"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
