"""Unit tests for lane detection module"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.road_analysis.lane_detector import LaneDetector


class TestLaneDetector:
    """Test cases for LaneDetector"""

    def test_initialization(self):
        detector = LaneDetector()
        assert detector is not None

    def test_detect_on_blank_frame(self):
        """Blank frame has no edges: detect() must not crash and returns no lines"""
        detector = LaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        result = detector.detect(frame)
        assert result["method"] == "hough"
        assert result["left_line"] is None
        assert result["right_line"] is None
        assert result["departure_warning"] is None

    def test_detect_road_markings_on_blank_frame(self):
        detector = LaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        markings = detector.detect_road_markings(frame)
        assert markings["stop_lines"] == []
        assert markings["zebra_crossings"] == []
        assert markings["speed_bumps"] == []
        assert markings["arrows"] == []

    def test_detect_road_edges_on_blank_frame(self):
        detector = LaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect_road_edges(frame) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
