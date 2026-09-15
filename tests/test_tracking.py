"""Unit tests for multi-object tracking, speed estimation, and zone logic"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.tracker import MultiObjectTracker, iou
from src.analytics.speed_estimator import SpeedEstimator
from src.analytics.zone_logic import ZoneLogic


class TestIOU:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


class TestMultiObjectTracker:
    def test_stable_id_for_moving_object(self):
        tracker = MultiObjectTracker(min_hits=3)
        ids_per_frame = []
        for frame in range(8):
            x = 100 + frame * 10
            det = {"bbox": np.array([x, 200, x + 50, 260]), "confidence": 0.9, "class_name": "car"}
            tracked = tracker.update([det])
            ids_per_frame.append([t["track_id"] for t in tracked])

        confirmed_ids = [ids for ids in ids_per_frame if ids]
        assert len(confirmed_ids) > 0
        assert all(ids == [1] for ids in confirmed_ids)

    def test_no_id_switch_across_brief_occlusion(self):
        tracker = MultiObjectTracker(min_hits=3, max_age=10)
        for frame in range(5):
            x = 100 + frame * 10
            det = {"bbox": np.array([x, 200, x + 50, 260]), "confidence": 0.9, "class_name": "car"}
            tracker.update([det])

        for _ in range(4):
            tracker.update([])  # occluded, no detections

        x = 100 + 9 * 10
        det = {"bbox": np.array([x, 200, x + 50, 260]), "confidence": 0.9, "class_name": "car"}
        tracked = tracker.update([det])
        assert [t["track_id"] for t in tracked] == [1]

    def test_handles_many_simultaneous_objects(self):
        tracker = MultiObjectTracker(min_hits=1)
        n_objects = 35
        detections = [
            {"bbox": np.array([i * 30, 100, i * 30 + 25, 150]), "confidence": 0.9, "class_name": "car"}
            for i in range(n_objects)
        ]
        tracked = tracker.update(detections)
        assert len(tracked) == n_objects
        assert len({t["track_id"] for t in tracked}) == n_objects


class TestSpeedEstimator:
    def test_known_pixel_velocity_converts_correctly(self):
        import cv2
        estimator = SpeedEstimator(fps=30, pixels_per_meter=10.0)
        speed = None
        for frame_idx in range(3):
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            x = 50 + frame_idx * 10
            cv2.rectangle(img, (x, 100), (x + 60, 180), (255, 255, 255), -1)
            cv2.rectangle(img, (x + 10, 110), (x + 20, 120), (0, 0, 0), -1)
            cv2.rectangle(img, (x + 35, 130), (x + 45, 145), (0, 0, 0), -1)
            track = [{"track_id": 1, "bbox": (x, 100, x + 60, 180)}]
            result = estimator.estimate_speed(img, track)
            speed = result.get(1)
        # 10 px/frame @ 30fps / 10 px-per-meter = 30 m/s = 108 km/h
        assert speed == pytest.approx(108.0, abs=2.0)


class TestZoneLogic:
    def test_dwell_time_tracks_continuous_occupancy(self):
        zone_logic = ZoneLogic()
        zone_logic.register_zone("zone_a", np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))

        inside = [{"track_id": 1, "bbox": (40, 40, 60, 60)}]
        outside = [{"track_id": 1, "bbox": (200, 200, 220, 220)}]

        entries = zone_logic.check_zone_entry(inside)
        assert entries["zone_a"] == [1]
        assert zone_logic.get_dwell_time("zone_a", 1) >= 0

        zone_logic.check_zone_entry(outside)
        assert zone_logic.get_dwell_time("zone_a", 1) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
