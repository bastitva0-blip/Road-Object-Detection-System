"""Unit tests for traffic analytics: heatmap, flow map, near-miss, accident, queue, peak hour"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.event_detector import EventDetector


class TestNearMiss:
    def test_close_pair_flagged(self):
        ed = EventDetector(near_miss_meters=2.0, pixels_per_meter=10.0)
        tracks = [
            {"track_id": 1, "bbox": (100, 100, 140, 160)},
            {"track_id": 2, "bbox": (110, 110, 150, 170)},
        ]
        pairs = ed.detect_near_miss(tracks)
        assert len(pairs) == 1
        assert pairs[0][:2] == (1, 2)

    def test_far_pair_not_flagged(self):
        ed = EventDetector(near_miss_meters=2.0, pixels_per_meter=10.0)
        tracks = [
            {"track_id": 1, "bbox": (100, 100, 140, 160)},
            {"track_id": 2, "bbox": (500, 500, 540, 560)},
        ]
        assert ed.detect_near_miss(tracks) == []


class TestAccidentDetection:
    def test_overlapping_boxes_flagged(self):
        ed = EventDetector(accident_iou_threshold=0.3)
        tracks = [
            {"track_id": 1, "bbox": (100, 100, 160, 160), "trajectory": [(130, 130)] * 6},
            {"track_id": 2, "bbox": (110, 110, 170, 170), "trajectory": [(140, 140)] * 6},
        ]
        result = ed.detect_accident(tracks)
        assert len(result) == 1
        assert result[0]["track_ids"] == [1, 2]
        assert result[0]["confidence"] == 0.5

    def test_sudden_stop_boosts_confidence(self):
        ed = EventDetector(accident_iou_threshold=0.3)
        tracks = [
            {"track_id": 1, "bbox": (100, 100, 160, 160),
             "trajectory": [(100, 100), (110, 100), (120, 100), (130, 100), (131, 100), (131, 100)]},
            {"track_id": 2, "bbox": (110, 110, 170, 170), "trajectory": [(140, 140)] * 6},
        ]
        result = ed.detect_accident(tracks)
        assert result[0]["confidence"] == 1.0

    def test_non_overlapping_boxes_not_flagged(self):
        ed = EventDetector()
        tracks = [
            {"track_id": 1, "bbox": (0, 0, 40, 40), "trajectory": []},
            {"track_id": 2, "bbox": (500, 500, 540, 540), "trajectory": []},
        ]
        assert ed.detect_accident(tracks) == []


class TestQueueLength:
    def test_queue_length_from_spread(self):
        ed = EventDetector(pixels_per_meter=10.0)
        vehicles = [{"bbox": (100, 100, 140, 160)}, {"bbox": (100, 300, 140, 360)}]
        result = ed.estimate_queue_length(vehicles)
        assert result["vehicle_count"] == 2
        assert result["queue_length_m"] == pytest.approx(20.0, abs=0.5)

    def test_empty_tracks_zero_queue(self):
        ed = EventDetector()
        assert ed.estimate_queue_length([]) == {"queue_length_m": 0.0, "vehicle_count": 0}


class TestPeakHour:
    def test_spike_detected(self):
        ed = EventDetector(peak_hour_window_seconds=1800)
        now = time.time()
        for i in range(20):
            ed.record_vehicle_count(5, timestamp=now - (20 - i) * 10)
        for _ in range(6):
            ed.record_vehicle_count(20, timestamp=now)
        assert ed.is_peak_hour() is True

    def test_steady_traffic_not_peak(self):
        ed = EventDetector(peak_hour_window_seconds=1800)
        now = time.time()
        for i in range(20):
            ed.record_vehicle_count(5, timestamp=now - (20 - i) * 10)
        assert ed.is_peak_hour() is False

    def test_insufficient_samples_not_peak(self):
        ed = EventDetector()
        ed.record_vehicle_count(5)
        assert ed.is_peak_hour() is False


class TestHeatmap:
    def test_density_grid_bins_correctly(self):
        ed = EventDetector()
        tracks = [{"bbox": (100, 100, 140, 160)}] * 5
        density = ed.generate_heatmap(tracks, (720, 1280), grid_size=(10, 10))
        assert density.sum() == 5.0
        assert (density > 0).sum() == 1

    def test_overlay_matches_frame_shape(self):
        ed = EventDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        density = ed.generate_heatmap([{"bbox": (100, 100, 140, 160)}], (720, 1280), grid_size=(10, 10))
        overlay = ed.render_heatmap_overlay(frame, density)
        assert overlay.shape == frame.shape

    def test_empty_density_no_crash(self):
        ed = EventDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        density = np.zeros((10, 10), dtype=np.float32)
        overlay = ed.render_heatmap_overlay(frame, density)
        assert overlay.shape == frame.shape


class TestFlowVectors:
    def test_aggregates_velocity_by_cell(self):
        ed = EventDetector()
        tracks = [
            {"bbox": (100, 100, 140, 160), "velocity": (5.0, 0.0)},
            {"bbox": (105, 105, 145, 165), "velocity": (4.0, 1.0)},
        ]
        vectors = ed.compute_flow_vectors(tracks, (720, 1280), grid_size=(4, 4))
        assert len(vectors) == 1
        assert vectors[0]["vector"] == pytest.approx((4.5, 0.5))

    def test_stationary_tracks_excluded(self):
        ed = EventDetector()
        tracks = [{"bbox": (100, 100, 140, 160), "velocity": (0.0, 0.0)}]
        assert ed.compute_flow_vectors(tracks, (720, 1280)) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
