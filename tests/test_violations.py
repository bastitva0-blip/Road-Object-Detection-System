"""Unit tests for traffic light classification, red light violations, and jaywalking"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.zone_logic import ZoneLogic, classify_traffic_light_state


class TestTrafficLightClassification:
    def test_red_bulb_lit(self):
        import cv2
        img = np.full((150, 50, 3), 30, dtype=np.uint8)
        cv2.circle(img, (25, 25), 18, (0, 0, 255), -1)
        assert classify_traffic_light_state(img, (0, 0, 50, 150)) == "red"

    def test_green_bulb_lit(self):
        import cv2
        img = np.full((150, 50, 3), 30, dtype=np.uint8)
        cv2.circle(img, (25, 125), 18, (0, 255, 0), -1)
        assert classify_traffic_light_state(img, (0, 0, 50, 150)) == "green"

    def test_unlit_light_is_unknown(self):
        img = np.full((150, 50, 3), 30, dtype=np.uint8)
        assert classify_traffic_light_state(img, (0, 0, 50, 150)) == "unknown"

    def test_empty_crop_is_unknown(self):
        img = np.full((150, 50, 3), 30, dtype=np.uint8)
        assert classify_traffic_light_state(img, (200, 200, 250, 300)) == "unknown"


class TestRedLightViolation:
    def _zone(self):
        zone_logic = ZoneLogic()
        zone_logic.register_zone("stop_line", np.array([[500, 600], [700, 600], [700, 680], [500, 680]]))
        return zone_logic

    def test_vehicle_in_zone_on_red_flagged_once(self):
        zone_logic = self._zone()
        vehicles = [{"track_id": 1, "bbox": (600, 610, 660, 670), "class_name": "car"}]

        assert zone_logic.detect_red_light_violation(vehicles, "red") == [1]
        assert zone_logic.detect_red_light_violation(vehicles, "red") == []  # already flagged

    def test_no_violation_on_green(self):
        zone_logic = self._zone()
        vehicles = [{"track_id": 1, "bbox": (600, 610, 660, 670), "class_name": "car"}]
        assert zone_logic.detect_red_light_violation(vehicles, "green") == []

    def test_pedestrians_excluded_from_red_light_check(self):
        zone_logic = self._zone()
        people = [{"track_id": 1, "bbox": (600, 610, 660, 670), "class_name": "person"}]
        assert zone_logic.detect_red_light_violation(people, "red") == []

    def test_reset_rearms_detection(self):
        zone_logic = self._zone()
        vehicles = [{"track_id": 1, "bbox": (600, 610, 660, 670), "class_name": "car"}]
        zone_logic.detect_red_light_violation(vehicles, "red")
        zone_logic.reset_red_light_violations()
        assert zone_logic.detect_red_light_violation(vehicles, "red") == [1]

    def test_no_stop_line_zone_returns_empty(self):
        zone_logic = ZoneLogic()
        vehicles = [{"track_id": 1, "bbox": (600, 610, 660, 670), "class_name": "car"}]
        assert zone_logic.detect_red_light_violation(vehicles, "red") == []


class TestJaywalking:
    def _zone(self):
        zone_logic = ZoneLogic()
        zone_logic.register_zone("road_zone", np.array([[0, 0], [200, 0], [200, 200], [0, 200]]))
        zone_logic.register_zone("zebra_crossing", np.array([[50, 50], [150, 50], [150, 100], [50, 100]]))
        return zone_logic

    def test_pedestrian_outside_crossing_flagged(self):
        zone_logic = self._zone()
        people = [{"track_id": 2, "bbox": (10, 150, 20, 160), "class_name": "person"}]
        assert zone_logic.detect_jaywalking(people) == [2]

    def test_pedestrian_inside_crossing_not_flagged(self):
        zone_logic = self._zone()
        people = [{"track_id": 1, "bbox": (60, 60, 70, 70), "class_name": "person"}]
        assert zone_logic.detect_jaywalking(people) == []

    def test_pedestrian_outside_road_not_flagged(self):
        zone_logic = self._zone()
        people = [{"track_id": 3, "bbox": (300, 300, 310, 310), "class_name": "person"}]
        assert zone_logic.detect_jaywalking(people) == []

    def test_vehicles_excluded(self):
        zone_logic = self._zone()
        vehicles = [{"track_id": 4, "bbox": (10, 150, 20, 160), "class_name": "car"}]
        assert zone_logic.detect_jaywalking(vehicles) == []

    def test_missing_zones_returns_empty(self):
        zone_logic = ZoneLogic()
        people = [{"track_id": 1, "bbox": (10, 150, 20, 160), "class_name": "person"}]
        assert zone_logic.detect_jaywalking(people) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
