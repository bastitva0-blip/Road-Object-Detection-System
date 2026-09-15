"""
Zone-based detection logic: zone membership, dwell time, and trajectory analytics.

Red light violation / jaywalking detection (Phase 4) build on top of the zone
membership tracking implemented here.
"""

import time
from typing import Any, Dict, List

import cv2
import numpy as np


class ZoneLogic:
    """Zone-based event detection and dwell-time tracking"""

    def __init__(self):
        """Initialize zone logic"""
        self.zones: Dict[str, np.ndarray] = {}  # name -> polygon points, shape (N, 2)
        self._occupancy: Dict[str, Dict[int, float]] = {}  # zone -> {track_id: entry_time}
        self._red_light_violators: set = set()

    def register_zone(self, zone_name: str, polygon: np.ndarray):
        """
        Register a detection zone

        Args:
            zone_name: Name of the zone
            polygon: Zone boundary as array of (x, y) points
        """
        self.zones[zone_name] = np.array(polygon, dtype=np.int32)
        self._occupancy.setdefault(zone_name, {})

    def _center(self, bbox) -> tuple:
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    def point_in_zone(self, point: tuple, zone_name: str) -> bool:
        """Check whether a point lies inside a registered zone"""
        polygon = self.zones.get(zone_name)
        if polygon is None or len(polygon) < 3:
            return False
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def check_zone_entry(self, tracks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Check which tracked objects are currently inside each zone

        Args:
            tracks: List of tracked objects (each needs "track_id" and "bbox")

        Returns:
            Dictionary mapping zone names to lists of track IDs inside them
        """
        zone_entries: Dict[str, List[int]] = {name: [] for name in self.zones}
        now = time.time()

        for zone_name, polygon in self.zones.items():
            if len(polygon) < 3:
                continue
            occupancy = self._occupancy.setdefault(zone_name, {})
            currently_inside = set()

            for track in tracks:
                track_id = track["track_id"]
                center = self._center(track["bbox"])
                if self.point_in_zone(center, zone_name):
                    zone_entries[zone_name].append(track_id)
                    currently_inside.add(track_id)
                    occupancy.setdefault(track_id, now)

            for track_id in list(occupancy.keys()):
                if track_id not in currently_inside:
                    del occupancy[track_id]

        return zone_entries

    def get_dwell_time(self, zone_name: str, track_id: int) -> float:
        """Seconds a track has continuously spent inside a zone (0 if not present)"""
        entry_time = self._occupancy.get(zone_name, {}).get(track_id)
        if entry_time is None:
            return 0.0
        return time.time() - entry_time

    def detect_jaywalking(self, detections: List[Dict[str, Any]]) -> List[int]:
        """
        Detect pedestrians inside the road but outside a designated crossing.

        Requires "road_zone" and "zebra_crossing" zones to be registered; returns
        [] otherwise (nothing to check against).

        Args:
            detections: Tracked people (each needs "track_id", "bbox", "class_name")

        Returns:
            Track IDs of pedestrians in the road but outside the crossing
        """
        if "road_zone" not in self.zones or "zebra_crossing" not in self.zones:
            return []

        violators = []
        for det in detections:
            if det.get("class_name") not in _PEDESTRIAN_CLASSES:
                continue
            center = self._center(det["bbox"])
            if self.point_in_zone(center, "road_zone") and not self.point_in_zone(center, "zebra_crossing"):
                violators.append(det["track_id"])
        return violators

    def detect_red_light_violation(self, detections: List[Dict[str, Any]], traffic_light_state: str) -> List[int]:
        """
        Detect vehicles crossing the stop line while the light is red.

        Requires a "stop_line" zone registered as the intersection area beyond
        the line (a thin line has ~zero area and is unreliable for point-in-polygon).
        Each track fires at most once per red phase; call `reset_red_light_violations()`
        when the light turns green to re-arm.

        Args:
            detections: Tracked vehicles (each needs "track_id", "bbox", "class_name")
            traffic_light_state: "red", "yellow", "green", or "unknown"

        Returns:
            Track IDs of vehicles violating the red light this call
        """
        if traffic_light_state != "red" or "stop_line" not in self.zones:
            return []

        violators = []
        for det in detections:
            if det.get("class_name") not in _VEHICLE_CLASSES:
                continue
            track_id = det["track_id"]
            if track_id in self._red_light_violators:
                continue  # already flagged this red phase
            center = self._center(det["bbox"])
            if self.point_in_zone(center, "stop_line"):
                violators.append(track_id)
                self._red_light_violators.add(track_id)
        return violators

    def reset_red_light_violations(self):
        """Re-arm red light violation detection (call when the light turns green)"""
        self._red_light_violators.clear()


def classify_traffic_light_state(frame: np.ndarray, bbox) -> str:
    """
    Classify a cropped traffic-light detection as red/yellow/green by comparing
    "litness" (saturation x brightness within that color's hue range) across the
    top/middle/bottom thirds of the box, matching the standard vertical layout.

    Args:
        frame: Full frame (BGR)
        bbox: Traffic light bounding box (x1, y1, x2, y2)

    Returns:
        "red", "yellow", "green", or "unknown"
    """
    x1, y1, x2, y2 = map(int, bbox)
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv.shape[0]
    thirds = {
        "red": hsv[0:h // 3, :],
        "yellow": hsv[h // 3: 2 * h // 3, :],
        "green": hsv[2 * h // 3:, :],
    }
    hue_ranges = {
        "red": _RED_HUE_RANGES,
        "yellow": [((20, 70, 70), (35, 255, 255))],
        "green": [((40, 70, 70), (90, 255, 255))],
    }

    scores = {}
    for color, region in thirds.items():
        if region.size == 0:
            scores[color] = 0.0
            continue
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        for lower, upper in hue_ranges[color]:
            mask |= cv2.inRange(region, np.array(lower), np.array(upper))
        scores[color] = float(np.mean(mask > 0))

    best = max(scores, key=scores.get)
    return best if scores[best] > 0.15 else "unknown"


_PEDESTRIAN_CLASSES = {"person", "cyclist", "motorcyclist", "child"}
_VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "auto", "bicycle",
    "tempo", "tractor", "ambulance", "police", "fire_truck",
}
_RED_HUE_RANGES = [((0, 70, 70), (10, 255, 255)), ((170, 70, 70), (180, 255, 255))]
