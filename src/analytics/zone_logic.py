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
        """Detect people crossing outside designated crossings"""
        # Placeholder for Phase 4
        return []

    def detect_red_light_violation(self, detections: List[Dict[str, Any]], traffic_light_state: str) -> List[int]:
        """Detect vehicles crossing on red light"""
        # Placeholder for Phase 4
        return []
