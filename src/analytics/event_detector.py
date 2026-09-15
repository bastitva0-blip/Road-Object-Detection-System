"""
Traffic analytics: density heatmap, flow direction map, near-miss/accident
detection, queue length estimation, and peak-hour detection.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x1,y1,x2,y2] boxes"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class EventDetector:
    """Detect traffic events: near-misses, accidents, queues, peak hours; build heatmaps/flow maps"""

    def __init__(self, near_miss_meters: float = 2.0, pixels_per_meter: float = 10.0,
                 accident_iou_threshold: float = 0.3, sudden_stop_ratio: float = 0.3,
                 peak_hour_window_seconds: float = 1800.0):
        """
        Initialize event detector

        Args:
            near_miss_meters: Distance threshold (meters) between two objects to flag a near-miss
            pixels_per_meter: Calibration factor (see scripts/calibrate_camera.py)
            accident_iou_threshold: Bounding-box overlap above which two tracks are "colliding"
            sudden_stop_ratio: Recent speed must drop below this fraction of prior speed to count as a sudden stop
            peak_hour_window_seconds: Rolling window used for peak-hour comparison (default 30 min)
        """
        self.near_miss_px = near_miss_meters * pixels_per_meter
        self.pixels_per_meter = pixels_per_meter
        self.accident_iou_threshold = accident_iou_threshold
        self.sudden_stop_ratio = sudden_stop_ratio
        self.peak_hour_window_seconds = peak_hour_window_seconds
        self._count_history: deque = deque()  # (timestamp, count)

    def _center(self, bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    def detect_near_miss(self, tracks: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
        """
        Detect near-miss events between tracked objects

        Args:
            tracks: Tracked objects (each needs "track_id", "bbox")

        Returns:
            List of (track_id1, track_id2, distance_meters) for pairs closer than the threshold
        """
        pairs = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                c1 = self._center(tracks[i]["bbox"])
                c2 = self._center(tracks[j]["bbox"])
                dist_px = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                if dist_px < self.near_miss_px:
                    pairs.append((tracks[i]["track_id"], tracks[j]["track_id"],
                                  round(dist_px / self.pixels_per_meter, 2)))
        return pairs

    def detect_accident(self, tracks: List[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Detect potential accidents: overlapping bounding boxes, boosted by a recent sudden stop

        Args:
            tracks: Tracked objects (each needs "track_id", "bbox", "trajectory")
            frame: Current frame (unused currently; kept for a future visual-cue signal)

        Returns:
            List of {"track_ids", "iou", "confidence"} accident candidates
        """
        stopped_tracks = set()
        for t in tracks:
            traj = t.get("trajectory", [])
            if len(traj) < 6:
                continue
            recent_speed = float(np.hypot(*np.subtract(traj[-1], traj[-3])))
            earlier_speed = float(np.hypot(*np.subtract(traj[-4], traj[-6])))
            if earlier_speed > 2 and recent_speed < earlier_speed * self.sudden_stop_ratio:
                stopped_tracks.add(t["track_id"])

        accidents = []
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                overlap = _iou(tuple(tracks[i]["bbox"]), tuple(tracks[j]["bbox"]))
                if overlap < self.accident_iou_threshold:
                    continue
                involved = {tracks[i]["track_id"], tracks[j]["track_id"]}
                confidence = 0.5 + (0.5 if involved & stopped_tracks else 0.0)
                accidents.append({
                    "track_ids": sorted(involved),
                    "iou": round(overlap, 2),
                    "confidence": confidence,
                })
        return accidents

    def estimate_queue_length(self, vehicle_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate queue length from the spread of vehicle centers along the camera's Y axis

        Args:
            vehicle_tracks: Tracked vehicles (each needs "bbox")

        Returns:
            {"queue_length_m", "vehicle_count"}
        """
        if not vehicle_tracks:
            return {"queue_length_m": 0.0, "vehicle_count": 0}
        ys = [self._center(t["bbox"])[1] for t in vehicle_tracks]
        pixel_span = max(ys) - min(ys)
        return {
            "queue_length_m": round(pixel_span / self.pixels_per_meter, 1),
            "vehicle_count": len(vehicle_tracks),
        }

    def record_vehicle_count(self, count: int, timestamp: Optional[float] = None):
        """Append a vehicle-count sample to the rolling window used by `is_peak_hour`"""
        ts = timestamp if timestamp is not None else time.time()
        self._count_history.append((ts, count))
        cutoff = ts - self.peak_hour_window_seconds
        while self._count_history and self._count_history[0][0] < cutoff:
            self._count_history.popleft()

    def is_peak_hour(self, spike_ratio: float = 1.5, recent_samples: int = 5) -> bool:
        """True if the most recent samples run well above the window's historical average"""
        if len(self._count_history) < recent_samples * 2:
            return False
        counts = [c for _, c in self._count_history]
        historical = np.mean(counts[:-recent_samples])
        recent = np.mean(counts[-recent_samples:])
        return bool(historical > 0 and recent > historical * spike_ratio)

    def generate_heatmap(self, tracks: List[Dict[str, Any]], frame_shape: Tuple[int, int],
                          grid_size: Tuple[int, int] = (20, 20)) -> np.ndarray:
        """
        Bin tracked object centers into a density grid

        Args:
            tracks: Tracked objects (each needs "bbox")
            frame_shape: (height, width) of the source frame
            grid_size: (rows, cols) of the density grid

        Returns:
            2D float32 array of per-cell object counts, shape `grid_size`
        """
        height, width = frame_shape[:2]
        gh, gw = grid_size
        density = np.zeros((gh, gw), dtype=np.float32)
        for t in tracks:
            cx, cy = self._center(t["bbox"])
            gx = min(max(int(cx / width * gw), 0), gw - 1)
            gy = min(max(int(cy / height * gh), 0), gh - 1)
            density[gy, gx] += 1
        return density

    def render_heatmap_overlay(self, frame: np.ndarray, density: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a density grid onto `frame` as a smoothed JET-colormap overlay"""
        height, width = frame.shape[:2]
        resized = cv2.resize(density, (width, height), interpolation=cv2.INTER_LINEAR)
        resized = cv2.GaussianBlur(resized, (31, 31), 0)
        peak = float(resized.max())
        normalized = (resized / peak * 255).astype(np.uint8) if peak > 0 else resized.astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)

    def compute_flow_vectors(self, tracks: List[Dict[str, Any]], frame_shape: Tuple[int, int],
                              grid_size: Tuple[int, int] = (8, 8)) -> List[Dict[str, Any]]:
        """
        Aggregate per-track velocity (from the tracker's Kalman state) into a grid of flow vectors

        Args:
            tracks: Tracked objects (each needs "bbox" and "velocity" as (vx, vy) px/frame)
            frame_shape: (height, width) of the source frame
            grid_size: (rows, cols) of the flow grid

        Returns:
            List of {"origin": (x, y), "vector": (vx, vy)} per occupied grid cell
        """
        height, width = frame_shape[:2]
        gh, gw = grid_size
        cell_h, cell_w = height / gh, width / gw

        buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for t in tracks:
            vx, vy = t.get("velocity", (0.0, 0.0))
            if vx == 0 and vy == 0:
                continue
            cx, cy = self._center(t["bbox"])
            gx = min(max(int(cx / cell_w), 0), gw - 1)
            gy = min(max(int(cy / cell_h), 0), gh - 1)
            buckets.setdefault((gx, gy), []).append((vx, vy))

        vectors = []
        for (gx, gy), vecs in buckets.items():
            avg_vx = float(np.mean([v[0] for v in vecs]))
            avg_vy = float(np.mean([v[1] for v in vecs]))
            origin = (int((gx + 0.5) * cell_w), int((gy + 0.5) * cell_h))
            vectors.append({"origin": origin, "vector": (avg_vx, avg_vy)})
        return vectors
