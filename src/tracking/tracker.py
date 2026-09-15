"""
Multi-object tracker: Kalman-filter motion model + ByteTrack-style two-stage
IOU association.

ByteTrack was chosen over DeepSORT for this phase: DeepSORT needs a deep
appearance-embedding backbone (extra model download + GPU-friendly inference),
while ByteTrack gets comparable occlusion handling from IOU + a two-stage
high/low-confidence match, and runs entirely on CPU with only numpy/scipy
(already project dependencies).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]


def _bbox_to_z(bbox: BBox) -> np.ndarray:
    """Convert [x1,y1,x2,y2] to [center_x, center_y, area, aspect_ratio]"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w * h, w / h if h != 0 else 1.0])


def _z_to_bbox(z: np.ndarray) -> BBox:
    """Convert [center_x, center_y, area, aspect_ratio] back to [x1,y1,x2,y2]"""
    cx, cy, s, r = z[0], z[1], max(z[2], 1e-6), max(z[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def iou(box_a: BBox, box_b: BBox) -> float:
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


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter over [cx, cy, area, aspect_ratio]"""

    def __init__(self, bbox: BBox):
        # State: [cx, cy, s, r, vcx, vcy, vs] (aspect ratio assumed constant)
        self.x = np.zeros(7)
        self.x[:4] = _bbox_to_z(bbox)

        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # position += velocity
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 100.0  # high initial uncertainty on velocity
        self.Q = np.eye(7) * 0.01
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0  # less trust in noisy area/aspect measurements

    def predict(self) -> BBox:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.x[2] <= 0:  # area went non-physical, zero its velocity
            self.x[6] = 0.0
        return _z_to_bbox(self.x)

    def update(self, bbox: BBox):
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def velocity_px_per_frame(self) -> Tuple[float, float]:
        return float(self.x[4]), float(self.x[5])

    def state_bbox(self) -> BBox:
        return _z_to_bbox(self.x)


class Track:
    """A tracked object: Kalman state, ID, class, and position history"""

    def __init__(self, track_id: int, bbox: BBox, class_name: str, trajectory_length: int = 30):
        self.track_id = track_id
        self.class_name = class_name
        self.kf = _KalmanBoxTracker(bbox)
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        self.age = 0
        self.confirmed = False
        self.trajectory_length = trajectory_length
        cx, cy = _bbox_to_z(bbox)[:2]
        self.trajectory: List[Tuple[float, float]] = [(float(cx), float(cy))]

    def predict(self) -> BBox:
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, bbox: BBox, class_name: str):
        self.kf.update(bbox)
        self.class_name = class_name
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        cx, cy = self.kf.x[0], self.kf.x[1]
        self.trajectory.append((float(cx), float(cy)))
        if len(self.trajectory) > self.trajectory_length:
            self.trajectory.pop(0)

    def predict_future_position(self, frames_ahead: int = 10) -> Tuple[float, float]:
        """Linear extrapolation of center position `frames_ahead` frames forward"""
        vx, vy = self.kf.velocity_px_per_frame()
        cx, cy = self.kf.x[0], self.kf.x[1]
        return float(cx + vx * frames_ahead), float(cy + vy * frames_ahead)

    def as_dict(self, confirmed: bool) -> Dict[str, Any]:
        bbox = self.kf.state_bbox()
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": np.array(bbox),
            "confirmed": confirmed,
            "trajectory": list(self.trajectory),
            "age": self.age,
            "hits": self.hits,
        }


class MultiObjectTracker:
    """ByteTrack-style multi-object tracker: two-stage IOU association + Kalman motion"""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, high_conf_threshold: float = 0.5):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames a track survives with no matching detection
            min_hits: Minimum consecutive hits before a track is "confirmed"
            iou_threshold: Minimum IOU to accept a detection-track match
            high_conf_threshold: Detections at/above this confidence are matched first
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_conf_threshold = high_conf_threshold
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections (each with "bbox", "confidence", "class_name")

        Returns:
            List of confirmed tracked objects with track IDs, bboxes, and trajectories
        """
        for track in self.tracks:
            track.predict()

        high_conf = [d for d in detections if d["confidence"] >= self.high_conf_threshold]
        low_conf = [d for d in detections if d["confidence"] < self.high_conf_threshold]

        unmatched_track_idx = list(range(len(self.tracks)))

        matched, unmatched_dets, unmatched_track_idx = self._associate(high_conf, unmatched_track_idx)
        for det_idx, track_idx in matched:
            det = high_conf[det_idx]
            self.tracks[track_idx].update(tuple(det["bbox"]), det["class_name"])

        matched2, _, unmatched_track_idx = self._associate(low_conf, unmatched_track_idx)
        for det_idx, track_idx in matched2:
            det = low_conf[det_idx]
            self.tracks[track_idx].update(tuple(det["bbox"]), det["class_name"])

        for det_idx in unmatched_dets:
            det = high_conf[det_idx]
            self.tracks.append(Track(self._next_id, tuple(det["bbox"]), det["class_name"]))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if not t.confirmed and t.hit_streak >= self.min_hits:
                t.confirmed = True
            if t.time_since_update == 0 and t.confirmed:
                results.append(t.as_dict(confirmed=True))
        return results

    def _associate(self, dets: List[Dict[str, Any]], track_idx: List[int]
                    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IOU-based Hungarian matching between detections and a subset of tracks"""
        if not dets or not track_idx:
            return [], list(range(len(dets))), track_idx

        cost = np.zeros((len(dets), len(track_idx)))
        for i, det in enumerate(dets):
            det_bbox = tuple(det["bbox"])
            for j, ti in enumerate(track_idx):
                cost[i, j] = 1.0 - iou(det_bbox, self.tracks[ti].kf.state_bbox())

        row_idx, col_idx = linear_sum_assignment(cost)

        matched: List[Tuple[int, int]] = []
        matched_dets, matched_tracks = set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= 1.0 - self.iou_threshold:
                matched.append((r, track_idx[c]))
                matched_dets.add(r)
                matched_tracks.add(track_idx[c])

        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        unmatched_tracks = [ti for ti in track_idx if ti not in matched_tracks]
        return matched, unmatched_dets, unmatched_tracks
