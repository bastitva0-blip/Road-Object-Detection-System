"""
Vehicle speed estimation using sparse optical flow (Lucas-Kanade) + calibration

Approach: for each tracked bounding box, find good-to-track corners inside it in
the previous frame, follow them into the current frame with pyramidal LK optical
flow, and convert the average pixel displacement to km/h using a pixels-per-meter
calibration factor (see scripts/calibrate_camera.py). Perspective distortion is
NOT corrected here (would need a homography from a calibration script/checkerboard
capture); pixels_per_meter is a scene-average approximation, most accurate near
the calibration line and less accurate near the horizon.
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """Estimate vehicle speed from frame-to-frame optical flow inside tracked boxes"""

    def __init__(self, fps: int = 30, pixels_per_meter: float = 10.0,
                 max_corners: int = 20, smoothing: float = 0.6):
        """
        Initialize speed estimator

        Args:
            fps: Frames per second of the video source
            pixels_per_meter: Calibration factor (see scripts/calibrate_camera.py)
            max_corners: Max optical-flow feature points tracked per box
            smoothing: EMA weight (0-1) applied to new speed readings per track
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.max_corners = max_corners
        self.smoothing = smoothing
        self.prev_gray: np.ndarray = None
        self._smoothed_speeds: Dict[int, float] = {}

        self._lk_params = dict(
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def estimate_speed(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Estimate speed for tracked objects

        Args:
            frame: Current frame (BGR)
            tracks: Tracked objects (each needs "track_id" and "bbox")

        Returns:
            Dictionary mapping track_id to estimated speed in km/h
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        speeds: Dict[int, float] = {}

        if self.prev_gray is not None:
            height, width = gray.shape
            for track in tracks:
                track_id = track["track_id"]
                x1, y1, x2, y2 = map(int, track["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                speed_kmh = self._estimate_box_speed(x1, y1, x2, y2, gray)
                if speed_kmh is None:
                    continue

                prev = self._smoothed_speeds.get(track_id, speed_kmh)
                smoothed = self.smoothing * speed_kmh + (1 - self.smoothing) * prev
                self._smoothed_speeds[track_id] = smoothed
                speeds[track_id] = smoothed

        self.prev_gray = gray
        return speeds

    def _estimate_box_speed(self, x1: int, y1: int, x2: int, y2: int, gray: np.ndarray) -> float:
        """Track corners inside [x1,y1,x2,y2] via LK optical flow, return speed in km/h"""
        roi_prev = self.prev_gray[y1:y2, x1:x2]
        if roi_prev.size == 0:
            return None

        corners = cv2.goodFeaturesToTrack(roi_prev, maxCorners=self.max_corners,
                                          qualityLevel=0.3, minDistance=5)
        if corners is None or len(corners) == 0:
            return None

        corners[:, :, 0] += x1
        corners[:, :, 1] += y1

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, corners, None, **self._lk_params
        )
        if next_pts is None:
            return None

        status = status.reshape(-1)
        good_old = corners.reshape(-1, 2)[status == 1]
        good_new = next_pts.reshape(-1, 2)[status == 1]
        if len(good_old) == 0:
            return None

        displacement = np.mean(good_new - good_old, axis=0)
        pixel_speed_per_sec = np.hypot(displacement[0], displacement[1]) * self.fps
        meters_per_sec = pixel_speed_per_sec / self.pixels_per_meter
        return float(meters_per_sec * 3.6)  # km/h

    def reset(self):
        """Clear optical flow state (call on scene cut or camera reconnect)"""
        self.prev_gray = None
        self._smoothed_speeds.clear()
