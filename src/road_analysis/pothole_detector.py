"""
Pothole, debris, and road damage detection

Architecture decision (Phase 2 prep): U-Net segmentation, not YOLOv8 detection or
ResNet+FCN, because potholes have irregular shapes/sizes that boxes fit poorly and
U-Net has a smaller footprint than ResNet+FCN for near-real-time CPU inference.
Training happens once the Roboflow pothole dataset is downloaded (needs an
external account, not available here). Until then, this module uses classical CV
heuristics (dark/anomalous blob detection) as a working stand-in — real accuracy
depends heavily on lighting/road-surface variation and needs validation against
labeled data before being trusted for anything beyond an early-warning demo.

Speed bump detection lives in `LaneDetector.detect_road_markings` (Phase 2) —
not duplicated here.

All detection runs on a cropped road ROI (not the full frame + mask) to keep
per-frame cost down — see scripts/benchmark_anomaly_detection.py.
"""

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PotholeDetector:
    """Pothole, debris, and waterlogging detection via classical CV heuristics"""

    def __init__(self, roi_top_fraction: float = 0.55):
        """
        Initialize pothole detector

        Args:
            roi_top_fraction: Fraction of frame height above which is ignored
                (sky/horizon), assuming a windshield-mounted forward camera
        """
        self.roi_top_fraction = roi_top_fraction

    def detect_anomalies(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect potholes, debris, and waterlogging in the road surface

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with detected anomalies (see module docstring re: speed_bumps)
        """
        height = frame.shape[0]
        roi_top = int(height * self.roi_top_fraction)
        roi_frame = frame[roi_top:, :]
        if roi_frame.size == 0:
            return {"potholes": [], "debris": [], "waterlogging": [], "speed_bumps": []}
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        potholes = self._detect_potholes(gray_roi)
        debris = self._detect_debris(roi_frame)
        waterlogging = self._detect_waterlogging(roi_frame, gray_roi)

        for item in potholes + debris + waterlogging:
            x, y, w, h = item["bbox"]
            item["bbox"] = (x, y + roi_top, w, h)

        return {
            "potholes": potholes,
            "debris": debris,
            "waterlogging": waterlogging,
            "speed_bumps": [],  # see LaneDetector.detect_road_markings (Phase 2)
        }

    def _detect_potholes(self, gray_roi: np.ndarray) -> List[Dict[str, Any]]:
        """Dark, roughly round blobs noticeably darker than the surrounding road"""
        road_mean = float(np.mean(gray_roi))

        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 0)
        dark_thresh = max(0, road_mean - 35)
        _, dark_mask = cv2.threshold(blurred, dark_thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        potholes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 150:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h != 0 else 0
            if not (0.4 <= aspect <= 2.5):
                continue  # too elongated to be a pothole (likely a shadow/crack line)

            region_mean = float(np.mean(gray_roi[y:y + h, x:x + w]))
            darkness = road_mean - region_mean
            potholes.append({
                "bbox": (x, y, w, h),
                "darkness": round(darkness, 1),
                "severity": self._classify_severity(area, darkness, gray_roi.shape),
            })
        return potholes

    def _detect_debris(self, roi_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Small blobs whose color deviates from the road's dominant (usually gray) color"""
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        median_sat = float(np.median(hsv[:, :, 1]))

        saturation = hsv[:, :, 1].astype(np.int16)
        anomaly_mask = (np.abs(saturation - median_sat) > 40).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debris = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (20 <= area <= 800):
                continue  # too small to be real, or too big to be loose debris
            x, y, w, h = cv2.boundingRect(c)
            debris.append({"bbox": (x, y, w, h)})
        return debris

    def _detect_waterlogging(self, roi_frame: np.ndarray, gray_roi: np.ndarray) -> List[Dict[str, Any]]:
        """Large, smooth (low-texture), bright/grayish-blue patches typical of pooled water"""
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        low_sat = hsv[:, :, 1] < 60
        bright = hsv[:, :, 2] > 120

        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F, ksize=3)
        smooth = cv2.blur(np.abs(laplacian), (9, 9)) < 8

        water_mask = (low_sat & bright & smooth).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        puddles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            puddles.append({"bbox": (x, y, w, h), "area": int(area)})
        return puddles

    def _classify_severity(self, area: float, darkness: float, roi_shape: Tuple[int, int]) -> str:
        """Low/medium/high from relative size and how much darker than the road it is"""
        roi_area = roi_shape[0] * roi_shape[1]
        area_ratio = area / roi_area

        score = 0
        score += 2 if area_ratio > 0.005 else (1 if area_ratio > 0.001 else 0)
        score += 2 if darkness > 60 else (1 if darkness > 35 else 0)

        if score >= 3:
            return "high"
        if score >= 1:
            return "medium"
        return "low"
