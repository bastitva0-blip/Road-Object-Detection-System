"""
Lane detection and road marking recognition using classical CV (Hough Transform).

LaneNet (deep-learning approach) is deferred: the Hough pipeline below meets the
Phase 2 FPS target (impact < 5 FPS) on CPU without a GPU or extra model download,
so it was chosen over LaneNet for this phase. `use_hough=False` remains a stub
for a future LaneNet backend.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Line = Tuple[int, int, int, int]


class LaneDetector:
    """Lane line and road marking detection using Hough Line Transform"""

    def __init__(self, use_hough: bool = True, smoothing_frames: int = 5):
        """
        Initialize lane detector

        Args:
            use_hough: Use Hough Line Transform (True) or LaneNet (False, not yet implemented)
            smoothing_frames: Number of frames to average fitted lane lines over
        """
        self.use_hough = use_hough
        self.smoothing_frames = smoothing_frames
        self._left_history: List[Tuple[float, float]] = []
        self._right_history: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Lane line detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect lane lines in frame

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with detected lanes, lane types, and departure warning
        """
        if not self.use_hough:
            logger.warning("LaneNet backend not implemented, falling back to Hough")

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        roi_edges = self._region_of_interest(edges)

        segments = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=100
        )

        left_segments, right_segments = self._split_segments(segments, width)

        left_fit = self._fit_lane(left_segments, height)
        right_fit = self._fit_lane(right_segments, height)

        left_fit = self._smooth(left_fit, self._left_history)
        right_fit = self._smooth(right_fit, self._right_history)

        left_line = self._fit_to_line(left_fit, height)
        right_line = self._fit_to_line(right_fit, height)

        return {
            "method": "hough",
            "edges": edges,
            "roi_edges": roi_edges,
            "raw_segments": segments,
            "left_line": left_line,
            "right_line": right_line,
            "left_type": self._classify_line_type(left_segments),
            "right_type": self._classify_line_type(right_segments),
            "departure_warning": self._lane_departure(left_line, right_line, width),
        }

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        """Mask edges to a trapezoid covering the road ahead of the camera"""
        height, width = edges.shape
        polygon = np.array([[
            (int(0.05 * width), height),
            (int(0.45 * width), int(0.6 * height)),
            (int(0.55 * width), int(0.6 * height)),
            (int(0.95 * width), height),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _split_segments(self, segments: Optional[np.ndarray], width: int) -> Tuple[List[Line], List[Line]]:
        """Split Hough segments into left/right lane candidates by slope and position"""
        left: List[Line] = []
        right: List[Line] = []
        if segments is None:
            return left, right

        mid_x = width / 2
        for seg in segments.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            if x2 == x1:
                continue  # vertical segment, slope undefined
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:
                continue  # near-horizontal, not a lane line
            if slope < 0 and max(x1, x2) < mid_x * 1.1:
                left.append((x1, y1, x2, y2))
            elif slope > 0 and min(x1, x2) > mid_x * 0.9:
                right.append((x1, y1, x2, y2))
        return left, right

    def _fit_lane(self, segments: List[Line], height: int) -> Optional[Tuple[float, float]]:
        """Fit x = m*y + b through segment endpoints (avoids vertical-line slope issues)"""
        if not segments:
            return None
        xs: List[int] = []
        ys: List[int] = []
        for x1, y1, x2, y2 in segments:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        m, b = np.polyfit(ys, xs, deg=1)
        return float(m), float(b)

    def _smooth(self, fit: Optional[Tuple[float, float]], history: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Average fit coefficients over recent frames to reduce jitter"""
        if fit is not None:
            history.append(fit)
        if len(history) > self.smoothing_frames:
            history.pop(0)
        if not history:
            return None
        m_avg = float(np.mean([f[0] for f in history]))
        b_avg = float(np.mean([f[1] for f in history]))
        return m_avg, b_avg

    def _fit_to_line(self, fit: Optional[Tuple[float, float]], height: int) -> Optional[Line]:
        """Convert x = m*y + b fit into a drawable line from bottom to mid-frame"""
        if fit is None:
            return None
        m, b = fit
        y1 = height
        y2 = int(height * 0.6)
        x1 = int(m * y1 + b)
        x2 = int(m * y2 + b)
        return x1, y1, x2, y2

    def _classify_line_type(self, segments: List[Line]) -> str:
        """Classify lane marking as solid or dashed based on segment fragmentation"""
        if not segments:
            return "unknown"
        if len(segments) <= 2:
            return "solid"
        lengths = [np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in segments]
        avg_len = float(np.mean(lengths))
        return "dashed" if avg_len < 80 else "solid"

    def _lane_departure(self, left_line: Optional[Line], right_line: Optional[Line], width: int) -> Optional[str]:
        """Warn if the frame center (assumed vehicle position) drifts toward a lane edge"""
        if left_line is None or right_line is None:
            return None
        lane_center = (left_line[0] + right_line[0]) / 2
        frame_center = width / 2
        offset = frame_center - lane_center
        threshold = width * 0.08
        if offset > threshold:
            return "drifting_right"
        if offset < -threshold:
            return "drifting_left"
        return None

    # ------------------------------------------------------------------
    # Road marking recognition
    # ------------------------------------------------------------------

    def detect_road_markings(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect road markings: stop lines, zebra crossings, speed bumps, arrows

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with detected markings
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        roi_mask = self._region_of_interest(white_mask)

        return {
            "stop_lines": self._detect_stop_lines(roi_mask, frame.shape[1]),
            "zebra_crossings": self._detect_zebra_crossing(roi_mask),
            "speed_bumps": self._detect_speed_bump(gray),
            "arrows": self._detect_arrows(roi_mask),
        }

    def _detect_stop_lines(self, white_mask: np.ndarray, width: int) -> List[Line]:
        """Detect wide, near-horizontal white bars (stop lines)"""
        lines = cv2.HoughLinesP(
            white_mask, 1, np.pi / 180, threshold=60,
            minLineLength=int(width * 0.25), maxLineGap=10
        )
        stop_lines: List[Line] = []
        if lines is None:
            return stop_lines
        for seg in lines.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            if x2 == x1:
                continue
            slope = abs((y2 - y1) / (x2 - x1))
            if slope < 0.2:  # near-horizontal
                stop_lines.append((x1, y1, x2, y2))
        return stop_lines

    def _detect_zebra_crossing(self, white_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detect zebra crossings as clusters of evenly spaced horizontal white stripes"""
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stripes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > h * 3 and w > white_mask.shape[1] * 0.05:
                stripes.append((x, y, w, h))

        if len(stripes) < 3:
            return []

        stripes.sort(key=lambda s: s[1])
        gaps = [stripes[i + 1][1] - stripes[i][1] for i in range(len(stripes) - 1)]
        if not gaps:
            return []
        mean_gap = np.mean(gaps)
        if mean_gap == 0 or np.std(gaps) / mean_gap > 0.5:
            return []  # irregular spacing, not a crossing pattern

        xs = [s[0] for s in stripes]
        ws = [s[2] for s in stripes]
        ys = [s[1] for s in stripes]
        hs = [s[3] for s in stripes]
        bbox = (min(xs), min(ys), max(x + w for x, w in zip(xs, ws)) - min(xs),
                max(y + h for y, h in zip(ys, hs)) - min(ys))
        return [{"bbox": bbox, "stripe_count": len(stripes)}]

    def _detect_speed_bump(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect speed bumps as thick horizontal bands in the lower part of the frame"""
        height, width = gray.shape
        lower = gray[int(height * 0.7):, :]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(lower, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(closed, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bumps = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > width * 0.4 and 5 < h < 40:
                bumps.append({"bbox": (x, y + int(height * 0.7), w, h)})
        return bumps

    def detect_road_edges(self, frame: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Fallback road-edge estimate for unmarked rural roads (no painted lane lines),
        using the outermost strong edges near the bottom of the ROI.

        Args:
            frame: Input frame (BGR)

        Returns:
            Dict with left_edge_x / right_edge_x in pixels, or None if no edges found
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        roi = self._region_of_interest(edges)

        scan_row = roi[int(height * 0.92), :]
        cols = np.where(scan_row > 0)[0]
        if len(cols) < 2:
            return None
        return {"left_edge_x": int(cols.min()), "right_edge_x": int(cols.max())}

    def _detect_arrows(self, white_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Detect arrow-shaped road markings via contour polygon approximation"""
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arrows = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(c)
            if 6 <= len(approx) <= 9 and h > w:
                arrows.append({"bbox": (x, y, w, h), "vertices": len(approx)})
        return arrows
