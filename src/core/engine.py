"""
Detection engine: owns the full per-frame pipeline (capture, preprocess, detect,
track, lane/road analysis, violation/anomaly analytics, alerts) behind one
thread-safe interface. Used by both `main.py` (desktop OpenCV window) and
`dashboard/backend.py` (FastAPI MJPEG stream) so the pipeline exists once
instead of being duplicated between the two front ends.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.config import SystemConfig
from core.video_capture import VideoCapture
from core.frame_processor import FrameProcessor
from detection.object_detector import ObjectDetector
from tracking.tracker import MultiObjectTracker
from alerts.event_logger import EventLogger
from alerts.alert_manager import AlertManager, AlertSeverity
from alerts.audio_alerts import AudioAlertPlayer
from alerts.telegram_notifier import TelegramNotifier
from road_analysis.lane_detector import LaneDetector
from road_analysis.traffic_sign_ocr import TrafficSignOCR
from road_analysis.pothole_detector import PotholeDetector
from analytics.speed_estimator import SpeedEstimator
from analytics.zone_logic import ZoneLogic, classify_traffic_light_state
from analytics.event_detector import EventDetector

logger = logging.getLogger(__name__)

_TRAFFIC_LIGHT_CLASSES = ("traffic light", "traffic_light")
_SIGN_CLASSES = ("speed_limit_or_no_entry",)
ANALYTICS_INTERVAL = 5  # run pothole/sign analytics every Nth frame, cache between
_ALERT_SEVERITY = {
    "red_light_violation": AlertSeverity.HIGH,
    "jaywalking": AlertSeverity.MEDIUM,
    "accident": AlertSeverity.CRITICAL,
    "near_miss": AlertSeverity.MEDIUM,
    "pothole": AlertSeverity.LOW,
}


class _SyntheticCamera:
    """Generates a moving-box frame instead of reading real hardware — demo/testing only"""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._frame_count = 0
        self._rng = np.random.default_rng(0)
        self._positions = self._rng.uniform(50, [width - 150, height - 200], size=(4, 2))
        self._velocities = self._rng.uniform(-3, 3, size=(4, 2))

    def is_opened(self) -> bool:
        return True

    def read(self):
        self._frame_count += 1
        frame = np.full((self.height, self.width, 3), 90, dtype=np.uint8)
        self._positions = np.clip(
            self._positions + self._velocities, 0, [self.width - 150, self.height - 200]
        )
        for x, y in self._positions:
            cv2.rectangle(frame, (int(x), int(y)), (int(x) + 100, int(y) + 150), (60, 60, 200), -1)
        cv2.putText(frame, "DEMO MODE (synthetic feed)", (10, self.height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return True, frame

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class DetectionEngine:
    """Owns the detection pipeline; thread-safe access to the latest annotated frame and stats"""

    def __init__(self, config: SystemConfig, webcam_index: int = 0, demo_mode: bool = False):
        """
        Initialize all pipeline components

        Args:
            config: Loaded system configuration
            webcam_index: Camera device index (ignored if demo_mode)
            demo_mode: Use a synthetic video feed instead of a real camera
        """
        self.config = config
        self.demo_mode = demo_mode
        self._lock = threading.Lock()
        self.frame_count = 0

        self.cap = (
            _SyntheticCamera(config.capture_width, config.capture_height) if demo_mode
            else VideoCapture(camera_index=webcam_index, width=config.capture_width, height=config.capture_height)
        )
        if not self.cap.is_opened():
            raise RuntimeError(f"Failed to open camera index {webcam_index}")

        self.frame_processor = FrameProcessor()
        self.detector = ObjectDetector(model_name=config.model_name, device="cpu")
        self.tracker = MultiObjectTracker() if config.enable_tracking else None
        self.lane_detector = LaneDetector() if config.enable_lane_detection else None

        self.speed_estimator = None
        if self.tracker is not None and config.speed_estimation.get("enabled", True):
            self.speed_estimator = SpeedEstimator(
                fps=config.fps, pixels_per_meter=config.speed_estimation.get("pixels_per_meter", 10.0)
            )

        self.zone_logic = ZoneLogic()
        for zone_name, zone_cfg in config.zones.items():
            points = zone_cfg.get("points") or []
            if zone_cfg.get("enabled") and len(points) >= 3:
                self.zone_logic.register_zone(zone_name, points)

        self.sign_ocr = TrafficSignOCR() if config.enable_analytics else None
        self.pothole_detector = PotholeDetector() if config.enable_analytics else None
        self.event_detector = EventDetector(
            pixels_per_meter=config.speed_estimation.get("pixels_per_meter", 10.0)
        ) if config.enable_analytics else None

        self.event_logger = EventLogger(log_dir=config.log_dir)
        self.alert_manager = AlertManager(cooldown_seconds=config.alert_cooldown_seconds)
        self.audio_player = AudioAlertPlayer() if config.enable_audio_alerts else None
        self.telegram = None
        if config.enable_push_notifications:
            self.telegram = TelegramNotifier(
                os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
            )
        self._register_alert_callbacks()

        self.recorder: Optional[cv2.VideoWriter] = None
        self.show_heatmap = False
        self._cached_signs: list = []
        self._cached_speed_limit: Optional[int] = None
        self._cached_anomalies: Dict[str, Any] = {"potholes": [], "debris": [], "waterlogging": []}
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_stats: Dict[str, Any] = {}
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def _register_alert_callbacks(self):
        def log_callback(severity: AlertSeverity, details: Dict[str, Any]):
            self.event_logger.log_event(details.pop("_alert_type", "alert"),
                                        {**details, "severity": severity.name})

        def audio_callback(severity: AlertSeverity, details: Dict[str, Any]):
            if self.audio_player is not None:
                self.audio_player.play(severity.name)

        def telegram_callback(severity: AlertSeverity, details: Dict[str, Any]):
            if self.telegram is not None and severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
                self.telegram.send_message(f"{severity.name}: {details}")

        for alert_type in _ALERT_SEVERITY:
            self.alert_manager.register_callback(alert_type, log_callback)
            self.alert_manager.register_callback(alert_type, audio_callback)
            self.alert_manager.register_callback(alert_type, telegram_callback)

    def _fire(self, alert_type: str, details: Dict[str, Any]):
        severity = _ALERT_SEVERITY.get(alert_type, AlertSeverity.LOW)
        self.alert_manager.fire_alert(alert_type, severity, {"_alert_type": alert_type, **details})

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self) -> bool:
        """
        Read, process, and annotate the next frame. Updates latest_frame/latest_stats.

        Returns:
            True if a frame was processed, False if the camera returned no frame
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame, camera unavailable")
            return False

        self.frame_count += 1
        frame = self.frame_processor.process(frame, low_light=False)

        try:
            detections = self.detector.detect(frame, conf=self.config.confidence_threshold)
        except Exception:
            logger.exception(f"Detection failed on frame {self.frame_count}")
            detections = {"all": [], "people": [], "vehicles": [], "animals": [], "infrastructure": []}

        tracks = self._run_tracking(frame, detections)
        self._run_lane_detection(frame)
        self._run_violation_detection(frame, detections, tracks)
        self._run_anomaly_detection(frame)
        self._run_heatmap(frame, tracks)
        self._draw_hud(frame, detections)

        if self.recorder is not None:
            self.recorder.write(frame)

        if self.event_detector is not None:
            vehicle_count = len(detections.get("vehicles", []))
            self.event_detector.record_vehicle_count(vehicle_count)

        with self._lock:
            self._latest_frame = frame
            self._latest_stats = self._build_stats(detections, tracks)

        return True

    def _run_tracking(self, frame, detections) -> list:
        if self.tracker is None:
            for det in detections["all"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return []

        try:
            tracks = self.tracker.update(detections["all"])
            speeds = self.speed_estimator.estimate_speed(frame, tracks) if self.speed_estimator else {}
            for track in tracks:
                x1, y1, x2, y2 = map(int, track["bbox"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {track['track_id']} {track['class_name']}"
                speed = speeds.get(track["track_id"])
                if speed is not None:
                    label += f" {speed:.0f} km/h"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                trajectory = track.get("trajectory", [])
                for i in range(1, len(trajectory)):
                    p1 = tuple(map(int, trajectory[i - 1]))
                    p2 = tuple(map(int, trajectory[i]))
                    cv2.line(frame, p1, p2, (0, 200, 255), 2)

            if self.zone_logic.zones:
                zone_entries = self.zone_logic.check_zone_entry(tracks)
                for zone_name, polygon in self.zone_logic.zones.items():
                    cv2.polylines(frame, [polygon], True, (200, 200, 0), 2)
                    x, y = polygon[0]
                    count = len(zone_entries.get(zone_name, []))
                    cv2.putText(frame, f"{zone_name}: {count}", (int(x), int(y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            return tracks
        except Exception:
            logger.exception(f"Tracking failed on frame {self.frame_count}")
            return []

    def _run_lane_detection(self, frame):
        if self.lane_detector is None:
            return
        try:
            lanes = self.lane_detector.detect(frame)
            markings = self.lane_detector.detect_road_markings(frame)
            for key, color in (("left_line", (255, 0, 0)), ("right_line", (0, 0, 255))):
                line = lanes.get(key)
                if line is not None:
                    x1, y1, x2, y2 = line
                    cv2.line(frame, (x1, y1), (x2, y2), color, 4)
            warning = lanes.get("departure_warning")
            if warning:
                cv2.putText(frame, f"LANE WARNING: {warning}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for x1, y1, x2, y2 in markings.get("stop_lines", []):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            for crossing in markings.get("zebra_crossings", []):
                x, y, w, h = crossing["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            for bump in markings.get("speed_bumps", []):
                x, y, w, h = bump["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            for arrow in markings.get("arrows", []):
                x, y, w, h = arrow["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        except Exception:
            logger.exception(f"Lane detection failed on frame {self.frame_count}")

    def _run_violation_detection(self, frame, detections, tracks):
        if not (self.config.enable_analytics and self.tracker is not None):
            return
        try:
            light_state = "unknown"
            for det in detections["infrastructure"]:
                if det["class_name"] in _TRAFFIC_LIGHT_CLASSES:
                    light_state = classify_traffic_light_state(frame, det["bbox"])
                    cv2.putText(frame, f"Light: {light_state}", (10, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    break

            if "stop_line" in self.zone_logic.zones:
                if light_state != "red":
                    self.zone_logic.reset_red_light_violations()
                for track_id in self.zone_logic.detect_red_light_violation(tracks, light_state):
                    self._fire("red_light_violation", {"track_id": track_id})

            for track_id in self.zone_logic.detect_jaywalking(tracks):
                self._fire("jaywalking", {"track_id": track_id})

            if self.event_detector is not None:
                for id1, id2, dist_m in self.event_detector.detect_near_miss(tracks):
                    self._fire("near_miss", {"track_ids": [id1, id2], "distance_m": dist_m})
                for accident in self.event_detector.detect_accident(tracks):
                    self._fire("accident", accident)
        except Exception:
            logger.exception(f"Violation detection failed on frame {self.frame_count}")

    def _run_anomaly_detection(self, frame):
        if not (self.sign_ocr is not None and self.pothole_detector is not None):
            return
        try:
            if self.frame_count % ANALYTICS_INTERVAL == 0:
                signs = self.sign_ocr.detect_sign(frame)["signs"]
                self._cached_signs = signs
                self._cached_speed_limit = None
                for sign in signs:
                    if sign["sign_type"] in _SIGN_CLASSES:
                        x, y, w, h = sign["bbox"]
                        inset = frame[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
                        limit = self.sign_ocr.read_speed_limit(inset)
                        if limit is not None:
                            self._cached_speed_limit = limit
                            break
                self._cached_anomalies = self.pothole_detector.detect_anomalies(frame)
                for pothole in self._cached_anomalies.get("potholes", []):
                    if pothole["severity"] == "high":
                        self._fire("pothole", {"bbox": pothole["bbox"], "severity": pothole["severity"]})

            for sign in self._cached_signs:
                x, y, w, h = sign["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 2)
            if self._cached_speed_limit is not None:
                cv2.putText(frame, f"Speed limit: {self._cached_speed_limit}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            for pothole in self._cached_anomalies.get("potholes", []):
                x, y, w, h = pothole["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for debris in self._cached_anomalies.get("debris", []):
                x, y, w, h = debris["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            for puddle in self._cached_anomalies.get("waterlogging", []):
                x, y, w, h = puddle["bbox"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        except Exception:
            logger.exception(f"Sign/pothole analytics failed on frame {self.frame_count}")

    def _run_heatmap(self, frame, tracks):
        if not (self.show_heatmap and self.event_detector is not None):
            return
        try:
            density = self.event_detector.generate_heatmap(tracks, frame.shape[:2])
            overlay = self.event_detector.render_heatmap_overlay(frame, density)
            frame[:] = overlay

            for flow in self.event_detector.compute_flow_vectors(tracks, frame.shape[:2]):
                ox, oy = flow["origin"]
                vx, vy = flow["vector"]
                scale = 8  # px/frame vectors are tiny; scale up so arrows are visible
                end = (int(ox + vx * scale), int(oy + vy * scale))
                cv2.arrowedLine(frame, (ox, oy), end, (255, 255, 255), 2, tipLength=0.4)
        except Exception:
            logger.exception(f"Heatmap rendering failed on frame {self.frame_count}")

    def _draw_hud(self, frame, detections):
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Objects: {len(detections['all'])}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _build_stats(self, detections, tracks) -> Dict[str, Any]:
        return {
            "fps": round(self.fps, 1),
            "uptime": round(time.time() - self._start_time, 1),
            "frame_count": self.frame_count,
            "total_detections": len(detections["all"]),
            "vehicle_count": len(detections.get("vehicles", [])),
            "pedestrian_count": len(detections.get("people", [])),
            "animal_count": len(detections.get("animals", [])),
            "active_tracks": len(tracks),
            "recording": self.recorder is not None,
            "recent_events": self.alert_manager.get_recent_alerts(limit=20),
        }

    @property
    def fps(self) -> float:
        elapsed = time.time() - self._start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

    # ------------------------------------------------------------------
    # External controls (used by main.py keyboard handling and the dashboard API)
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_latest_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest_stats)

    def toggle_recording(self) -> bool:
        """Start or stop recording; returns True if now recording"""
        if self.recorder is None:
            frame = self.get_latest_frame()
            if frame is None:
                return False
            rec_dir = Path(self.config.recording_dir)
            rec_dir.mkdir(parents=True, exist_ok=True)
            rec_path = rec_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            h, w = frame.shape[:2]
            self.recorder = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w, h))
            logger.info(f"Recording started: {rec_path}")
            return True
        else:
            self.recorder.release()
            self.recorder = None
            logger.info("Recording stopped")
            return False

    def toggle_heatmap(self) -> bool:
        self.show_heatmap = not self.show_heatmap
        return self.show_heatmap

    def save_screenshot(self) -> Optional[str]:
        frame = self.get_latest_frame()
        if frame is None:
            return None
        filename = f"data/screenshots/frame_{self.frame_count:06d}.png"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")
        return filename

    def update_confidence_threshold(self, value: float):
        self.config.confidence_threshold = max(0.0, min(1.0, value))

    def shutdown(self):
        """Release the camera/recorder and flush logs"""
        if self.recorder is not None:
            self.recorder.release()
        self.cap.release()
        self.event_logger.save_json_log()
        logger.info(f"Engine shutdown. Processed {self.frame_count} frames")
