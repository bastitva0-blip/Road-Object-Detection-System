"""
Main entry point for the Road Object Detection System
"""

import argparse
import logging
import time
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import SystemConfig
from core.video_capture import VideoCapture
from core.frame_processor import FrameProcessor
from detection.object_detector import ObjectDetector
from tracking.tracker import MultiObjectTracker
from alerts.event_logger import EventLogger
from road_analysis.lane_detector import LaneDetector
from analytics.speed_estimator import SpeedEstimator
from analytics.zone_logic import ZoneLogic, classify_traffic_light_state
from road_analysis.traffic_sign_ocr import TrafficSignOCR
from road_analysis.pothole_detector import PotholeDetector

_TRAFFIC_LIGHT_CLASSES = ("traffic light", "traffic_light")
_SIGN_CLASSES = ("speed_limit_or_no_entry",)
ANALYTICS_INTERVAL = 5  # run pothole/sign analytics every Nth frame, cache between

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str):
    """Configure root logging to file and console"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path / "app.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class FPSCounter:
    """Rolling FPS counter"""

    def __init__(self, window: int = 30):
        self.window = window
        self.timestamps = []
        self.fps = 0.0

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            self.fps = (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0
        return self.fps


def _draw_lanes(frame, lanes: dict, markings: dict):
    """Overlay lane lines, lane-departure warning, and road markings onto frame"""
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
        cv2.putText(frame, "Zebra crossing", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for bump in markings.get("speed_bumps", []):
        x, y, w, h = bump["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(frame, "Speed bump", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    for arrow in markings.get("arrows", []):
        x, y, w, h = arrow["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)


def _draw_tracks(frame, tracks: list, speeds: dict):
    """Draw tracked boxes with ID/speed labels and trajectory trails"""
    for track in tracks:
        x1, y1, x2, y2 = map(int, track["bbox"])
        track_id = track["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {track_id} {track['class_name']}"
        speed = speeds.get(track_id)
        if speed is not None:
            label += f" {speed:.0f} km/h"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        trajectory = track.get("trajectory", [])
        for i in range(1, len(trajectory)):
            p1 = tuple(map(int, trajectory[i - 1]))
            p2 = tuple(map(int, trajectory[i]))
            cv2.line(frame, p1, p2, (0, 200, 255), 2)


def _draw_signs(frame, signs: list, speed_limit: int = None):
    """Draw traffic sign candidate boxes and any OCR-read speed limit"""
    for sign in signs:
        x, y, w, h = sign["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 2)
        cv2.putText(frame, sign["sign_type"], (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
    if speed_limit is not None:
        cv2.putText(frame, f"Speed limit: {speed_limit}", (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)


def _draw_anomalies(frame, anomalies: dict):
    """Draw pothole/debris/waterlogging overlays"""
    for pothole in anomalies.get("potholes", []):
        x, y, w, h = pothole["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"Pothole ({pothole['severity']})", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for debris in anomalies.get("debris", []):
        x, y, w, h = debris["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
    for puddle in anomalies.get("waterlogging", []):
        x, y, w, h = puddle["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(frame, "Waterlogging", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def _draw_zones(frame, zone_logic: ZoneLogic, zone_entries: dict):
    """Draw registered zone polygons with current occupant counts"""
    for zone_name, polygon in zone_logic.zones.items():
        cv2.polylines(frame, [polygon], isClosed=True, color=(200, 200, 0), thickness=2)
        count = len(zone_entries.get(zone_name, []))
        x, y = polygon[0]
        cv2.putText(frame, f"{zone_name}: {count}", (int(x), int(y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)


def main():
    """Main application loop"""
    parser = argparse.ArgumentParser(description="Road Object Detection System")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Config file path")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    print("🚗 Road Object Detection System")
    print("=" * 50)

    # Load configuration
    try:
        config = SystemConfig.from_yaml(args.config)
        print(f"✓ Configuration loaded from {args.config}")
    except FileNotFoundError:
        print(f"⚠ Config file not found, using default settings")
        config = SystemConfig()

    setup_logging(config.log_dir)
    logger.info("Starting Road Object Detection System")

    # Initialize components
    print("\nInitializing components...")

    frame_count = 0
    event_logger = None

    try:
        with VideoCapture(camera_index=args.webcam, width=config.capture_width, height=config.capture_height) as cap:
            if not cap.is_opened():
                logger.error(f"Failed to open camera index {args.webcam}")
                print("✗ Failed to open camera")
                return
            print("✓ Camera initialized")

            frame_processor = FrameProcessor()
            print("✓ Frame processor initialized")

            detector = ObjectDetector(model_name=config.model_name, device="cpu")
            print("✓ Detector initialized")

            tracker = MultiObjectTracker() if config.enable_tracking else None
            print("✓ Tracker initialized" if tracker else "⚠ Tracking disabled")

            lane_detector = LaneDetector() if config.enable_lane_detection else None
            print("✓ Lane detector initialized" if lane_detector else "⚠ Lane detection disabled")

            speed_estimator = None
            if tracker is not None and config.speed_estimation.get("enabled", True):
                speed_estimator = SpeedEstimator(
                    fps=config.fps,
                    pixels_per_meter=config.speed_estimation.get("pixels_per_meter", 10.0),
                )
            print("✓ Speed estimator initialized" if speed_estimator else "⚠ Speed estimation disabled")

            zone_logic = ZoneLogic()
            for zone_name, zone_cfg in config.zones.items():
                points = zone_cfg.get("points") or []
                if zone_cfg.get("enabled") and len(points) >= 3:
                    zone_logic.register_zone(zone_name, points)
            print(f"✓ Zone logic initialized ({len(zone_logic.zones)} zone(s))")

            sign_ocr = TrafficSignOCR() if config.enable_analytics else None
            pothole_detector = PotholeDetector() if config.enable_analytics else None
            print("✓ Traffic sign/pothole analytics initialized" if config.enable_analytics
                  else "⚠ Sign/pothole analytics disabled")

            event_logger = EventLogger(log_dir=config.log_dir)
            print("✓ Event logger initialized")

            print("\n" + "=" * 50)
            print("Controls: 'q' quit | 'p' pause | 's' screenshot | 'r' toggle recording\n")

            fps_counter = FPSCounter()
            recorder = None
            paused = False
            frame = None
            cached_signs: list = []
            cached_speed_limit = None
            cached_anomalies: dict = {"potholes": [], "debris": [], "waterlogging": []}

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to read frame, camera unavailable")
                        print("✗ Failed to read frame")
                        break

                    frame_count += 1
                    frame = frame_processor.process(frame, low_light=False)
                    tracks = []

                    try:
                        detections = detector.detect(frame, conf=config.confidence_threshold)
                    except Exception:
                        logger.exception(f"Detection failed on frame {frame_count}")
                        detections = {"all": []}

                    if tracker is not None:
                        try:
                            tracks = tracker.update(detections["all"])
                            speeds = speed_estimator.estimate_speed(frame, tracks) if speed_estimator else {}
                            _draw_tracks(frame, tracks, speeds)
                            if zone_logic.zones:
                                zone_entries = zone_logic.check_zone_entry(tracks)
                                _draw_zones(frame, zone_logic, zone_entries)
                        except Exception:
                            logger.exception(f"Tracking failed on frame {frame_count}")
                    else:
                        for det in detections["all"]:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            confidence = det["confidence"]
                            class_name = det["class_name"]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if lane_detector is not None:
                        try:
                            lanes = lane_detector.detect(frame)
                            markings = lane_detector.detect_road_markings(frame)
                            _draw_lanes(frame, lanes, markings)
                        except Exception:
                            logger.exception(f"Lane detection failed on frame {frame_count}")

                    if config.enable_analytics and tracker is not None:
                        try:
                            light_state = "unknown"
                            for det in detections["infrastructure"]:
                                if det["class_name"] in _TRAFFIC_LIGHT_CLASSES:
                                    light_state = classify_traffic_light_state(frame, det["bbox"])
                                    cv2.putText(frame, f"Light: {light_state}", (10, 160),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                    break

                            if "stop_line" in zone_logic.zones:
                                if light_state != "red":
                                    zone_logic.reset_red_light_violations()
                                for track_id in zone_logic.detect_red_light_violation(tracks, light_state):
                                    event_logger.log_event("red_light_violation", {"track_id": track_id})
                                    logger.warning(f"Red light violation: track {track_id}")

                            for track_id in zone_logic.detect_jaywalking(tracks):
                                event_logger.log_event("jaywalking", {"track_id": track_id})
                                logger.warning(f"Jaywalking: track {track_id}")
                        except Exception:
                            logger.exception(f"Violation detection failed on frame {frame_count}")

                    if sign_ocr is not None and pothole_detector is not None:
                        try:
                            if frame_count % ANALYTICS_INTERVAL == 0:
                                signs = sign_ocr.detect_sign(frame)["signs"]
                                cached_signs = signs
                                cached_speed_limit = None
                                for sign in signs:
                                    if sign["sign_type"] in _SIGN_CLASSES:
                                        x, y, w, h = sign["bbox"]
                                        inset = frame[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
                                        limit = sign_ocr.read_speed_limit(inset)
                                        if limit is not None:
                                            cached_speed_limit = limit
                                            break
                                cached_anomalies = pothole_detector.detect_anomalies(frame)
                            _draw_signs(frame, cached_signs, cached_speed_limit)
                            _draw_anomalies(frame, cached_anomalies)
                        except Exception:
                            logger.exception(f"Sign/pothole analytics failed on frame {frame_count}")

                    fps = fps_counter.tick()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Objects: {len(detections['all'])}", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if recorder is not None:
                        recorder.write(frame)

                    if frame_count % 30 == 0:
                        logger.info(f"Frame {frame_count}: FPS={fps:.1f}, objects={len(detections['all'])}")

                if frame is not None:
                    display = frame.copy()
                    if paused:
                        cv2.putText(display, "PAUSED", (10, 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Road Object Detection", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")
                elif key == ord('s') and frame is not None:
                    filename = f"data/screenshots/frame_{frame_count:06d}.png"
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(filename, frame)
                    logger.info(f"Screenshot saved: {filename}")
                    print(f"Screenshot saved: {filename}")
                elif key == ord('r') and frame is not None:
                    if recorder is None:
                        rec_dir = Path(config.recording_dir)
                        rec_dir.mkdir(parents=True, exist_ok=True)
                        rec_path = rec_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                        h, w = frame.shape[:2]
                        recorder = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w, h))
                        logger.info(f"Recording started: {rec_path}")
                        print(f"Recording started: {rec_path}")
                    else:
                        recorder.release()
                        recorder = None
                        logger.info("Recording stopped")
                        print("Recording stopped")

            if recorder is not None:
                recorder.release()
    except Exception:
        logger.exception("Unhandled error in main loop")
        raise
    finally:
        logger.info(f"Shutdown gracefully. Processed {frame_count} frames")
        cv2.destroyAllWindows()

    print(f"\nShutdown gracefully. Processed {frame_count} frames")
    if event_logger is not None:
        event_logger.save_json_log()
        print("Logs saved")


if __name__ == "__main__":
    main()
