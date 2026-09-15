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

            event_logger = EventLogger(log_dir=config.log_dir)
            print("✓ Event logger initialized")

            print("\n" + "=" * 50)
            print("Controls: 'q' quit | 'p' pause | 's' screenshot | 'r' toggle recording\n")

            fps_counter = FPSCounter()
            recorder = None
            paused = False
            frame = None

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.error("Failed to read frame, camera unavailable")
                        print("✗ Failed to read frame")
                        break

                    frame_count += 1
                    frame = frame_processor.process(frame, low_light=False)

                    try:
                        detections = detector.detect(frame, conf=config.confidence_threshold)
                    except Exception:
                        logger.exception(f"Detection failed on frame {frame_count}")
                        detections = {"all": []}

                    for det in detections["all"]:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        confidence = det["confidence"]
                        class_name = det["class_name"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
