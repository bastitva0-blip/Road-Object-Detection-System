"""
Main entry point for the Road Object Detection System (desktop OpenCV window).

The actual pipeline lives in core/engine.py (DetectionEngine), shared with the
web dashboard (dashboard/backend.py) — this file is just the cv2.imshow loop
and keyboard handling on top of it.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import SystemConfig
from core.engine import DetectionEngine

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


def main():
    """Main application loop"""
    parser = argparse.ArgumentParser(description="Road Object Detection System")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Config file path")
    parser.add_argument("--demo", action="store_true", help="Run with a synthetic video feed instead of a webcam")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    print("🚗 Road Object Detection System")
    print("=" * 50)

    try:
        config = SystemConfig.from_yaml(args.config)
        print(f"✓ Configuration loaded from {args.config}")
    except FileNotFoundError:
        print("⚠ Config file not found, using default settings")
        config = SystemConfig()

    setup_logging(config.log_dir)
    logger.info("Starting Road Object Detection System")

    print("\nInitializing components...")
    try:
        engine = DetectionEngine(config, webcam_index=args.webcam, demo_mode=args.demo)
    except RuntimeError as exc:
        logger.error(str(exc))
        print(f"✗ {exc}")
        return
    print("✓ Engine initialized" + (" (demo mode: synthetic feed)" if args.demo else ""))

    print("\n" + "=" * 50)
    print("Controls: 'q' quit | 'p' pause | 's' screenshot | 'r' toggle recording | 'h' toggle heatmap\n")

    paused = False
    try:
        while True:
            if not paused:
                if not engine.process_frame():
                    print("✗ Failed to read frame")
                    break
                if engine.frame_count % 30 == 0:
                    stats = engine.get_latest_stats()
                    logger.info(f"Frame {engine.frame_count}: FPS={stats['fps']:.1f}, "
                                f"objects={stats['total_detections']}")

            frame = engine.get_latest_frame()
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
            elif key == ord('s'):
                filename = engine.save_screenshot()
                if filename:
                    print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                now_recording = engine.toggle_recording()
                print("Recording started" if now_recording else "Recording stopped")
            elif key == ord('h'):
                now_showing = engine.toggle_heatmap()
                print("Heatmap on" if now_showing else "Heatmap off")
    except Exception:
        logger.exception("Unhandled error in main loop")
        raise
    finally:
        engine.shutdown()
        cv2.destroyAllWindows()

    print(f"\nShutdown gracefully. Processed {engine.frame_count} frames")
    print("Logs saved")


if __name__ == "__main__":
    main()
