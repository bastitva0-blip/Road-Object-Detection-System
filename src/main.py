"""
Main entry point for the Road Object Detection System
"""

import argparse
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
    
    # Initialize components
    print("\nInitializing components...")
    
    with VideoCapture(camera_index=args.webcam, width=config.capture_width, height=config.capture_height) as cap:
        if not cap.is_opened():
            print("✗ Failed to open camera")
            return
        print("✓ Camera initialized")
        
        # Initialize frame processor
        frame_processor = FrameProcessor()
        print("✓ Frame processor initialized")
        
        # Initialize detector
        detector = ObjectDetector(model_name=config.model_name, device="cpu")
        print("✓ Detector initialized")
        
        # Initialize tracker
        tracker = MultiObjectTracker() if config.enable_tracking else None
        print("✓ Tracker initialized" if tracker else "⚠ Tracking disabled")
        
        # Initialize logger
        logger = EventLogger(log_dir=config.log_dir)
        print("✓ Event logger initialized")
        
        print("\n" + "=" * 50)
        print("Press 'q' to quit\n")
        
        # Main loop
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Process frame
            frame = frame_processor.process(frame, low_light=False)
            
            # Run detection
            detections = detector.detect(frame, conf=config.confidence_threshold)
            
            # Annotate frame
            for det in detections["all"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                confidence = det["confidence"]
                class_name = det["class_name"]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Road Object Detection", frame)
            
            # Log frame stats
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(detections['all'])} objects detected")
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"data/screenshots/frame_{frame_count:06d}.png"
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    # Cleanup
    print(f"\nShutdown gracefully. Processed {frame_count} frames")
    logger.save_json_log()
    print("Logs saved")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
