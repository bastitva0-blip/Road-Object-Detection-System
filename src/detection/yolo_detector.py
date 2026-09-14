"""
YOLOv8 detector wrapper
"""

from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np


class YOLODetector:
    """YOLOv8 object detector"""
    
    def __init__(self, model_name: str = "yolov8n", device: str = "cpu"):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_name: Model name (nano, small, medium, large, extra)
            device: Device to use ("cpu", "0" for GPU)
        """
        self.model_name = model_name
        self.device = device
        self.model = YOLO(f"{model_name}.pt")
    
    def detect(self, frame: np.ndarray, conf: float = 0.5, iou: float = 0.45) -> List[Dict[str, Any]]:
        """
        Run detection on a frame
        
        Args:
            frame: Input frame (BGR)
            conf: Confidence threshold
            iou: IOU threshold
        
        Returns:
            List of detections with boxes, classes, and confidences
        """
        results = self.model(frame, conf=conf, iou=iou, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return detections
