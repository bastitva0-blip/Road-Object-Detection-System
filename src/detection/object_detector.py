"""
High-level object detection interface
"""

from typing import List, Dict, Any
import numpy as np
from .yolo_detector import YOLODetector


class ObjectDetector:
    """High-level interface for object detection"""
    
    # Detection classes mapping
    CLASSES = {
        # People & Vulnerable Road Users
        "person": 0,
        "cyclist": 1,
        "motorcyclist": 2,
        "child": 3,
        
        # Vehicles
        "car": 4,
        "truck": 5,
        "bus": 6,
        "motorcycle": 7,
        "auto": 8,
        "bicycle": 9,
        "tempo": 10,
        "tractor": 11,
        "ambulance": 12,
        "police": 13,
        "fire_truck": 14,
        
        # Animals
        "cow": 15,
        "buffalo": 16,
        "dog": 17,
        
        # Infrastructure
        "traffic_light": 18,
        "stop_sign": 19,
        "speed_limit_sign": 20,
        "no_entry": 21,
        "one_way": 22,
    }
    
    def __init__(self, model_name: str = "yolov8n", device: str = "cpu"):
        """Initialize object detector"""
        self.detector = YOLODetector(model_name=model_name, device=device)
    
    def detect(self, frame: np.ndarray, conf: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection and categorize by class type
        
        Args:
            frame: Input frame
            conf: Confidence threshold
        
        Returns:
            Dictionary with categorized detections
        """
        detections = self.detector.detect(frame, conf=conf)
        
        categorized = {
            "people": [],
            "vehicles": [],
            "animals": [],
            "infrastructure": [],
            "all": detections
        }
        
        for det in detections:
            class_name = det["class_name"]
            if class_name in ["person", "cyclist", "motorcyclist", "child"]:
                categorized["people"].append(det)
            elif class_name in ["car", "truck", "bus", "motorcycle", "auto", "bicycle", "tempo", "tractor", "ambulance", "police", "fire_truck"]:
                categorized["vehicles"].append(det)
            elif class_name in ["cow", "buffalo", "dog"]:
                categorized["animals"].append(det)
            elif class_name in ["traffic_light", "stop_sign", "speed_limit_sign", "no_entry", "one_way"]:
                categorized["infrastructure"].append(det)
        
        return categorized
