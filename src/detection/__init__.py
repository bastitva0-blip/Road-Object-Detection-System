"""
Detection modules for object, vehicle, and infrastructure detection
"""

from .yolo_detector import YOLODetector
from .object_detector import ObjectDetector

__all__ = ["YOLODetector", "ObjectDetector"]
