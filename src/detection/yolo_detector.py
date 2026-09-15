"""
YOLOv8 detector wrapper
"""

import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path("data/models/yolov8")


class YOLODetector:
    """YOLOv8 object detector"""

    def __init__(self, model_name: str = "yolov8n", device: str = "cpu", imgsz: int = 640):
        """
        Initialize YOLOv8 detector

        Args:
            model_name: Model name (nano, small, medium, large, extra)
            device: Device to use ("cpu", "0" for GPU)
            imgsz: Inference image size; smaller is faster on CPU
        """
        self.model_name = model_name
        self.device = device
        self.imgsz = imgsz
        self.half = device != "cpu"  # FP16 only supported on GPU

        weights_path = self._resolve_weights(model_name)
        self.model = YOLO(weights_path)
        if device == "cpu":
            # Fuse Conv+BN layers for faster CPU inference
            self.model.fuse()

    def _resolve_weights(self, model_name: str) -> str:
        """Load weights from local cache if present, else download and cache them"""
        cached = MODELS_CACHE_DIR / f"{model_name}.pt"
        if cached.exists():
            logger.info(f"Loading cached weights: {cached}")
            return str(cached)

        logger.info(f"No cached weights for {model_name}, downloading")
        downloaded = YOLO(f"{model_name}.pt")
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded.save(str(cached))
        logger.info(f"Cached weights to {cached}")
        return str(cached)

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
        kwargs = {"half": True} if self.half else {}
        results = self.model(frame, conf=conf, iou=iou, device=self.device,
                              imgsz=self.imgsz, verbose=False, **kwargs)

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
