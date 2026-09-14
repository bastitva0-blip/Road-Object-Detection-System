"""Unit tests for detection module"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.object_detector import ObjectDetector


class TestObjectDetector:
    """Test cases for ObjectDetector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = ObjectDetector()
        assert detector is not None
    
    def test_detect_single_frame(self):
        """Test detection on a single frame"""
        detector = ObjectDetector()
        # Create dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections = detector.detect(frame)
        assert "people" in detections
        assert "vehicles" in detections
        assert "animals" in detections
        assert "infrastructure" in detections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
