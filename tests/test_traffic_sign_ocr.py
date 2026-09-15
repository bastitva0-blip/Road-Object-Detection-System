"""Unit tests for traffic sign detection and OCR"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.road_analysis.traffic_sign_ocr import TrafficSignOCR

requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract-ocr binary not installed"
)


class TestSignDetection:
    def test_detects_red_and_blue_candidates(self):
        import cv2
        ocr = TrafficSignOCR()
        frame = np.full((400, 400, 3), 100, dtype=np.uint8)
        cv2.circle(frame, (100, 100), 40, (255, 255, 255), -1)
        cv2.circle(frame, (100, 100), 40, (0, 0, 255), 8)
        cv2.circle(frame, (300, 300), 40, (255, 0, 0), -1)

        result = ocr.detect_sign(frame)
        assert len(result["signs"]) == 2

    def test_no_false_positives_on_gray_frame(self):
        ocr = TrafficSignOCR()
        frame = np.full((400, 400, 3), 100, dtype=np.uint8)
        result = ocr.detect_sign(frame)
        assert result["signs"] == []


class TestOCR:
    @requires_tesseract
    def test_reads_speed_limit_digits(self):
        import cv2
        ocr = TrafficSignOCR()
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize("60", cv2.FONT_HERSHEY_SIMPLEX, 3, 8)
        x, y = 100 - tw // 2, 100 + th // 2
        cv2.putText(img, "60", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)

        assert ocr.read_speed_limit(img) == 60

    @requires_tesseract
    def test_out_of_range_digits_rejected(self):
        import cv2
        ocr = TrafficSignOCR()
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.putText(img, "999", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)

        assert ocr.read_speed_limit(img) is None

    def test_ocr_returns_none_without_tesseract_or_empty_region(self):
        ocr = TrafficSignOCR()
        assert ocr.read_speed_limit(np.zeros((0, 0, 3), dtype=np.uint8)) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
