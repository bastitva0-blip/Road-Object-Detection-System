"""
Traffic sign detection and OCR (Tesseract).

No fine-tuned traffic-sign YOLOv8 model is available (needs a labeled dataset,
e.g. Roboflow/IDD, which requires an external account/download not available
here). Sign *candidates* are instead found with classical color+shape CV
(red/blue HSV masks + contour shape), which needs no training data. Swap
`detect_sign` for a fine-tuned-model call once Phase 4 dataset work lands;
`read_speed_limit`/`read_license_plate` (the OCR half) do not depend on that.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

# HSV ranges for common sign colors (OpenCV hue range 0-179)
_RED_RANGES = [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))]
_BLUE_RANGE = ((100, 70, 50), (130, 255, 255))


class TrafficSignOCR:
    """Traffic sign candidate detection + OCR for speed limits and license plates"""

    def __init__(self, min_area: int = 400):
        """
        Initialize traffic sign OCR

        Args:
            min_area: Minimum contour area (px^2) to consider as a sign candidate
        """
        self.min_area = min_area
        if not _TESSERACT_AVAILABLE:
            logger.warning("pytesseract not installed, OCR will be unavailable")

    def detect_sign(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect traffic sign candidates in frame via color + shape heuristics

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with "signs": list of {"bbox", "sign_type", "confidence"}
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        signs = []
        signs.extend(self._find_candidates(hsv, _RED_RANGES, "red"))
        signs.extend(self._find_candidates(hsv, [_BLUE_RANGE], "blue"))
        return {"signs": signs, "ocr_results": []}

    def _find_candidates(self, hsv: np.ndarray, ranges, color: str) -> List[Dict[str, Any]]:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h != 0 else 0
            if not (0.6 <= aspect <= 1.6):
                continue  # signs are roughly as wide as tall

            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            sign_type = self._classify_shape(circularity, color)

            candidates.append({
                "bbox": (x, y, w, h),
                "sign_type": sign_type,
                "confidence": round(min(circularity, 1.0), 2),
            })
        return candidates

    def _classify_shape(self, circularity: float, color: str) -> str:
        """Rough sign category from color + how circular the blob is (no fine-grained classifier)"""
        if color == "red":
            if circularity > 0.75:
                return "speed_limit_or_no_entry"
            return "stop_or_yield"
        return "mandatory_or_one_way"

    def read_speed_limit(self, sign_region: np.ndarray) -> Optional[int]:
        """
        Read speed limit number from a cropped sign image

        Args:
            sign_region: Cropped sign image (BGR). Best accuracy comes from
                cropping tightly to the number itself (excluding the colored
                ring/border) — pass an inset crop, not the raw detect_sign bbox.

        Returns:
            Speed limit value (5-150) or None if not confidently read
        """
        text = self._ocr(sign_region, whitelist="0123456789", psm=8)
        if not text:
            return None
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            return None
        value = int(digits)
        return value if 5 <= value <= 150 else None

    def read_license_plate(self, plate_region: np.ndarray) -> Optional[str]:
        """
        Read alphanumeric text from a cropped license plate image

        Args:
            plate_region: Cropped plate image (BGR)

        Returns:
            Cleaned plate string, or None if OCR found nothing usable
        """
        text = self._ocr(plate_region, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", psm=7)
        if not text:
            return None
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        return cleaned if len(cleaned) >= 4 else None

    def _ocr(self, region: np.ndarray, whitelist: str, psm: int) -> Optional[str]:
        """Preprocess a crop (grayscale, contrast threshold) and run Tesseract on it"""
        if not _TESSERACT_AVAILABLE or region is None or region.size == 0:
            return None
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
            config = f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
            return pytesseract.image_to_string(thresh, config=config).strip()
        except Exception:
            logger.exception("OCR failed")
            return None
