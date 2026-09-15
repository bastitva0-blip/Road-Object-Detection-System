#!/usr/bin/env python3
"""
Benchmark road anomaly detection (potholes/debris/waterlogging) and traffic
sign candidate detection, to size the ANALYTICS_INTERVAL throttle in main.py.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from road_analysis.pothole_detector import PotholeDetector
from road_analysis.traffic_sign_ocr import TrafficSignOCR


def benchmark(iterations: int = 30, width: int = 1280, height: int = 720):
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 120, (height, width, 3), dtype=np.uint8)

    pothole_detector = PotholeDetector()
    pothole_detector.detect_anomalies(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(iterations):
        pothole_detector.detect_anomalies(frame)
    pothole_ms = (time.perf_counter() - start) / iterations * 1000

    sign_ocr = TrafficSignOCR()
    sign_ocr.detect_sign(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(iterations):
        sign_ocr.detect_sign(frame)
    sign_ms = (time.perf_counter() - start) / iterations * 1000

    print(f"Benchmarking road anomaly + sign detection: {iterations} iterations @ {width}x{height}\n")
    print(f"{'pothole/debris/waterlogging':32s}: {pothole_ms:6.2f} ms/frame")
    print(f"{'sign candidate detection':32s}: {sign_ms:6.2f} ms/frame")
    print(f"{'combined':32s}: {pothole_ms + sign_ms:6.2f} ms/frame")
    print("\n(main.py throttles both to every ANALYTICS_INTERVAL-th frame; OCR calls")
    print(" only fire on a detected speed-limit-shaped candidate, not every frame)")


if __name__ == "__main__":
    benchmark()
