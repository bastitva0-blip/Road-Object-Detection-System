#!/usr/bin/env python3
"""
Benchmark the Hough-based lane detection + road marking pipeline.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from road_analysis.lane_detector import LaneDetector


def synthetic_road_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a synthetic frame with two white lane lines for benchmarking"""
    import cv2
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.line(frame, (int(width * 0.2), height), (int(width * 0.45), int(height * 0.6)), (255, 255, 255), 6)
    cv2.line(frame, (int(width * 0.8), height), (int(width * 0.55), int(height * 0.6)), (255, 255, 255), 6)
    return frame


def benchmark(iterations: int = 100, width: int = 1280, height: int = 720):
    frame = synthetic_road_frame(width, height)
    detector = LaneDetector()

    detector.detect(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(iterations):
        detector.detect(frame)
    lane_elapsed = time.perf_counter() - start

    detector.detect_road_markings(frame)  # warm-up
    start = time.perf_counter()
    for _ in range(iterations):
        detector.detect_road_markings(frame)
    markings_elapsed = time.perf_counter() - start

    lane_ms = (lane_elapsed / iterations) * 1000
    markings_ms = (markings_elapsed / iterations) * 1000
    total_fps = 1000 / (lane_ms + markings_ms)

    print(f"Benchmarking lane detection pipeline: {iterations} iterations @ {width}x{height}\n")
    print(f"{'lane line detection':32s}: {lane_ms:6.2f} ms/frame")
    print(f"{'road marking detection':32s}: {markings_ms:6.2f} ms/frame")
    print(f"{'combined':32s}: {lane_ms + markings_ms:6.2f} ms/frame  ({total_fps:5.1f} FPS max)")


if __name__ == "__main__":
    benchmark()
