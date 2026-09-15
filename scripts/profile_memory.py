#!/usr/bin/env python3
"""
Profile memory usage of the detection pipeline (frame processing + YOLOv8 inference).
Requires: psutil (pip install psutil)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.frame_processor import FrameProcessor
from detection.object_detector import ObjectDetector

try:
    import psutil
except ImportError:
    print("psutil not installed. Run: pip install psutil")
    sys.exit(1)


def mb(bytes_val: int) -> float:
    return bytes_val / (1024 * 1024)


def profile(frames: int = 50, width: int = 1280, height: int = 720):
    process = psutil.Process()

    baseline = mb(process.memory_info().rss)
    print(f"Baseline RSS: {baseline:.1f} MB")

    frame_processor = FrameProcessor()
    detector = ObjectDetector(model_name="yolov8n", device="cpu")
    after_init = mb(process.memory_info().rss)
    print(f"After model load: {after_init:.1f} MB (+{after_init - baseline:.1f} MB)")

    peak = after_init
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    for i in range(frames):
        processed = frame_processor.process(frame.copy(), low_light=False)
        detector.detect(processed, conf=0.5)
        current = mb(process.memory_info().rss)
        peak = max(peak, current)

    final = mb(process.memory_info().rss)
    print(f"After {frames} frames: {final:.1f} MB (+{final - after_init:.1f} MB since load)")
    print(f"Peak RSS: {peak:.1f} MB")
    if final - after_init > 50:
        print("WARNING: >50MB growth after warmup, possible leak")


if __name__ == "__main__":
    profile()
