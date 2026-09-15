#!/usr/bin/env python3
"""
Benchmark the FrameProcessor preprocessing pipeline (CLAHE + denoise).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.frame_processor import FrameProcessor


def benchmark(iterations: int = 100, width: int = 1280, height: int = 720):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    configs = [
        ("no preprocessing", FrameProcessor(enable_clahe=False, enable_denoise=False)),
        ("denoise only", FrameProcessor(enable_clahe=False, enable_denoise=True)),
        ("clahe only (low light)", FrameProcessor(enable_clahe=True, enable_denoise=False)),
        ("clahe + denoise (low light)", FrameProcessor(enable_clahe=True, enable_denoise=True)),
    ]

    print(f"Benchmarking preprocessing pipeline: {iterations} iterations @ {width}x{height}\n")
    for name, processor in configs:
        low_light = "low light" in name
        # warm-up
        processor.process(frame.copy(), low_light=low_light)

        start = time.perf_counter()
        for _ in range(iterations):
            processor.process(frame.copy(), low_light=low_light)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"{name:32s}: {avg_ms:6.2f} ms/frame  ({1000 / avg_ms:5.1f} FPS max)")


if __name__ == "__main__":
    benchmark()
