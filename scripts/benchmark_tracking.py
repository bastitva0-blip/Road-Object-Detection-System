#!/usr/bin/env python3
"""
Benchmark the multi-object tracker + speed estimator with many simultaneous objects.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracking.tracker import MultiObjectTracker
from analytics.speed_estimator import SpeedEstimator


def benchmark(n_objects: int = 40, frames: int = 60, width: int = 1280, height: int = 720):
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, [width - 60, height - 100], size=(n_objects, 2))
    velocities = rng.uniform(-4, 4, size=(n_objects, 2))

    tracker = MultiObjectTracker()
    speed_estimator = SpeedEstimator(fps=30, pixels_per_meter=10.0)
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    start = time.perf_counter()
    id_switches = 0
    seen_ids = set()
    for _ in range(frames):
        positions = np.clip(positions + velocities, 0, [width - 60, height - 100])
        detections = [
            {"bbox": np.array([x, y, x + 60, y + 100]), "confidence": 0.9, "class_name": "car"}
            for x, y in positions
        ]
        tracks = tracker.update(detections)
        speed_estimator.estimate_speed(frame, tracks)
        seen_ids.update(t["track_id"] for t in tracks)
    elapsed = time.perf_counter() - start

    ms_per_frame = (elapsed / frames) * 1000
    print(f"Benchmarking tracking + speed pipeline: {n_objects} objects, {frames} frames\n")
    print(f"{'tracking + speed':32s}: {ms_per_frame:6.2f} ms/frame  ({1000 / ms_per_frame:5.1f} FPS max)")
    print(f"{'unique track IDs created':32s}: {tracker._next_id - 1} (expected ~{n_objects} if no ID switching)")


if __name__ == "__main__":
    benchmark()
