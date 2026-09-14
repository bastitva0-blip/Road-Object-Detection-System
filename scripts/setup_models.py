#!/usr/bin/env python3
"""
Setup script to download and prepare models
"""

import os
from pathlib import Path
from ultralytics import YOLO


def setup_yolo_models():
    """Download YOLOv8 models"""
    print("Downloading YOLOv8 models...")
    
    models_dir = Path("data/models/yolov8")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_sizes = ["nano", "small", "medium"]
    
    for size in model_sizes:
        model_name = f"yolov8{size[0]}"  # yolov8n, yolov8s, yolov8m
        print(f"  Downloading {model_name}...")
        model = YOLO(f"{model_name}.pt")
        print(f"  ✓ {model_name} downloaded")
    
    print("✓ YOLOv8 models setup complete")


def setup_project_structure():
    """Ensure all necessary directories exist"""
    print("Creating project directories...")
    
    dirs = [
        "data/logs",
        "data/recordings",
        "data/screenshots",
        "data/datasets",
        "data/models",
        "notebooks",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    print("✓ Project structure ready")


if __name__ == "__main__":
    print("=" * 50)
    print("Road Object Detection - Setup Script")
    print("=" * 50 + "\n")
    
    setup_project_structure()
    print()
    setup_yolo_models()
    
    print("\n" + "=" * 50)
    print("Setup complete! Ready to run: python src/main.py")
    print("=" * 50)
