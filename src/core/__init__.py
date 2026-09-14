"""
Core modules for video capture and frame processing
"""

from .video_capture import VideoCapture
from .frame_processor import FrameProcessor
from .config import SystemConfig

__all__ = ["VideoCapture", "FrameProcessor", "SystemConfig"]
