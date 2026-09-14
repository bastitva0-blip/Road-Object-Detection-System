"""
Video capture module for webcam handling
"""

import cv2
from typing import Optional, Tuple


class VideoCapture:
    """Wrapper for OpenCV VideoCapture with preprocessing"""
    
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize video capture
        
        Args:
            camera_index: Index of the camera device
            width: Capture width in pixels
            height: Capture height in pixels
            fps: Target frames per second
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = cv2.VideoCapture(camera_index)
        self._configure_camera()
    
    def _configure_camera(self):
        """Configure camera properties"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    def read(self) -> Tuple[bool, Optional[object]]:
        """Read a frame from the camera"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """Release the camera"""
        self.cap.release()
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap.isOpened()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
