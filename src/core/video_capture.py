"""
Video capture module for webcam handling
"""

import logging
import time
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VideoCapture:
    """Wrapper for OpenCV VideoCapture with preprocessing"""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 max_reconnect_attempts: int = 3, reconnect_delay: float = 1.0):
        """
        Initialize video capture

        Args:
            camera_index: Index of the camera device
            width: Capture width in pixels
            height: Capture height in pixels
            fps: Target frames per second
            max_reconnect_attempts: Retries on read failure before giving up
            reconnect_delay: Seconds to wait between reconnect attempts
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap = cv2.VideoCapture(camera_index)
        self._configure_camera()

    def _configure_camera(self):
        """Configure camera properties"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def _reconnect(self) -> bool:
        """Attempt to reopen camera device after disconnection"""
        logger.warning(f"Camera {self.camera_index} disconnected, attempting reconnect")
        self.cap.release()
        for attempt in range(1, self.max_reconnect_attempts + 1):
            time.sleep(self.reconnect_delay)
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self._configure_camera()
                logger.info(f"Camera {self.camera_index} reconnected on attempt {attempt}")
                return True
            logger.warning(f"Reconnect attempt {attempt}/{self.max_reconnect_attempts} failed")
        logger.error(f"Camera {self.camera_index} reconnect failed after {self.max_reconnect_attempts} attempts")
        return False

    def read(self) -> Tuple[bool, Optional[object]]:
        """Read a frame from the camera, reconnecting on failure"""
        ret, frame = self.cap.read()
        if not ret:
            if self._reconnect():
                ret, frame = self.cap.read()
            else:
                return False, None
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
