"""
Frame preprocessing module for image enhancement
"""

import cv2
import numpy as np
from typing import Optional


class FrameProcessor:
    """Frame preprocessing for detection optimization"""
    
    def __init__(self, enable_clahe: bool = True, enable_denoise: bool = True):
        """
        Initialize frame processor
        
        Args:
            enable_clahe: Enable Contrast Limited Adaptive Histogram Equalization
            enable_denoise: Enable denoising
        """
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        
        if enable_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def process(self, frame: np.ndarray, low_light: bool = False) -> np.ndarray:
        """
        Process a frame for detection
        
        Args:
            frame: Input frame (BGR)
            low_light: Apply low-light enhancement
        
        Returns:
            Processed frame
        """
        if self.enable_denoise:
            frame = cv2.fastNlMeansDenoising(frame, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        if low_light and self.enable_clahe:
            # Convert to LAB for better low-light processing
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        return frame
    
    @staticmethod
    def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize frame to target dimensions"""
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
