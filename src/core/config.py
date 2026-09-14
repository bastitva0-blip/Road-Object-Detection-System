"""
System configuration management
"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SystemConfig:
    """Main system configuration"""
    
    # Video capture settings
    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    fps: int = 30
    
    # Detection settings
    model_name: str = "yolov8n"  # nano for CPU, small for GPU
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    
    # Processing
    enable_tracking: bool = True
    enable_lane_detection: bool = True
    enable_analytics: bool = True
    
    # Output
    save_logs: bool = True
    save_recordings: bool = False
    log_dir: str = "data/logs"
    recording_dir: str = "data/recordings"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "SystemConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.__dict__
