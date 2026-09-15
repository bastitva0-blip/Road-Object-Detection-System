"""
System configuration management
"""

import yaml
from dataclasses import dataclass, field
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

    # Detection classes to enable
    detection_classes: Dict[str, bool] = field(default_factory=lambda: {
        "people": True, "vehicles": True, "animals": True,
        "infrastructure": True, "road_markings": False,
    })

    # Alert settings
    alert_cooldown_seconds: int = 5
    enable_audio_alerts: bool = True
    enable_push_notifications: bool = False

    # Analytics settings
    speed_estimation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "pixels_per_meter": 10.0,
    })

    # Zone definitions (polygon points filled in at runtime or via config)
    zones: Dict[str, Any] = field(default_factory=dict)

    # Environmental adaptation
    night_mode: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, "brightness_threshold": 50,
    })
    fog_detection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, "clarity_threshold": 100,
    })

    @classmethod
    def from_yaml(cls, config_path: str) -> "SystemConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.__dict__
