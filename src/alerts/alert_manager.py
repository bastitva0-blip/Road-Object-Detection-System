"""
Alert management system
"""

from typing import Dict, Any, Callable, List
from enum import Enum
from datetime import datetime


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertManager:
    """Manage and dispatch alerts"""
    
    def __init__(self, cooldown_seconds: int = 5):
        """
        Initialize alert manager
        
        Args:
            cooldown_seconds: Minimum seconds between identical alerts
        """
        self.cooldown_seconds = cooldown_seconds
        self.alert_callbacks: Dict[str, List[Callable]] = {}
        self.last_alert_time: Dict[str, float] = {}
    
    def register_callback(self, alert_type: str, callback: Callable):
        """
        Register a callback for an alert type
        
        Args:
            alert_type: Type of alert
            callback: Function to call when alert fires
        """
        if alert_type not in self.alert_callbacks:
            self.alert_callbacks[alert_type] = []
        self.alert_callbacks[alert_type].append(callback)
    
    def fire_alert(self, alert_type: str, severity: AlertSeverity, details: Dict[str, Any]):
        """
        Fire an alert
        
        Args:
            alert_type: Type of alert
            severity: Severity level
            details: Alert details
        """
        now = datetime.now().timestamp()
        
        # Check cooldown
        if alert_type in self.last_alert_time:
            if now - self.last_alert_time[alert_type] < self.cooldown_seconds:
                return
        
        self.last_alert_time[alert_type] = now
        
        # Dispatch callbacks
        if alert_type in self.alert_callbacks:
            for callback in self.alert_callbacks[alert_type]:
                callback(severity, details)
