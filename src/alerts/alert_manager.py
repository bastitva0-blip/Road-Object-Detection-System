"""
Alert management system
"""

from collections import deque
from typing import Dict, Any, Callable, Deque, List, Optional
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

    def __init__(self, cooldown_seconds: int = 5, history_size: int = 200):
        """
        Initialize alert manager

        Args:
            cooldown_seconds: Minimum seconds between identical alerts
            history_size: Max alerts kept in the pollable history (see get_recent_alerts)
        """
        self.cooldown_seconds = cooldown_seconds
        self.alert_callbacks: Dict[str, List[Callable]] = {}
        self.last_alert_time: Dict[str, float] = {}
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)

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

        self._history.append({
            "alert_type": alert_type,
            "severity": severity.name,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

        # Dispatch callbacks
        if alert_type in self.alert_callbacks:
            for callback in self.alert_callbacks[alert_type]:
                callback(severity, details)

    def get_recent_alerts(self, limit: int = 20, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Poll recently fired alerts without needing a registered callback (e.g. for a dashboard)

        Args:
            limit: Max alerts to return, most recent first
            alert_type: Filter to a single alert type, or None for all

        Returns:
            List of {"alert_type", "severity", "details", "timestamp"}, newest first
        """
        history = list(self._history)
        if alert_type is not None:
            history = [a for a in history if a["alert_type"] == alert_type]
        return list(reversed(history))[:limit]
