"""
Event logging to CSV and JSON
"""

import csv
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


class EventLogger:
    """Log detection events to CSV and JSON"""
    
    def __init__(self, log_dir: str = "data/logs"):
        """
        Initialize event logger
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV log file
        self.csv_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.json_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.events = []
    
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log an event
        
        Args:
            event_type: Type of event
            details: Event details
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **details
        }
        
        self.events.append(event)
        self._write_event(event)
    
    def _write_event(self, event: Dict[str, Any]):
        """Write a single event to CSV"""
        if not self.csv_file.exists():
            # Create CSV with headers
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=event.keys())
                writer.writeheader()
                writer.writerow(event)
        else:
            # Append to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=event.keys())
                writer.writerow(event)
    
    def save_json_log(self):
        """Save events to JSON file"""
        with open(self.json_file, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def get_events(self, event_type: str = None, limit: int = 100):
        """
        Retrieve logged events
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
        
        Returns:
            List of events
        """
        if event_type:
            return [e for e in self.events if e["event_type"] == event_type][-limit:]
        return self.events[-limit:]
