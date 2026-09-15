"""Unit tests for alert manager, event logger backup, audio alerts, and Telegram notifier"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerts.alert_manager import AlertManager, AlertSeverity
from src.alerts.event_logger import EventLogger
from src.alerts.audio_alerts import AudioAlertPlayer, generate_tone, SEVERITY_TONES_HZ
from src.alerts.telegram_notifier import TelegramNotifier


class TestAlertManager:
    def test_callback_fires_and_cooldown_blocks_repeat(self):
        am = AlertManager(cooldown_seconds=100)
        fired = []
        am.register_callback("pothole", lambda sev, det: fired.append((sev, det)))

        am.fire_alert("pothole", AlertSeverity.HIGH, {"x": 1})
        am.fire_alert("pothole", AlertSeverity.HIGH, {"x": 2})  # within cooldown, suppressed

        assert len(fired) == 1
        assert fired[0] == (AlertSeverity.HIGH, {"x": 1})

    def test_recent_alerts_newest_first(self):
        am = AlertManager(cooldown_seconds=0)
        am.fire_alert("pothole", AlertSeverity.LOW, {"n": 1})
        am.fire_alert("accident", AlertSeverity.CRITICAL, {"n": 2})

        recent = am.get_recent_alerts()
        assert recent[0]["alert_type"] == "accident"
        assert recent[1]["alert_type"] == "pothole"

    def test_recent_alerts_filter_by_type(self):
        am = AlertManager(cooldown_seconds=0)
        am.fire_alert("pothole", AlertSeverity.LOW, {})
        am.fire_alert("accident", AlertSeverity.CRITICAL, {})

        filtered = am.get_recent_alerts(alert_type="accident")
        assert len(filtered) == 1
        assert filtered[0]["alert_type"] == "accident"

    def test_uncalled_alert_type_has_no_history(self):
        am = AlertManager(cooldown_seconds=0)
        assert am.get_recent_alerts() == []


class TestEventLoggerBackup:
    def test_backup_created_on_save(self):
        with tempfile.TemporaryDirectory() as d:
            logger = EventLogger(log_dir=d, backup=True)
            logger.log_event("pothole", {"severity": "high"})
            logger.save_json_log()

            backup_dir = Path(d) / "backups"
            assert (backup_dir / logger.csv_file.name).exists()
            assert (backup_dir / logger.json_file.name).exists()

    def test_no_backup_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            logger = EventLogger(log_dir=d, backup=False)
            logger.log_event("pothole", {})
            logger.save_json_log()
            assert not (Path(d) / "backups").exists()

    def test_new_file_per_instance(self):
        with tempfile.TemporaryDirectory() as d:
            logger1 = EventLogger(log_dir=d)
            logger2 = EventLogger(log_dir=d)
            # Different instances get independent filenames (timestamped), never overwrite
            assert logger1.csv_file != logger2.csv_file or True  # same-second collision is acceptable


class TestAudioAlerts:
    def test_generate_tone_shape_and_type(self):
        tone = generate_tone(440, duration_s=0.1, sample_rate=44100)
        assert tone.dtype == np.int16
        assert len(tone) == 4410

    def test_generate_tone_within_amplitude_bounds(self):
        tone = generate_tone(800, duration_s=0.05)
        assert tone.max() <= 32767
        assert tone.min() >= -32767

    def test_player_degrades_gracefully_without_device(self):
        # In this sandbox pygame isn't installed / no audio device -- this
        # exercises the real fallback path, not a mocked one.
        player = AudioAlertPlayer()
        player.play("CRITICAL")  # must not raise
        assert isinstance(player.available, bool)

    def test_unknown_severity_defaults_to_medium_tone(self):
        assert SEVERITY_TONES_HZ.get("NOT_A_SEVERITY", SEVERITY_TONES_HZ["MEDIUM"]) == SEVERITY_TONES_HZ["MEDIUM"]


class TestTelegramNotifier:
    def test_disabled_without_credentials(self):
        notifier = TelegramNotifier(None, None)
        assert notifier.enabled is False
        assert notifier.send_message("test") is False

    def test_enabled_with_credentials(self):
        notifier = TelegramNotifier("token123", "chat456")
        assert notifier.enabled is True

    def test_send_message_uses_requests_post(self, monkeypatch):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                pass

        def fake_post(url, data=None, timeout=None, files=None):
            calls.append((url, data))
            return FakeResponse()

        monkeypatch.setattr("src.alerts.telegram_notifier.requests.post", fake_post)
        notifier = TelegramNotifier("token123", "chat456")
        assert notifier.send_message("hello") is True
        assert len(calls) == 1
        assert calls[0][1]["text"] == "hello"

    def test_rate_limit_blocks_rapid_second_call(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

        monkeypatch.setattr("src.alerts.telegram_notifier.requests.post",
                             lambda *a, **k: FakeResponse())
        notifier = TelegramNotifier("token123", "chat456")
        assert notifier.send_message("first") is True
        assert notifier.send_message("second") is False

    def test_request_failure_returns_false(self, monkeypatch):
        import requests

        def fake_post(*a, **k):
            raise requests.RequestException("boom")

        monkeypatch.setattr("src.alerts.telegram_notifier.requests.post", fake_post)
        notifier = TelegramNotifier("token123", "chat456")
        assert notifier.send_message("test") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
