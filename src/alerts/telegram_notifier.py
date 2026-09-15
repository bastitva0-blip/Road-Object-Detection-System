"""
Telegram alert notifications via the Bot HTTPS API.

Uses plain `requests` calls instead of the `python-telegram-bot` package —
that library pulls in an async framework for a feature we don't need (just
two POST calls: sendMessage and sendPhoto). Needs TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID (see .env.example); without them this degrades to a no-op
logger, same pattern as AudioAlertPlayer. Not exercised against a real bot
here — sending a live message needs the user's own bot token, an external
service action outside what this environment can supply or should fabricate.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT_S = 10
_RATE_LIMIT_SECONDS = 3.0


class TelegramNotifier:
    """Send alert notifications to a Telegram chat"""

    def __init__(self, bot_token: Optional[str], chat_id: Optional[str]):
        """
        Initialize the notifier

        Args:
            bot_token: Telegram bot token from @BotFather (None disables sending)
            chat_id: Target chat ID (None disables sending)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._last_sent = 0.0
        if not (bot_token and chat_id):
            logger.warning("Telegram notifier disabled: bot_token/chat_id not configured")

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _rate_limited(self) -> bool:
        import time
        now = time.time()
        if now - self._last_sent < _RATE_LIMIT_SECONDS:
            return True
        self._last_sent = now
        return False

    def send_message(self, text: str) -> bool:
        """
        Send a text alert

        Args:
            text: Message body

        Returns:
            True if sent, False if disabled, rate-limited, or the request failed
        """
        if not self.enabled:
            logger.info(f"[telegram suppressed, not configured] {text}")
            return False
        if self._rate_limited():
            logger.info("[telegram suppressed, rate limited]")
            return False

        url = _API_BASE.format(token=self.bot_token, method="sendMessage")
        try:
            response = requests.post(
                url, data={"chat_id": self.chat_id, "text": text}, timeout=_TIMEOUT_S
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            logger.exception("Telegram sendMessage failed")
            return False

    def send_photo(self, image_bytes: bytes, caption: str = "") -> bool:
        """
        Send an alert with an attached image (e.g. a violation screenshot)

        Args:
            image_bytes: JPEG/PNG image bytes
            caption: Optional caption text

        Returns:
            True if sent, False if disabled, rate-limited, or the request failed
        """
        if not self.enabled:
            logger.info(f"[telegram photo suppressed, not configured] {caption}")
            return False
        if self._rate_limited():
            logger.info("[telegram photo suppressed, rate limited]")
            return False

        url = _API_BASE.format(token=self.bot_token, method="sendPhoto")
        try:
            response = requests.post(
                url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("alert.jpg", image_bytes, "image/jpeg")},
                timeout=_TIMEOUT_S,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            logger.exception("Telegram sendPhoto failed")
            return False
