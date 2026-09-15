"""
Audio alerts: severity -> distinct beep tone, played via pygame.mixer when an
audio device is available. Degrades gracefully (logs + no-ops) when pygame or
an audio device isn't available — true in this development sandbox (pygame
failed to build here — no SDL dev headers) and on many headless deployment
targets, so this must never crash the main detection loop.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_TONES_HZ = {
    "LOW": 440,
    "MEDIUM": 800,
    "HIGH": 1200,
    "CRITICAL": 1600,
}


def generate_tone(frequency_hz: float, duration_s: float = 0.3, sample_rate: int = 44100) -> np.ndarray:
    """Generate a mono sine-wave tone as int16 PCM samples, with a short fade to avoid clicks"""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    tone = np.sin(2 * np.pi * frequency_hz * t)
    fade = 0.02
    envelope = np.minimum(1.0, np.minimum(t / fade, (duration_s - t) / fade))
    return (tone * envelope * 32767 * 0.5).astype(np.int16)


class AudioAlertPlayer:
    """Play a distinct beep tone per alert severity"""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the audio alert player. Attempts to open pygame's mixer;
        if unavailable (no pygame, no audio device), `play()` logs instead.

        Args:
            sample_rate: PCM sample rate for generated tones
        """
        self.sample_rate = sample_rate
        self._mixer = None
        self._available = False
        try:
            import pygame
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
            self._mixer = pygame.mixer
            self._available = True
        except Exception as exc:
            logger.warning(f"Audio alerts unavailable ({exc}); beeps will be logged, not played")

    @property
    def available(self) -> bool:
        return self._available

    def play(self, severity: str):
        """
        Play the tone for a severity name (LOW/MEDIUM/HIGH/CRITICAL)

        Args:
            severity: Alert severity name (case-insensitive); unknown names default to MEDIUM's tone
        """
        frequency = SEVERITY_TONES_HZ.get(severity.upper(), SEVERITY_TONES_HZ["MEDIUM"])
        if not self._available:
            logger.info(f"[audio alert suppressed, no device] {severity} -> {frequency}Hz")
            return
        try:
            samples = generate_tone(frequency, sample_rate=self.sample_rate)
            sound = self._mixer.Sound(buffer=samples.tobytes())
            sound.play()
        except Exception:
            logger.exception("Failed to play audio alert")
