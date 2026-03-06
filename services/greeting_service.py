"""
services/greeting_service.py — Time-aware offline voice greeting using pyttsx3.
"""
import pyttsx3
import threading
import logging
import time
from datetime import datetime
from config.settings import KNOWN_GREETING_CD

log = logging.getLogger(__name__)


def _time_greeting(name: str) -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return f"Good morning {name}. Attendance recorded."
    elif 12 <= hour < 17:
        return f"Good afternoon {name}. Attendance recorded."
    return f"Good evening {name}. Welcome back."


class GreetingService:
    """
    Non-blocking, time-aware TTS greeting with per-student cooldown.
    Each greeting runs in a daemon thread; never blocks the camera loop.
    """

    def __init__(self, cooldown: int = KNOWN_GREETING_CD):
        self.cooldown = cooldown
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _on_cooldown(self, name: str) -> bool:
        with self._lock:
            return (time.monotonic() - self._last.get(name.lower(), 0)) < self.cooldown

    def _record(self, name: str) -> None:
        with self._lock:
            self._last[name.lower()] = time.monotonic()

    def _speak(self, text: str) -> None:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            log.error("TTS error: %s", exc)

    def greet(self, name: str) -> bool:
        """
        Speak a greeting for `name` if not on cooldown.
        Returns True if greeting was triggered, False if suppressed.
        """
        if self._on_cooldown(name):
            log.debug("'%s' greeting on cooldown.", name)
            return False
        text = _time_greeting(name)
        log.info("Speaking greeting for '%s': %s", name, text)
        self._record(name)
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()
        return True
