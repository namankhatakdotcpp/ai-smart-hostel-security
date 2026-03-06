"""
services/telegram_service.py — Telegram Bot notification service.
All Telegram API calls are concentrated here.
"""
import logging
import os
import threading
import requests
from datetime import datetime
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_API
log = logging.getLogger(__name__)


def _configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _post_message(text: str) -> None:
    if not _configured():
        log.warning("Telegram not configured — skipping message.")
        return
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="sendMessage")
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text,
                                     "parse_mode": "HTML"}, timeout=6)
        log.debug("sendMessage → HTTP %d", r.status_code)
    except requests.RequestException as exc:
        log.error("Telegram message error: %s", exc)


def _post_photo(image_path: str, caption: str) -> None:
    if not _configured():
        log.warning("Telegram not configured — skipping photo.")
        return
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="sendPhoto")
    try:
        if os.path.isfile(image_path):
            with open(image_path, "rb") as img:
                r = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                    files={"photo": (os.path.basename(image_path), img, "image/jpeg")},
                    timeout=10,
                )
        else:
            r = requests.post(
                TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method="sendMessage"),
                json={"chat_id": TELEGRAM_CHAT_ID, "text": caption, "parse_mode": "HTML"},
                timeout=6,
            )
        log.debug("sendPhoto → HTTP %d | %s", r.status_code, r.text[:80])
    except requests.RequestException as exc:
        log.error("Telegram photo error: %s", exc)


def _async(fn, *args) -> None:
    threading.Thread(target=fn, args=args, daemon=True).start()


# ── Public API ─────────────────────────────────────────────────────────────────

def send_entry_alert(name: str, image_path: str) -> None:
    """Send a hostel-entry notification with the captured image."""
    now = datetime.now().strftime("%I:%M %p")
    caption = (
        f"🚪 <b>Hostel Entry</b>\n"
        f"Name:   <b>{name.title()}</b>\n"
        f"Time:   {now}\n"
        f"Status: Attendance Marked"
    )
    log.info("Queuing entry alert for '%s' with image: %s", name, image_path)
    _async(_post_photo, image_path, caption)


def send_unknown_alert(image_path: str) -> None:
    """Send an unknown-person security alert with the captured image."""
    now = datetime.now().strftime("%I:%M %p")
    caption = (
        f"⚠️ <b>Unknown Person Detected</b>\n"
        f"Time:     {now}\n"
        f"Location: Hostel Entry Gate"
    )
    log.warning("Queuing unknown-person alert, image: %s", image_path)
    _async(_post_photo, image_path, caption)


def send_security_threat(count: int, window_minutes: int) -> None:
    """Send a high-priority repeated-intruder alert (text only)."""
    now = datetime.now().strftime("%I:%M %p")
    text = (
        f"🚨 <b>SECURITY ALERT</b>\n"
        f"Repeated unknown visitor detected.\n"
        f"Count:    {count} in {window_minutes} minutes\n"
        f"Time:     {now}\n"
        f"Location: Hostel Entry Gate"
    )
    log.critical("SECURITY THREAT ALERT — %d unknowns in %dm.", count, window_minutes)
    _async(_post_message, text)
