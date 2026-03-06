"""
config/settings.py — Single source of truth for all system parameters.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH        = os.path.join(BASE_DIR, "data", "attendance.db")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
CAPTURES_DIR   = os.path.join(BASE_DIR, "captures")
THREAT_LOG     = os.path.join(BASE_DIR, "data", "threat_log.json")

# ── Camera ─────────────────────────────────────────────────────────────────────

def find_builtin_camera() -> int:
    """
    Return the AVFoundation device index for the MacBook FaceTime HD Camera.

    Uses `ffmpeg -f avfoundation -list_devices` to read device names so the
    correct index is found even when Continuity Camera is connected and macOS
    has reassigned device indices.

    Falls back to index 0 if ffmpeg is unavailable or FaceTime isn't listed.
    """
    import re, subprocess
    BUILTIN_KEYWORDS = ("facetime", "built-in", "isight", "apple camera")
    EXCLUDE_KEYWORDS = ("iphone", "ipad", "continuity", "virtual", "obs")

    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stderr  # ffmpeg prints device list to stderr

        in_video_section = False
        for line in output.splitlines():
            if "AVFoundation video devices" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices" in line:
                break
            if not in_video_section:
                continue

            # Each line looks like: [AVFoundation ...] [0] FaceTime HD Camera
            match = re.search(r"\[(\d+)\]\s+(.+)", line)
            if not match:
                continue

            idx  = int(match.group(1))
            name = match.group(2).strip().lower()

            # Skip known external / virtual cameras
            if any(k in name for k in EXCLUDE_KEYWORDS):
                continue
            # Accept known built-in camera names
            if any(k in name for k in BUILTIN_KEYWORDS):
                import logging
                logging.getLogger("config.settings").info(
                    "Auto-detected built-in camera: '%s' at index %d",
                    match.group(2).strip(), idx,
                )
                return idx

    except Exception:
        pass   # ffmpeg not found or timed out — fall back to 0

    return 0   # safe default


# Resolve once at import time so every module shares the same value
CAMERA_ID  = find_builtin_camera()
TARGET_FPS = 15
FRAME_SKIP = 2        # Run InsightFace every N frames (CPU optimisation)
FRAME_W    = 640
FRAME_H    = 480

# ── Recognition ────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.50   # Best-match cosine similarity minimum
FRAMES_TO_CAPTURE    = 20     # Samples collected during registration
BLUR_THRESHOLD       = 80.0   # Laplacian variance below this → blurry

# ── Alerts and cooldowns ───────────────────────────────────────────────────────
UNKNOWN_COOLDOWN   = 60    # Seconds between unknown-person Telegram alerts
KNOWN_GREETING_CD  = 30    # Seconds between voice greetings per student
SECURITY_WINDOW    = 120   # Rolling window (s) for threat detection
SECURITY_THRESHOLD = 3     # Unknown detections in window to trigger alert

# ── Telegram ───────────────────────────────────────────────────────────────────
# Credentials: set directly here OR override via a root-level config.py
TELEGRAM_BOT_TOKEN = "8676269692:AAFTv2azUqLgkcLxFjLaCfS2yGXJcSmnEC0"
TELEGRAM_CHAT_ID   = "1490124186"

try:
    import config as _cfg           # root-level config.py can override credentials
    if hasattr(_cfg, "BOT_TOKEN"):
        TELEGRAM_BOT_TOKEN = _cfg.BOT_TOKEN
    if hasattr(_cfg, "CHAT_ID"):
        TELEGRAM_CHAT_ID   = str(_cfg.CHAT_ID)
    if hasattr(_cfg, "TELEGRAM_BOT_TOKEN"):
        TELEGRAM_BOT_TOKEN = _cfg.TELEGRAM_BOT_TOKEN
    if hasattr(_cfg, "TELEGRAM_CHAT_ID"):
        TELEGRAM_CHAT_ID   = str(_cfg.TELEGRAM_CHAT_ID)
except Exception:
    pass   # use the defaults set above

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(BASE_DIR, "app.log")
