import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# Root directory is the parent of the 'app' directory containing this config file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT_DIR, "camera_config.json")


def _env_int(name, default):
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print(f"  [Config] Warning: {name}={val!r} is not a valid integer, using default {default}")
        return default


def _env_float(name, default):
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"  [Config] Warning: {name}={val!r} is not a valid number, using default {default}")
        return default


def _env_str(name, default):
    val = os.environ.get(name, "").strip()
    return val if val else default


def _env_port(name, default):
    val = _env_int(name, default)
    if not (1 <= val <= 65535):
        print(f"  [Config] Warning: {name}={val} is out of range 1-65535, using default {default}")
        return default
    return val


# Server ports
WEB_UI_PORT = _env_port("WEB_UI_PORT", 5552)
MEDIAMTX_PORT = _env_port("MEDIAMTX_PORT", 8554)
MEDIAMTX_API_PORT = _env_port("MEDIAMTX_API_PORT", 9997)
ONVIF_BASE_PORT = _env_port("ONVIF_BASE_PORT", 8001)

# Thread configuration
WSGI_MAX_WORKERS = max(1, _env_int("WSGI_MAX_WORKERS", 20))

# AI defaults
AI_DEFAULT_MODEL = _env_str("AI_DEFAULT_MODEL", "yolov8n.pt")
AI_INFERENCE_FRAME_WIDTH = max(1, _env_int("AI_INFERENCE_FRAME_WIDTH", 640))
AI_COOLDOWN_SECONDS = max(0.0, _env_float("AI_COOLDOWN_SECONDS", 5.0))
AI_TARGET_INTERVAL = max(0.01, _env_float("AI_TARGET_INTERVAL", 0.50))
AI_CONFIDENCE_THRESHOLD = max(1, min(100, _env_int("AI_CONFIDENCE_THRESHOLD", 40)))
AI_MOTION_SENSITIVITY = max(1, min(100, _env_int("AI_MOTION_SENSITIVITY", 50)))

# RTSP frame grabber
GRABBER_RECONNECT_BASE = max(0.5, _env_float("GRABBER_RECONNECT_BASE", 1.0))
GRABBER_RECONNECT_MAX = max(1.0, _env_float("GRABBER_RECONNECT_MAX", 30.0))

# Video encoding (GridFusion)
import re as _re

_BITRATE_RE = _re.compile(r"^\d+[kKmM]?$")
_X264_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}


def _env_bitrate(name, default):
    val = _env_str(name, default)
    if not _BITRATE_RE.match(val):
        print(f"  [Config] Warning: {name}={val!r} is not a valid bitrate, using default {default}")
        return default
    return val


def _env_preset(name, default):
    val = _env_str(name, default)
    if val not in _X264_PRESETS:
        print(f"  [Config] Warning: {name}={val!r} is not a valid x264 preset, using default {default}")
        return default
    return val


GF_VIDEO_BITRATE = _env_bitrate("GF_VIDEO_BITRATE", "2500k")
GF_VIDEO_BUFSIZE = _env_bitrate("GF_VIDEO_BUFSIZE", "5000k")
GF_ENCODER_PRESET = _env_preset("GF_ENCODER_PRESET", "ultrafast")
