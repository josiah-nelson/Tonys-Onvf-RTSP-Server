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
    return int(val)


def _env_float(name, default):
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    return float(val)


def _env_str(name, default):
    val = os.environ.get(name, "").strip()
    return val if val else default


# Server ports
WEB_UI_PORT = _env_int("WEB_UI_PORT", 5552)
MEDIAMTX_PORT = _env_int("MEDIAMTX_PORT", 8554)
MEDIAMTX_API_PORT = _env_int("MEDIAMTX_API_PORT", 9997)
ONVIF_BASE_PORT = _env_int("ONVIF_BASE_PORT", 8001)

# Thread configuration
WSGI_MAX_WORKERS = max(1, _env_int("WSGI_MAX_WORKERS", 20))

# AI defaults
AI_DEFAULT_MODEL = _env_str("AI_DEFAULT_MODEL", "yolov8n.pt")
AI_INFERENCE_FRAME_WIDTH = max(1, _env_int("AI_INFERENCE_FRAME_WIDTH", 640))
AI_COOLDOWN_SECONDS = max(0.0, _env_float("AI_COOLDOWN_SECONDS", 5.0))
AI_TARGET_INTERVAL = max(0.01, _env_float("AI_TARGET_INTERVAL", 0.50))

# Video encoding (GridFusion)
GF_VIDEO_BITRATE = _env_str("GF_VIDEO_BITRATE", "2500k")
GF_VIDEO_BUFSIZE = _env_str("GF_VIDEO_BUFSIZE", "5000k")
GF_ENCODER_PRESET = _env_str("GF_ENCODER_PRESET", "ultrafast")
