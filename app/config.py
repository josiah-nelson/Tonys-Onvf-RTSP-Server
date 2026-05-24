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

# Server ports
WEB_UI_PORT = int(os.environ.get("WEB_UI_PORT", 5552))
MEDIAMTX_PORT = int(os.environ.get("MEDIAMTX_PORT", 8554))
MEDIAMTX_API_PORT = int(os.environ.get("MEDIAMTX_API_PORT", 9997))
ONVIF_BASE_PORT = int(os.environ.get("ONVIF_BASE_PORT", 8001))

# Thread configuration
WSGI_MAX_WORKERS = int(os.environ.get("WSGI_MAX_WORKERS", 20))

# AI defaults
AI_DEFAULT_MODEL = os.environ.get("AI_DEFAULT_MODEL", "yolov8n.pt")
AI_INFERENCE_FRAME_WIDTH = int(os.environ.get("AI_INFERENCE_FRAME_WIDTH", 640))
AI_COOLDOWN_SECONDS = float(os.environ.get("AI_COOLDOWN_SECONDS", 5.0))
AI_TARGET_INTERVAL = float(os.environ.get("AI_TARGET_INTERVAL", 0.50))

# Video encoding (GridFusion)
GF_VIDEO_BITRATE = os.environ.get("GF_VIDEO_BITRATE", "2500k")
GF_VIDEO_BUFSIZE = os.environ.get("GF_VIDEO_BUFSIZE", "5000k")
GF_ENCODER_PRESET = os.environ.get("GF_ENCODER_PRESET", "ultrafast")
