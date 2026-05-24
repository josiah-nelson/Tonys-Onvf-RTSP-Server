"""Tests for environment-based configuration.

Verifies that:
- All config values have correct defaults when no .env is present
- Environment variable overrides work for all value types (int, float, str)
- Config module loads from .env file when python-dotenv is available
"""

import importlib
import os
import sys
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _reload_config(**env_overrides):
    """Reload app.config with optional env var overrides."""
    with mock.patch.dict(os.environ, env_overrides, clear=False):
        import app.config
        importlib.reload(app.config)
        return app.config


class TestDefaultValues:

    def test_web_ui_port(self):
        cfg = _reload_config()
        assert cfg.WEB_UI_PORT == 5552

    def test_mediamtx_port(self):
        cfg = _reload_config()
        assert cfg.MEDIAMTX_PORT == 8554

    def test_mediamtx_api_port(self):
        cfg = _reload_config()
        assert cfg.MEDIAMTX_API_PORT == 9997

    def test_onvif_base_port(self):
        cfg = _reload_config()
        assert cfg.ONVIF_BASE_PORT == 8001

    def test_wsgi_max_workers(self):
        cfg = _reload_config()
        assert cfg.WSGI_MAX_WORKERS == 20

    def test_ai_default_model(self):
        cfg = _reload_config()
        assert cfg.AI_DEFAULT_MODEL == "yolov8n.pt"

    def test_ai_inference_frame_width(self):
        cfg = _reload_config()
        assert cfg.AI_INFERENCE_FRAME_WIDTH == 640

    def test_ai_cooldown_seconds(self):
        cfg = _reload_config()
        assert cfg.AI_COOLDOWN_SECONDS == 5.0

    def test_ai_target_interval(self):
        cfg = _reload_config()
        assert cfg.AI_TARGET_INTERVAL == 0.50

    def test_gf_video_bitrate(self):
        cfg = _reload_config()
        assert cfg.GF_VIDEO_BITRATE == "2500k"

    def test_gf_video_bufsize(self):
        cfg = _reload_config()
        assert cfg.GF_VIDEO_BUFSIZE == "5000k"

    def test_gf_encoder_preset(self):
        cfg = _reload_config()
        assert cfg.GF_ENCODER_PRESET == "ultrafast"


class TestEnvOverrides:

    def test_override_int_port(self):
        cfg = _reload_config(WEB_UI_PORT="9999")
        assert cfg.WEB_UI_PORT == 9999

    def test_override_int_workers(self):
        cfg = _reload_config(WSGI_MAX_WORKERS="8")
        assert cfg.WSGI_MAX_WORKERS == 8

    def test_override_string_model(self):
        cfg = _reload_config(AI_DEFAULT_MODEL="yolov8s.pt")
        assert cfg.AI_DEFAULT_MODEL == "yolov8s.pt"

    def test_override_float_cooldown(self):
        cfg = _reload_config(AI_COOLDOWN_SECONDS="10.0")
        assert cfg.AI_COOLDOWN_SECONDS == 10.0

    def test_override_float_interval(self):
        cfg = _reload_config(AI_TARGET_INTERVAL="1.0")
        assert cfg.AI_TARGET_INTERVAL == 1.0

    def test_override_string_bitrate(self):
        cfg = _reload_config(GF_VIDEO_BITRATE="5000k")
        assert cfg.GF_VIDEO_BITRATE == "5000k"

    def test_override_frame_width(self):
        cfg = _reload_config(AI_INFERENCE_FRAME_WIDTH="320")
        assert cfg.AI_INFERENCE_FRAME_WIDTH == 320


class TestEnvExampleFile:

    def test_env_example_exists(self):
        path = os.path.join(_REPO_ROOT, ".env.example")
        assert os.path.isfile(path), ".env.example must exist in repo root"

    def test_env_example_documents_all_config_vars(self):
        path = os.path.join(_REPO_ROOT, ".env.example")
        with open(path) as f:
            content = f.read()

        expected_vars = [
            "WEB_UI_PORT", "MEDIAMTX_PORT", "MEDIAMTX_API_PORT",
            "ONVIF_BASE_PORT", "WSGI_MAX_WORKERS", "AI_DEFAULT_MODEL",
            "AI_INFERENCE_FRAME_WIDTH", "AI_COOLDOWN_SECONDS",
            "AI_TARGET_INTERVAL", "GF_VIDEO_BITRATE", "GF_VIDEO_BUFSIZE",
            "GF_ENCODER_PRESET",
        ]
        for var in expected_vars:
            assert var in content, f"{var} not documented in .env.example"

    def test_env_in_gitignore(self):
        path = os.path.join(_REPO_ROOT, ".gitignore")
        with open(path) as f:
            content = f.read()
        assert ".env" in content
