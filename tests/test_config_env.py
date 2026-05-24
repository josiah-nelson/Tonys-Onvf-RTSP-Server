"""Tests for environment-based configuration.

Verifies that:
- All config values have correct defaults when no .env is present
- Environment variable overrides work for all value types (int, float, str)
- Empty string env vars fall back to defaults (Docker-safe)
- Boundary validation prevents dangerous values
"""

import importlib
import os
import sys
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_CONFIG_KEYS = [
    "WEB_UI_PORT", "MEDIAMTX_PORT", "MEDIAMTX_API_PORT", "ONVIF_BASE_PORT",
    "WSGI_MAX_WORKERS", "AI_DEFAULT_MODEL", "AI_INFERENCE_FRAME_WIDTH",
    "AI_COOLDOWN_SECONDS", "AI_TARGET_INTERVAL", "GF_VIDEO_BITRATE",
    "GF_VIDEO_BUFSIZE", "GF_ENCODER_PRESET",
]


def _reload_config(**env_overrides):
    """Reload app.config with optional env var overrides.

    Strips known config keys from the environment first so that
    CI/Docker env vars don't leak into default-value assertions.
    """
    clean_env = {k: v for k, v in os.environ.items() if k not in _CONFIG_KEYS}
    clean_env.update(env_overrides)
    with mock.patch.dict(os.environ, clean_env, clear=True):
        import app.config
        importlib.reload(app.config)
        return app.config


class TestDefaultValues:

    def test_web_ui_port(self):
        assert _reload_config().WEB_UI_PORT == 5552

    def test_mediamtx_port(self):
        assert _reload_config().MEDIAMTX_PORT == 8554

    def test_mediamtx_api_port(self):
        assert _reload_config().MEDIAMTX_API_PORT == 9997

    def test_onvif_base_port(self):
        assert _reload_config().ONVIF_BASE_PORT == 8001

    def test_wsgi_max_workers(self):
        assert _reload_config().WSGI_MAX_WORKERS == 20

    def test_ai_default_model(self):
        assert _reload_config().AI_DEFAULT_MODEL == "yolov8n.pt"

    def test_ai_inference_frame_width(self):
        assert _reload_config().AI_INFERENCE_FRAME_WIDTH == 640

    def test_ai_cooldown_seconds(self):
        assert _reload_config().AI_COOLDOWN_SECONDS == 5.0

    def test_ai_target_interval(self):
        assert _reload_config().AI_TARGET_INTERVAL == 0.50

    def test_gf_video_bitrate(self):
        assert _reload_config().GF_VIDEO_BITRATE == "2500k"

    def test_gf_video_bufsize(self):
        assert _reload_config().GF_VIDEO_BUFSIZE == "5000k"

    def test_gf_encoder_preset(self):
        assert _reload_config().GF_ENCODER_PRESET == "ultrafast"


class TestEnvOverrides:

    def test_override_int_port(self):
        assert _reload_config(WEB_UI_PORT="9999").WEB_UI_PORT == 9999

    def test_override_int_workers(self):
        assert _reload_config(WSGI_MAX_WORKERS="8").WSGI_MAX_WORKERS == 8

    def test_override_string_model(self):
        assert _reload_config(AI_DEFAULT_MODEL="yolov8s.pt").AI_DEFAULT_MODEL == "yolov8s.pt"

    def test_override_float_cooldown(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="10.0").AI_COOLDOWN_SECONDS == 10.0

    def test_override_float_interval(self):
        assert _reload_config(AI_TARGET_INTERVAL="1.0").AI_TARGET_INTERVAL == 1.0

    def test_override_string_bitrate(self):
        assert _reload_config(GF_VIDEO_BITRATE="5000k").GF_VIDEO_BITRATE == "5000k"

    def test_override_frame_width(self):
        assert _reload_config(AI_INFERENCE_FRAME_WIDTH="320").AI_INFERENCE_FRAME_WIDTH == 320


class TestEmptyStringFallback:
    """Empty env vars (common in Docker) should fall back to defaults."""

    def test_empty_int_falls_back(self):
        assert _reload_config(WEB_UI_PORT="").WEB_UI_PORT == 5552

    def test_empty_float_falls_back(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="").AI_COOLDOWN_SECONDS == 5.0

    def test_empty_string_falls_back(self):
        assert _reload_config(AI_DEFAULT_MODEL="").AI_DEFAULT_MODEL == "yolov8n.pt"

    def test_whitespace_only_falls_back(self):
        assert _reload_config(WSGI_MAX_WORKERS="  ").WSGI_MAX_WORKERS == 20


class TestBoundaryValidation:
    """Critical numeric values are clamped to safe minimums."""

    def test_wsgi_workers_minimum_1(self):
        assert _reload_config(WSGI_MAX_WORKERS="0").WSGI_MAX_WORKERS == 1

    def test_frame_width_minimum_1(self):
        assert _reload_config(AI_INFERENCE_FRAME_WIDTH="0").AI_INFERENCE_FRAME_WIDTH == 1

    def test_target_interval_minimum(self):
        cfg = _reload_config(AI_TARGET_INTERVAL="0")
        assert cfg.AI_TARGET_INTERVAL >= 0.01

    def test_cooldown_minimum_0(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="-1").AI_COOLDOWN_SECONDS == 0.0


class TestEnvExampleFile:

    def test_env_example_exists(self):
        path = os.path.join(_REPO_ROOT, ".env.example")
        assert os.path.isfile(path)

    def test_env_example_documents_all_config_vars(self):
        path = os.path.join(_REPO_ROOT, ".env.example")
        with open(path) as f:
            content = f.read()

        for var in _CONFIG_KEYS:
            assert var in content, f"{var} not documented in .env.example"

    def test_env_in_gitignore(self):
        path = os.path.join(_REPO_ROOT, ".gitignore")
        with open(path) as f:
            content = f.read()
        assert ".env" in content
