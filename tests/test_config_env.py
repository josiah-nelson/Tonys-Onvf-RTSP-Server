"""Tests for environment-based configuration.

Verifies that:
- All config values have correct defaults when no .env is present
- Environment variable overrides work for all value types (int, float, str)
- Empty string env vars fall back to defaults (Docker-safe)
- Malformed values fall back to defaults with a warning (not a crash)
- Boundary validation prevents dangerous values
- FFmpeg params are validated against safe patterns
"""

import importlib
import os
import sys
import types
from unittest import mock

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

    Removes app.config from sys.modules before reload and stubs the
    dotenv module so load_dotenv never reads a local .env file.
    """
    clean_env = {k: v for k, v in os.environ.items() if k not in _CONFIG_KEYS}
    clean_env.update(env_overrides)
    stub_dotenv = types.ModuleType("dotenv")
    stub_dotenv.load_dotenv = lambda *a, **kw: None
    sys.modules.pop("app.config", None)
    with mock.patch.dict(os.environ, clean_env, clear=True), \
         mock.patch.dict(sys.modules, {"dotenv": stub_dotenv}):
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

    def test_empty_int_falls_back(self):
        assert _reload_config(WEB_UI_PORT="").WEB_UI_PORT == 5552

    def test_empty_float_falls_back(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="").AI_COOLDOWN_SECONDS == 5.0

    def test_empty_string_falls_back(self):
        assert _reload_config(AI_DEFAULT_MODEL="").AI_DEFAULT_MODEL == "yolov8n.pt"

    def test_whitespace_only_falls_back(self):
        assert _reload_config(WSGI_MAX_WORKERS="  ").WSGI_MAX_WORKERS == 20


class TestMalformedValues:
    """Malformed env vars should fall back to defaults, not crash."""

    def test_non_numeric_int_falls_back(self):
        assert _reload_config(WEB_UI_PORT="abc").WEB_UI_PORT == 5552

    def test_non_numeric_float_falls_back(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="xyz").AI_COOLDOWN_SECONDS == 5.0

    def test_non_numeric_workers_falls_back(self):
        assert _reload_config(WSGI_MAX_WORKERS="auto").WSGI_MAX_WORKERS == 20


class TestBoundaryValidation:

    def test_wsgi_workers_minimum_1(self):
        assert _reload_config(WSGI_MAX_WORKERS="0").WSGI_MAX_WORKERS == 1

    def test_frame_width_minimum_1(self):
        assert _reload_config(AI_INFERENCE_FRAME_WIDTH="0").AI_INFERENCE_FRAME_WIDTH == 1

    def test_target_interval_minimum(self):
        assert _reload_config(AI_TARGET_INTERVAL="0").AI_TARGET_INTERVAL >= 0.01

    def test_cooldown_minimum_0(self):
        assert _reload_config(AI_COOLDOWN_SECONDS="-1").AI_COOLDOWN_SECONDS == 0.0

    def test_port_zero_falls_back(self):
        assert _reload_config(WEB_UI_PORT="0").WEB_UI_PORT == 5552

    def test_port_negative_falls_back(self):
        assert _reload_config(MEDIAMTX_PORT="-1").MEDIAMTX_PORT == 8554

    def test_port_too_high_falls_back(self):
        assert _reload_config(WEB_UI_PORT="70000").WEB_UI_PORT == 5552

    def test_port_65535_accepted(self):
        assert _reload_config(WEB_UI_PORT="65535").WEB_UI_PORT == 65535


class TestFFmpegParamValidation:

    def test_valid_bitrate_accepted(self):
        assert _reload_config(GF_VIDEO_BITRATE="5000k").GF_VIDEO_BITRATE == "5000k"

    def test_valid_bitrate_mega(self):
        assert _reload_config(GF_VIDEO_BITRATE="5M").GF_VIDEO_BITRATE == "5M"

    def test_invalid_bitrate_falls_back(self):
        assert _reload_config(GF_VIDEO_BITRATE="foo bar").GF_VIDEO_BITRATE == "2500k"

    def test_bitrate_injection_blocked(self):
        assert _reload_config(GF_VIDEO_BITRATE="2500k -i /etc/passwd").GF_VIDEO_BITRATE == "2500k"

    def test_valid_preset_accepted(self):
        assert _reload_config(GF_ENCODER_PRESET="slow").GF_ENCODER_PRESET == "slow"

    def test_invalid_preset_falls_back(self):
        assert _reload_config(GF_ENCODER_PRESET="malicious -x").GF_ENCODER_PRESET == "ultrafast"

    def test_unknown_preset_falls_back(self):
        assert _reload_config(GF_ENCODER_PRESET="turbo").GF_ENCODER_PRESET == "ultrafast"


class TestEnvExampleFile:

    def test_env_example_exists(self):
        assert os.path.isfile(os.path.join(_REPO_ROOT, ".env.example"))

    def test_env_example_documents_all_config_vars(self):
        with open(os.path.join(_REPO_ROOT, ".env.example")) as f:
            content = f.read()
        for var in _CONFIG_KEYS:
            assert var in content, f"{var} not documented in .env.example"

    def test_env_in_gitignore(self):
        with open(os.path.join(_REPO_ROOT, ".gitignore")) as f:
            assert ".env" in f.read()
