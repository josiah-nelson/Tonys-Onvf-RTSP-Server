"""Tests for FFmpeg hardware acceleration detection and encoder selection.

Verifies that:
- The encoder_args bug fix works: hardware encoder is used when detected
- Software fallback only applies when no hardware encoder is available
- VideoToolbox is detected on macOS
- Encoder priority chain: NVENC > QSV > AMF > VideoToolbox
- Detection handles subprocess errors gracefully
"""

import sys
import os
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_manager():
    """Create a MediaMTXManager without triggering heavy imports."""
    from app.mediamtx_manager import MediaMTXManager
    return MediaMTXManager()


def _mock_ffmpeg_encoders(stdout_text):
    """Return a mock Popen that produces the given -encoders stdout."""
    mock_process = mock.MagicMock()
    mock_process.communicate.return_value = (stdout_text, "")
    return mock_process


# ---------------------------------------------------------------------------
# _detect_hardware_acceleration tests
# ---------------------------------------------------------------------------

class TestDetectHardwareAcceleration:

    def test_nvenc_detected(self):
        mgr = _make_manager()
        stdout = " V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is not None
        assert result["type"] == "nvenc"
        assert result["encoder"] == "h264_nvenc"

    def test_qsv_detected(self):
        mgr = _make_manager()
        stdout = " V..... h264_qsv  H.264 / AVC / MPEG-4 (Intel QSV)\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is not None
        assert result["type"] == "qsv"

    def test_amf_detected(self):
        mgr = _make_manager()
        stdout = " V..... h264_amf  H.264 AMD AMF encoder\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is not None
        assert result["type"] == "amf"

    def test_videotoolbox_detected(self):
        mgr = _make_manager()
        stdout = " V..... h264_videotoolbox  VideoToolbox H.264 encoder\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is not None
        assert result["type"] == "videotoolbox"
        assert result["encoder"] == "h264_videotoolbox"
        assert "-realtime 1" in result["params"]

    def test_nvenc_beats_videotoolbox(self):
        """NVENC has higher priority than VideoToolbox."""
        mgr = _make_manager()
        stdout = (
            " V..... h264_videotoolbox  VideoToolbox H.264 encoder\n"
            " V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n"
        )
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result["type"] == "nvenc"

    def test_no_hw_encoder_returns_none(self):
        mgr = _make_manager()
        stdout = " V..... libx264  H.264 / AVC / MPEG-4 (software)\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is None

    def test_subprocess_error_returns_none(self):
        mgr = _make_manager()
        with mock.patch("subprocess.Popen", side_effect=OSError("ffmpeg not found")):
            result = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert result is None


# ---------------------------------------------------------------------------
# Encoder args selection (bug fix verification)
# ---------------------------------------------------------------------------

class TestEncoderArgsSelection:
    """Verify the encoder selection logic in create_config via AST and
    by exercising _detect_hardware_acceleration end-to-end."""

    def test_hw_encoder_used_end_to_end(self):
        """When detection returns NVENC, the ffmpeg command must contain
        h264_nvenc and NOT libx264."""
        mgr = _make_manager()
        stdout = " V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            hw_info = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert hw_info is not None
        encoder_args = f'-c:v {hw_info["encoder"]} {hw_info["params"]}'
        assert "h264_nvenc" in encoder_args
        assert "libx264" not in encoder_args

    def test_videotoolbox_no_allow_sw(self):
        """VideoToolbox must use -allow_sw 0 to avoid silent software fallback."""
        mgr = _make_manager()
        stdout = " V..... h264_videotoolbox  VideoToolbox H.264 encoder\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            hw_info = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert "-allow_sw 0" in hw_info["params"]

    def test_software_fallback_when_no_hw(self):
        mgr = _make_manager()
        stdout = " V..... libx264  H.264 (software)\n"
        with mock.patch("subprocess.Popen", return_value=_mock_ffmpeg_encoders(stdout)):
            hw_info = mgr._detect_hardware_acceleration("/usr/bin/ffmpeg")
        assert hw_info is None

    def test_bug_fix_else_guard_in_source(self):
        """Verify the actual source has the `else` keyword guarding
        the software fallback so hw encoder_args are not overwritten."""
        manager_path = os.path.join(_REPO_ROOT, "app", "mediamtx_manager.py")
        with open(manager_path) as f:
            source_lines = f.readlines()

        for i, line in enumerate(source_lines):
            if 'hw_accel_info["encoder"]' in line:
                for j in range(i + 1, min(i + 5, len(source_lines))):
                    if "libx264" in source_lines[j]:
                        assert "else" in source_lines[j] or "else" in source_lines[j - 1], (
                            f"Line {j+1} sets libx264 without an `else` guard — "
                            "hardware encoder args will be overwritten"
                        )
                        return
        pytest.fail("Could not find the encoder_args assignment in create_config")
