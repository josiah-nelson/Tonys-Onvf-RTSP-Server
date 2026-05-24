"""Tests for CoreML / Apple Neural Engine export and caching.

Verifies that:
- Cached .mlpackage is reused when valid
- Incomplete/stale caches are detected and re-exported
- Export failures write a skip marker to avoid retry storms
- Hash-based staleness detection works correctly
- Two-stage pipeline (motion gate -> YOLO) is preserved
"""

import ast
import os
import sys
import types
import tempfile
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.coreml_cache import (
    get_coreml_model_path, _is_cache_valid, _source_hash,
    _meta_path, _should_skip_export, _write_skip_marker,
)


def _make_valid_cache(cache_dir, base, source_path=None):
    """Create a valid cached .mlpackage with sentinel and hash."""
    os.makedirs(cache_dir, exist_ok=True)
    mlpackage = os.path.join(cache_dir, base + ".mlpackage")
    os.makedirs(mlpackage, exist_ok=True)
    with open(os.path.join(mlpackage, "model.mlmodel"), "w") as f:
        f.write("fake")
    with open(_meta_path(cache_dir, base, ".sentinel"), "w") as f:
        f.write("ok")
    if source_path and os.path.isfile(source_path):
        h = _source_hash(source_path)
        if h:
            with open(_meta_path(cache_dir, base, ".hash"), "w") as f:
                f.write(h)
    return mlpackage


# ---------------------------------------------------------------------------
# Cache validity tests
# ---------------------------------------------------------------------------

class TestCacheValidity:

    def test_valid_with_sentinel_and_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            _make_valid_cache(root, "yolov8n")
            assert _is_cache_valid(root, "yolov8n", "nonexistent.pt") is True

    def test_invalid_without_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            mlpackage = os.path.join(root, "yolov8n.mlpackage")
            os.makedirs(mlpackage)
            assert _is_cache_valid(root, "yolov8n", "nonexistent.pt") is False

    def test_invalid_without_artifact(self):
        with tempfile.TemporaryDirectory() as root:
            with open(_meta_path(root, "yolov8n", ".sentinel"), "w") as f:
                f.write("ok")
            assert _is_cache_valid(root, "yolov8n", "nonexistent.pt") is False

    def test_stale_when_hash_differs(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "model.pt")
            with open(source, "wb") as f:
                f.write(b"new weights")
            cache_dir = os.path.join(root, "cache")
            _make_valid_cache(cache_dir, "model")
            with open(_meta_path(cache_dir, "model", ".hash"), "w") as f:
                f.write("old_hash")
            assert _is_cache_valid(cache_dir, "model", source) is False

    def test_valid_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "model.pt")
            with open(source, "wb") as f:
                f.write(b"weights data")
            cache_dir = os.path.join(root, "cache")
            _make_valid_cache(cache_dir, "model", source)
            assert _is_cache_valid(cache_dir, "model", source) is True

    def test_valid_when_hash_uncomputable(self):
        """When hash can't be computed, treat cache as valid (can't verify)."""
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "real.pt")
            with open(source, "wb") as f:
                f.write(b"x")
            cache_dir = os.path.join(root, "cache")
            _make_valid_cache(cache_dir, "model", source)
            with mock.patch("app.coreml_cache._source_hash", return_value=None):
                assert _is_cache_valid(cache_dir, "model", source) is True

    def test_valid_when_hash_file_missing(self):
        """Missing hash file means staleness can't be checked — treat as valid."""
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "model.pt")
            with open(source, "wb") as f:
                f.write(b"weights")
            cache_dir = os.path.join(root, "cache")
            _make_valid_cache(cache_dir, "model", source)
            os.remove(_meta_path(cache_dir, "model", ".hash"))
            assert _is_cache_valid(cache_dir, "model", source) is True


# ---------------------------------------------------------------------------
# Skip marker tests
# ---------------------------------------------------------------------------

class TestSkipMarker:

    def test_no_skip_marker_means_export_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            assert _should_skip_export(root, "yolov8n") is False

    def test_skip_marker_suppresses_export(self):
        with tempfile.TemporaryDirectory() as root:
            _write_skip_marker(root, "yolov8n")
            assert _should_skip_export(root, "yolov8n") is True

    def test_expired_skip_marker_allows_export(self):
        with tempfile.TemporaryDirectory() as root:
            _write_skip_marker(root, "yolov8n")
            skip_file = _meta_path(root, "yolov8n", ".skip")
            os.utime(skip_file, (0, 0))
            assert _should_skip_export(root, "yolov8n") is False


# ---------------------------------------------------------------------------
# get_coreml_model_path tests
# ---------------------------------------------------------------------------

class TestGetCoremlModelPath:

    def test_returns_cached_mlpackage(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "yolov8n.pt")
            with open(source, "wb") as f:
                f.write(b"fake weights")
            cache_dir = os.path.join(root, ".coreml_cache")
            _make_valid_cache(cache_dir, "yolov8n", source)
            result = get_coreml_model_path(source, root)
        assert result is not None
        assert result.endswith(".mlpackage")

    def test_incomplete_cache_triggers_reexport(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_instance = mock.MagicMock()
        mock_instance.export.return_value = None
        mock_yolo_mod.YOLO = mock.MagicMock(return_value=mock_instance)

        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, ".coreml_cache")
            os.makedirs(cache_dir)
            mlpackage = os.path.join(cache_dir, "yolov8n.mlpackage")
            os.makedirs(mlpackage)

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                get_coreml_model_path("yolov8n.pt", root)

        mock_instance.export.assert_called_once()

    def test_stale_hash_triggers_reexport(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_instance = mock.MagicMock()
        mock_instance.export.return_value = None
        mock_yolo_mod.YOLO = mock.MagicMock(return_value=mock_instance)

        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "yolov8n.pt")
            with open(source, "wb") as f:
                f.write(b"new weights")
            cache_dir = os.path.join(root, ".coreml_cache")
            _make_valid_cache(cache_dir, "yolov8n")
            with open(_meta_path(cache_dir, "yolov8n", ".hash"), "w") as f:
                f.write("old_hash")

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                get_coreml_model_path(source, root)

        mock_instance.export.assert_called_once()

    def test_export_writes_sentinel_hash_and_mlpackage_dir(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "yolov8n.pt")
            with open(source, "wb") as f:
                f.write(b"fake weights")

            export_dir = os.path.join(root, "exported.mlpackage")
            os.makedirs(export_dir)
            with open(os.path.join(export_dir, "model.mlmodel"), "w") as f:
                f.write("fake")

            mock_yolo_mod = types.ModuleType("ultralytics")
            mock_instance = mock.MagicMock()
            mock_instance.export.return_value = export_dir
            mock_yolo_mod.YOLO = mock.MagicMock(return_value=mock_instance)

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path(source, root)

            cache_dir = os.path.join(root, ".coreml_cache")
            assert result == os.path.join(cache_dir, "yolov8n.mlpackage")
            assert os.path.isdir(result)
            assert os.path.isfile(_meta_path(cache_dir, "yolov8n", ".sentinel"))
            assert os.path.isfile(_meta_path(cache_dir, "yolov8n", ".hash"))

    def test_failed_export_writes_skip_marker(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_cls = mock.MagicMock()
        mock_yolo_cls.return_value.export.side_effect = RuntimeError("no coremltools")
        mock_yolo_mod.YOLO = mock_yolo_cls

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path("yolov8n.pt", root)

            assert result is None
            cache_dir = os.path.join(root, ".coreml_cache")
            assert os.path.isfile(_meta_path(cache_dir, "yolov8n", ".skip"))

    def test_skip_marker_prevents_retry(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_cls = mock.MagicMock()
        mock_yolo_mod.YOLO = mock_yolo_cls

        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, ".coreml_cache")
            _write_skip_marker(cache_dir, "yolov8n")

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path("yolov8n.pt", root)

            assert result is None
            mock_yolo_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Integration (structural)
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_ai_device_imports_coreml(self):
        path = os.path.join(_REPO_ROOT, "app", "ai_device.py")
        with open(path) as f:
            assert "coreml_cache" in f.read()

    def test_motion_gate_still_precedes_inference(self):
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            tree = ast.parse(f.read())

        loop_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_ai_detection_loop":
                loop_func = node
                break
        assert loop_func is not None

        motion_gate_line = None
        inference_line = None
        for node in ast.walk(loop_func):
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id == "change_pct":
                    motion_gate_line = node.lineno
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        if isinstance(node.value, ast.Call):
                            inference_line = node.lineno

        assert motion_gate_line is not None
        assert inference_line is not None
        assert motion_gate_line < inference_line

    def test_coreml_cache_in_gitignore(self):
        with open(os.path.join(_REPO_ROOT, ".gitignore")) as f:
            assert ".coreml_cache" in f.read()
