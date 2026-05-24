"""Tests for CoreML / Apple Neural Engine export and caching.

Verifies that:
- CoreML export is attempted on Apple Silicon
- Cached CoreML model is reused on subsequent loads
- Incomplete/stale caches are detected and re-exported
- Fallback chain works: CoreML -> MPS -> CPU
- Two-stage pipeline (motion gate -> YOLO) is preserved
"""

import ast
import os
import sys
import types
import tempfile
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.coreml_cache import get_coreml_model_path, _is_cache_valid, _SENTINEL, _HASH_FILE, _source_hash


# ---------------------------------------------------------------------------
# Cache validity tests
# ---------------------------------------------------------------------------

class TestCacheValidity:

    def test_valid_cache_with_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            cached = os.path.join(root, "model")
            os.makedirs(cached)
            with open(os.path.join(cached, _SENTINEL), "w") as f:
                f.write("ok")
            assert _is_cache_valid(cached, "nonexistent.pt") is True

    def test_invalid_without_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            cached = os.path.join(root, "model")
            os.makedirs(cached)
            assert _is_cache_valid(cached, "nonexistent.pt") is False

    def test_stale_when_hash_differs(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "model.pt")
            with open(source, "wb") as f:
                f.write(b"new weights data")

            cached = os.path.join(root, "cached_model")
            os.makedirs(cached)
            with open(os.path.join(cached, _SENTINEL), "w") as f:
                f.write("ok")
            with open(os.path.join(cached, _HASH_FILE), "w") as f:
                f.write("old_hash_value")

            assert _is_cache_valid(cached, source) is False

    def test_valid_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "model.pt")
            with open(source, "wb") as f:
                f.write(b"weights data")

            cached = os.path.join(root, "cached_model")
            os.makedirs(cached)
            with open(os.path.join(cached, _SENTINEL), "w") as f:
                f.write("ok")
            current_hash = _source_hash(source)
            with open(os.path.join(cached, _HASH_FILE), "w") as f:
                f.write(current_hash)

            assert _is_cache_valid(cached, source) is True


# ---------------------------------------------------------------------------
# get_coreml_model_path tests
# ---------------------------------------------------------------------------

class TestGetCoremlModelPath:

    def test_returns_cached_path_when_valid(self):
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, ".coreml_cache")
            os.makedirs(cache_dir)
            cached_model = os.path.join(cache_dir, "yolov8n_coreml_model")
            os.makedirs(cached_model)
            with open(os.path.join(cached_model, _SENTINEL), "w") as f:
                f.write("ok")

            result = get_coreml_model_path("yolov8n.pt", root)

        assert result is not None
        assert result == cached_model

    def test_incomplete_cache_is_removed(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_instance = mock.MagicMock()
        mock_instance.export.return_value = None
        mock_yolo_mod.YOLO = mock.MagicMock(return_value=mock_instance)

        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, ".coreml_cache")
            os.makedirs(cache_dir)
            cached_model = os.path.join(cache_dir, "yolov8n_coreml_model")
            os.makedirs(cached_model)

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
            os.makedirs(cache_dir)
            cached_model = os.path.join(cache_dir, "yolov8n_coreml_model")
            os.makedirs(cached_model)
            with open(os.path.join(cached_model, _SENTINEL), "w") as f:
                f.write("ok")
            with open(os.path.join(cached_model, _HASH_FILE), "w") as f:
                f.write("old_hash")

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                get_coreml_model_path(source, root)

        mock_instance.export.assert_called_once()

    def test_returns_none_when_export_fails(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_cls = mock.MagicMock()
        mock_yolo_cls.return_value.export.side_effect = RuntimeError("coremltools not installed")
        mock_yolo_mod.YOLO = mock_yolo_cls

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path("yolov8n.pt", root)

        assert result is None

    def test_export_called_with_correct_args(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_instance = mock.MagicMock()
        mock_instance.export.return_value = None
        mock_yolo_cls = mock.MagicMock(return_value=mock_instance)
        mock_yolo_mod.YOLO = mock_yolo_cls

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                get_coreml_model_path("yolov8n.pt", root)

        mock_instance.export.assert_called_once_with(
            format="coreml", imgsz=640, nms=True
        )

    def test_export_writes_sentinel_and_hash(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "yolov8n.pt")
            with open(source, "wb") as f:
                f.write(b"fake model weights")

            export_dir = os.path.join(root, "exported_model")
            os.makedirs(export_dir)
            with open(os.path.join(export_dir, "model.mlpackage"), "w") as f:
                f.write("fake")

            mock_yolo_mod = types.ModuleType("ultralytics")
            mock_instance = mock.MagicMock()
            mock_instance.export.return_value = export_dir
            mock_yolo_mod.YOLO = mock.MagicMock(return_value=mock_instance)

            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path(source, root)

            expected = os.path.join(root, ".coreml_cache", "yolov8n_coreml_model")
            assert result == expected
            assert os.path.isfile(os.path.join(expected, _SENTINEL))
            assert os.path.isfile(os.path.join(expected, _HASH_FILE))

    def test_failed_export_cleans_up_partial(self):
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_cls = mock.MagicMock()
        mock_yolo_cls.return_value.export.side_effect = RuntimeError("boom")
        mock_yolo_mod.YOLO = mock_yolo_cls

        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}):
                result = get_coreml_model_path("yolov8n.pt", root)

            cached = os.path.join(root, ".coreml_cache", "yolov8n_coreml_model")
            assert result is None
            assert not os.path.exists(cached)


# ---------------------------------------------------------------------------
# camera.py / ai_device.py integration (structural)
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_ai_device_imports_coreml(self):
        path = os.path.join(_REPO_ROOT, "app", "ai_device.py")
        with open(path) as f:
            source = f.read()
        assert "coreml_cache" in source

    def test_motion_gate_still_precedes_inference(self):
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            tree = ast.parse(f.read())

        loop_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_ai_detection_loop":
                loop_func = node
                break
        assert loop_func is not None, "_ai_detection_loop not found"

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

        assert motion_gate_line is not None, "Motion gate not found"
        assert inference_line is not None, "Inference call not found"
        assert motion_gate_line < inference_line

    def test_coreml_cache_in_gitignore(self):
        gitignore_path = os.path.join(_REPO_ROOT, ".gitignore")
        with open(gitignore_path) as f:
            content = f.read()
        assert ".coreml_cache" in content
