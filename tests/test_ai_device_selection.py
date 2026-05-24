"""Tests for Apple Silicon MPS device selection and AI model loading.

Verifies that:
- MPS is selected when available on Apple Silicon
- CPU fallback works when MPS is unavailable or fails
- Models are loaded onto the correct device
- The two-stage pipeline (OpenCV motion gate -> YOLO) is preserved:
  YOLO inference only fires when the motion threshold is exceeded
"""

import ast
import sys
import os
import types
from unittest import mock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from app.ai_device import select_device, get_shared_model, clear_models


# ---------------------------------------------------------------------------
# select_device() tests
# ---------------------------------------------------------------------------

class TestSelectDevice:

    def test_returns_mps_when_available(self):
        mock_backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        )
        mock_torch = types.SimpleNamespace(backends=mock_backends)

        with mock.patch.dict(sys.modules, {"torch": mock_torch}):
            assert select_device() == "mps"

    def test_returns_cpu_when_mps_unavailable(self):
        mock_backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        mock_torch = types.SimpleNamespace(backends=mock_backends)

        with mock.patch.dict(sys.modules, {"torch": mock_torch}):
            assert select_device() == "cpu"

    def test_returns_cpu_when_torch_import_fails(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            assert select_device() == "cpu"

    def test_returns_cpu_when_mps_attr_missing(self):
        mock_backends = types.SimpleNamespace()
        mock_torch = types.SimpleNamespace(backends=mock_backends)

        with mock.patch.dict(sys.modules, {"torch": mock_torch}):
            assert select_device() == "cpu"

    def test_returns_cpu_when_is_available_raises(self):
        def _boom():
            raise RuntimeError("MPS not supported")

        mock_backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=_boom)
        )
        mock_torch = types.SimpleNamespace(backends=mock_backends)

        with mock.patch.dict(sys.modules, {"torch": mock_torch}):
            assert select_device() == "cpu"


# ---------------------------------------------------------------------------
# get_shared_model() tests
# ---------------------------------------------------------------------------

class TestGetSharedModel:

    def setup_method(self):
        clear_models()

    def _make_mock_yolo(self):
        instance = mock.MagicMock()
        instance.device = "cpu"

        def _to(device):
            instance.device = device
            return instance

        instance.to = _to
        klass = mock.MagicMock(return_value=instance)
        return klass, instance

    def test_model_loaded_on_mps(self):
        yolo_cls, yolo_inst = self._make_mock_yolo()
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_mod.YOLO = yolo_cls

        with mock.patch("app.ai_device.select_device", return_value="mps"), \
             mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}), \
             mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            model = get_shared_model("yolov8n.pt")

        assert model is yolo_inst
        assert yolo_inst.device == "mps"

    def test_model_loaded_on_cpu_fallback(self):
        yolo_cls, yolo_inst = self._make_mock_yolo()
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_mod.YOLO = yolo_cls

        with mock.patch("app.ai_device.select_device", return_value="cpu"), \
             mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}), \
             mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            model = get_shared_model("yolov8n.pt")

        assert model is yolo_inst
        assert yolo_inst.device == "cpu"

    def test_model_cached_across_calls(self):
        yolo_cls, yolo_inst = self._make_mock_yolo()
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_mod.YOLO = yolo_cls

        with mock.patch("app.ai_device.select_device", return_value="cpu"), \
             mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}), \
             mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            m1 = get_shared_model("yolov8n.pt")
            m2 = get_shared_model("yolov8n.pt")

        assert m1 is m2
        yolo_cls.assert_called_once()

    def test_different_models_cached_separately(self):
        call_count = 0

        def _make_instance(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            inst = mock.MagicMock()
            inst.device = "cpu"
            inst.to = lambda d: setattr(inst, "device", d) or inst
            inst._id = call_count
            return inst

        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_mod.YOLO = _make_instance

        with mock.patch("app.ai_device.select_device", return_value="cpu"), \
             mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}), \
             mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            m1 = get_shared_model("yolov8n.pt")
            m2 = get_shared_model("yolov8s.pt")

        assert m1 is not m2
        assert call_count == 2

    def test_mps_failure_falls_back_to_cpu(self):
        """If model.to('mps') raises, get_shared_model retries on CPU."""
        instance = mock.MagicMock()
        instance.device = "cpu"
        call_log = []

        def _to(device):
            call_log.append(device)
            if device == "mps":
                raise RuntimeError("MPS kernel not available")
            instance.device = device
            return instance

        instance.to = _to
        yolo_cls = mock.MagicMock(return_value=instance)
        mock_yolo_mod = types.ModuleType("ultralytics")
        mock_yolo_mod.YOLO = yolo_cls

        with mock.patch("app.ai_device.select_device", return_value="mps"), \
             mock.patch.dict(sys.modules, {"ultralytics": mock_yolo_mod}), \
             mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            model = get_shared_model("yolov8n.pt")

        assert model is instance
        assert instance.device == "cpu"
        assert call_log == ["mps", "cpu"]


# ---------------------------------------------------------------------------
# Two-stage pipeline preservation tests
#
# The architecture intentionally keeps CPU usage low by using a cheap
# OpenCV pixel-difference check as a gate before expensive YOLO inference.
# ---------------------------------------------------------------------------

def _simulate_motion_gate(change_pct, motion_threshold, model, frame, conf, classes):
    """Replicate the motion-gate logic from camera._ai_detection_loop."""
    if change_pct >= motion_threshold:
        return model(frame, verbose=False, conf=conf, classes=classes,
                     device=model.device)
    return None


class TestTwoStagePipelinePreserved:

    def test_no_motion_means_no_inference(self):
        """Identical frames produce zero pixel change — YOLO must NOT fire."""
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")

        mock_model = mock.MagicMock()
        mock_model.device = "cpu"

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        prev_gray = gray.copy()

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        change_pct = (cv2.countNonZero(thresh) / thresh.size) * 100.0

        sensitivity = 50
        motion_threshold = max(0.1, 5.0 - (sensitivity / 100.0) * 5.0)

        result = _simulate_motion_gate(
            change_pct, motion_threshold, mock_model, frame, 0.4, [0]
        )
        assert result is None
        mock_model.assert_not_called()

    def test_full_motion_triggers_inference(self):
        """Black-to-white frame change must exceed threshold — YOLO fires."""
        np = pytest.importorskip("numpy")
        cv2 = pytest.importorskip("cv2")

        mock_model = mock.MagicMock()
        mock_model.device = "mps"
        mock_result = mock.MagicMock()
        mock_result.boxes = []
        mock_model.return_value = [mock_result]

        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.full((480, 640, 3), 255, dtype=np.uint8)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        frame_delta = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        change_pct = (cv2.countNonZero(thresh) / thresh.size) * 100.0

        sensitivity = 50
        motion_threshold = max(0.1, 5.0 - (sensitivity / 100.0) * 5.0)

        assert change_pct >= motion_threshold

        result = _simulate_motion_gate(
            change_pct, motion_threshold, mock_model, frame2, 0.4, [0]
        )
        assert result is not None
        mock_model.assert_called_once()

    def test_motion_gate_precedes_inference_in_source(self):
        """Structural: motion threshold check must appear before model() call
        in _ai_detection_loop, verified via AST to be refactoring-resistant."""
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
                left = node.left
                if isinstance(left, ast.Name) and left.id == "change_pct":
                    motion_gate_line = node.lineno

            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "results":
                        if isinstance(node.value, ast.Call):
                            func = node.value.func
                            if isinstance(func, ast.Name) and func.id == "model":
                                inference_line = node.lineno

        assert motion_gate_line is not None, "Motion gate comparison not found in AST"
        assert inference_line is not None, "model() inference call not found in AST"
        assert motion_gate_line < inference_line, (
            f"Motion gate (line {motion_gate_line}) must precede inference "
            f"(line {inference_line})"
        )

    def test_device_forwarded_to_inference_call(self):
        """The model() call must pass device=model.device, verified via AST."""
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            tree = ast.parse(f.read())

        loop_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_ai_detection_loop":
                loop_func = node
                break
        assert loop_func is not None

        found_device_kwarg = False
        for node in ast.walk(loop_func):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "model":
                    for kw in node.keywords:
                        if kw.arg == "device":
                            found_device_kwarg = True
                            break

        assert found_device_kwarg, (
            "model() call must include device= keyword argument"
        )

    def test_camera_imports_from_ai_device(self):
        """camera.py must use the shared ai_device module, not inline logic."""
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            tree = ast.parse(f.read())

        found_import = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and node.module == "ai_device" and node.level == 1):
                found_import = True
                break

        assert found_import, "camera.py must import from .ai_device"
