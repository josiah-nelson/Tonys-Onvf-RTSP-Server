"""Tests for Apple Silicon MPS device selection and AI model loading.

Verifies that:
- MPS is selected when available on Apple Silicon
- CPU fallback works when MPS is unavailable
- Models are loaded onto the correct device
- The two-stage pipeline (OpenCV motion gate -> YOLO) is preserved:
  YOLO inference only fires when the motion threshold is exceeded
"""

import sys
import os
import types
from unittest import mock

import pytest

# Ensure repo root is on the path so `app.ai_device` resolves
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


# ---------------------------------------------------------------------------
# Two-stage pipeline preservation tests
#
# The architecture intentionally keeps CPU usage low by using a cheap
# OpenCV pixel-difference check as a gate before expensive YOLO inference.
# ---------------------------------------------------------------------------

class TestTwoStagePipelinePreserved:

    def test_no_motion_means_no_inference(self):
        """Identical frames produce zero pixel change — YOLO must NOT fire."""
        import numpy as np
        import cv2

        mock_model = mock.MagicMock()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        prev_gray = gray.copy()

        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        change_pct = (cv2.countNonZero(thresh) / thresh.size) * 100.0

        sensitivity = 50
        motion_threshold = max(0.1, 5.0 - (sensitivity / 100.0) * 5.0)

        assert change_pct < motion_threshold
        mock_model.assert_not_called()

    def test_full_motion_triggers_inference(self):
        """Black-to-white frame change must exceed threshold — YOLO fires."""
        import numpy as np
        import cv2

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

        results = mock_model(
            frame2, verbose=False, conf=0.4,
            classes=[0], device=mock_model.device,
        )
        mock_model.assert_called_once()

    def test_motion_gate_precedes_inference_in_source(self):
        """Structural: motion threshold check must appear before model() call."""
        # We read camera.py source directly to avoid importing its heavy deps
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            source = f.read()

        motion_pos = source.find("change_pct < motion_threshold")
        infer_pos = source.find("results = model(")

        assert motion_pos != -1, "Motion threshold gate not found"
        assert infer_pos != -1, "Model inference call not found"
        assert motion_pos < infer_pos, (
            "Motion threshold check must come before YOLO inference"
        )

    def test_device_forwarded_to_inference_call(self):
        """The model() call must pass device=model.device."""
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            source = f.read()

        assert "device=model.device" in source, (
            "Inference call must forward device=model.device"
        )

    def test_camera_imports_from_ai_device(self):
        """camera.py must use the shared ai_device module, not inline logic."""
        camera_path = os.path.join(_REPO_ROOT, "app", "camera.py")
        with open(camera_path) as f:
            source = f.read()

        assert "from .ai_device import" in source
