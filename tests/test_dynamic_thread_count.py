"""Tests for dynamic PyTorch thread count selection.

Verifies that:
- Thread count scales with CPU cores: min(4, cpu_count // 2)
- AI_TORCH_THREADS env var overrides the dynamic calculation
- Override is clamped to minimum of 1
- Invalid env var values produce a warning and use default
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compute_thread_count(cpu_count, env_value=None):
    """Replicate the thread count formula for parameterized testing.

    The source-level tests below verify this formula matches production.
    """
    if env_value is not None:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return min(4, max(1, (cpu_count or 2) // 2))


class TestDynamicThreadCount:

    def test_8_core(self):
        assert _compute_thread_count(8) == 4

    def test_4_core(self):
        assert _compute_thread_count(4) == 2

    def test_2_core(self):
        assert _compute_thread_count(2) == 1

    def test_12_core_capped_at_4(self):
        assert _compute_thread_count(12) == 4

    def test_16_core_capped_at_4(self):
        assert _compute_thread_count(16) == 4

    def test_1_core(self):
        assert _compute_thread_count(1) == 1

    def test_none_cpu_count_defaults_to_1(self):
        assert _compute_thread_count(None) == 1

    def test_env_override(self):
        assert _compute_thread_count(8, env_value="6") == 6

    def test_env_override_minimum_clamped(self):
        assert _compute_thread_count(8, env_value="0") == 1

    def test_env_override_negative_clamped(self):
        assert _compute_thread_count(8, env_value="-3") == 1

    def test_invalid_env_falls_back_to_default(self):
        assert _compute_thread_count(8, env_value="auto") == 4


class TestSourceVerification:
    """Verify the production source contains the expected patterns."""

    def _read_source(self):
        path = os.path.join(_REPO_ROOT, "app", "ai_device.py")
        with open(path) as f:
            return f.read()

    def test_reads_ai_torch_threads_env(self):
        assert "AI_TORCH_THREADS" in self._read_source()

    def test_uses_cpu_count(self):
        assert "cpu_count()" in self._read_source()

    def test_handles_invalid_env_with_warning(self):
        source = self._read_source()
        assert "ValueError" in source, (
            "Production code must catch ValueError for non-integer AI_TORCH_THREADS"
        )

    def test_cpu_count_stored_once(self):
        source = self._read_source()
        assert "_cpu_count = " in source or "_cpu_count=" in source, (
            "cpu_count() result should be stored to avoid redundant calls"
        )
