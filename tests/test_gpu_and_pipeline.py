"""
tests/test_gpu_and_pipeline.py
==============================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Unit tests for engine/check_gpu.py and engine/test_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.check_gpu import (
    check_cuda_availability,
    check_fp16_support,
    get_gpu_info,
    run_gpu_diagnostic,
)
from engine.test_pipeline import check_dependencies, verify_pipeline_stack


def test_cuda_availability_check():
    """Verify that check_cuda_availability returns a boolean and status string."""
    is_avail, msg = check_cuda_availability()
    assert isinstance(is_avail, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_get_gpu_info_structure():
    """Verify get_gpu_info returns standard dictionary fields."""
    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "vram_total_gb" in info
    assert "vram_free_gb" in info


def test_fp16_support_check():
    """Verify check_fp16_support returns a boolean and message."""
    fp16_ok, msg = check_fp16_support()
    assert isinstance(fp16_ok, bool)
    assert isinstance(msg, str)


def test_run_gpu_diagnostic():
    """Verify full diagnostic report structure."""
    report = run_gpu_diagnostic()
    assert "cuda_available" in report
    assert "gpu_info" in report
    assert "fp16_supported" in report
    assert "production_ready" in report


def test_pipeline_check_dependencies():
    """Verify check_dependencies returns package status dictionary."""
    deps = check_dependencies()
    assert "torch" in deps
    assert "cv2" in deps
    assert "PIL" in deps
    assert isinstance(deps["torch"][0], bool)


def test_pipeline_verify_stack_dry_run():
    """Verify verify_pipeline_stack dry_run returns a structured status."""
    res = verify_pipeline_stack(dry_run=True)
    assert isinstance(res, dict)
    assert "success" in res
