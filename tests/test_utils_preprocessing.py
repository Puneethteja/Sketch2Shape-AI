"""
tests/test_utils_preprocessing.py
=================================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Verifies that utils/preprocessing.py exports all required functions and mirrors
engine/preprocessing.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils.preprocessing as up

def test_utils_preprocessing_exports():
    """Verify all required API symbols are exported from utils.preprocessing."""
    expected = [
        "ingest",
        "load_image",
        "to_rgb",
        "pad_and_resize",
        "bilateral_filter",
        "to_lineart",
        "preprocess",
        "TARGET_SIZE",
        "PAD_COLOR",
    ]
    for symbol in expected:
        assert hasattr(up, symbol), f"Missing {symbol} in utils.preprocessing"

def test_utils_preprocessing_pipeline_execution():
    """Verify that preprocess() called via utils.preprocessing works end-to-end."""
    rng = np.random.default_rng(42)
    fake_img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)

    result = up.preprocess(fake_img, return_tensor=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (512, 512, 3)

    lineart = up.preprocess(fake_img, return_lineart=True, return_tensor=False)
    assert isinstance(lineart, np.ndarray)
    assert lineart.shape == (512, 512)
