"""
tests/test_image_utils.py
=========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Unit tests for utils/image_utils.py.

Run with:
    pytest tests/test_image_utils.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.image_utils import (
    draw_overlay_mask,
    load_pil,
    ndarray_to_pil,
    pil_to_ndarray,
    save_image,
    stack_side_by_side,
)

@pytest.fixture
def sample_rgb() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

def test_save_and_load_pil(tmp_path: Path, sample_rgb: np.ndarray):
    save_path = tmp_path / "subdir" / "test_img.png"
    saved = save_image(sample_rgb, save_path)
    assert saved.exists()

    pil_img = load_pil(saved)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (128, 128)

def test_load_pil_not_found():
    with pytest.raises(FileNotFoundError):
        load_pil("missing_file_random.png")

def test_ndarray_pil_conversions(sample_rgb: np.ndarray):
    pil_img = ndarray_to_pil(sample_rgb)
    assert isinstance(pil_img, Image.Image)
    back_to_arr = pil_to_ndarray(pil_img)
    np.testing.assert_array_equal(sample_rgb, back_to_arr)

def test_stack_side_by_side(sample_rgb: np.ndarray):
    left = sample_rgb
    right = sample_rgb.copy()
    stacked = stack_side_by_side(left, right, gap=10)
    assert stacked.shape == (128, 128 + 10 + 128, 3)

def test_draw_overlay_mask(sample_rgb: np.ndarray):
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[30:60, 30:60] = 255
    overlay = draw_overlay_mask(sample_rgb, mask, color=(255, 0, 0), alpha=0.5)
    assert overlay.shape == sample_rgb.shape
    assert overlay.dtype == np.uint8
