"""
tests/test_segmentation.py
==========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Unit tests for utils/segmentation.py (rembg integration, alpha masking,
and aspect-ratio bounding-box centering).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.segmentation import (
    crop_and_center_subject,
    extract_alpha_mask,
    get_foreground_bbox,
    isolate_and_center,
    remove_background,
)


@pytest.fixture
def synthetic_subject_rgba() -> Image.Image:
    """Create a 300x200 RGBA image with a 100x80 active rectangle in the corner."""
    arr = np.zeros((200, 300, 4), dtype=np.uint8)
    # Background is transparent (alpha = 0)
    # Draw a non-centered red subject with alpha = 255
    arr[30:110, 40:140, 0] = 220   # R
    arr[30:110, 40:140, 1] = 50    # G
    arr[30:110, 40:140, 2] = 50    # B
    arr[30:110, 40:140, 3] = 255   # A (fully opaque)
    return Image.fromarray(arr, mode="RGBA")


def test_get_foreground_bbox(synthetic_subject_rgba: Image.Image):
    alpha = np.array(synthetic_subject_rgba.split()[-1])
    bbox = get_foreground_bbox(alpha)
    assert bbox is not None
    x_min, y_min, x_max, y_max = bbox
    assert x_min == 40
    assert y_min == 30
    assert x_max == 139
    assert y_max == 109


def test_get_foreground_bbox_empty():
    empty_alpha = np.zeros((100, 100), dtype=np.uint8)
    bbox = get_foreground_bbox(empty_alpha)
    assert bbox is None


def test_crop_and_center_subject(synthetic_subject_rgba: Image.Image):
    centered = crop_and_center_subject(
        synthetic_subject_rgba,
        target_size=(512, 512),
        padding_ratio=0.1,
    )
    assert isinstance(centered, Image.Image)
    assert centered.size == (512, 512)
    assert centered.mode == "RGBA"

    # Verify subject is placed roughly around center
    alpha = np.array(centered.split()[-1])
    center_patch = alpha[200:312, 200:312]
    assert np.any(center_patch > 0), "Subject was not centered in canvas"

    # Verify outer border remains transparent due to padding
    assert np.all(alpha[:20, :] == 0), "Padding region was not preserved"
    assert np.all(alpha[-20:, :] == 0), "Padding region was not preserved"


def test_remove_background_and_extract_alpha():
    # Synthetic sketch: white background with dark black scribble in the center
    arr = np.full((128, 128, 3), 255, dtype=np.uint8)
    arr[40:88, 40:88] = 0  # Black square

    # Extract alpha mask
    mask = extract_alpha_mask(arr, threshold=10)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8

    # Foreground should be detected
    assert np.any(mask == 255)


def test_isolate_and_center_pipeline():
    # Draw simple subject on light background
    img = Image.new("RGB", (256, 256), (250, 250, 250))
    for x in range(80, 160):
        for y in range(80, 160):
            img.putpixel((x, y), (20, 20, 20))

    result = isolate_and_center(
        img,
        target_size=(512, 512),
        padding_ratio=0.1,
        neutral_background=True,
    )
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)
    assert result.mode == "RGB"
