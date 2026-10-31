"""
tests/test_preprocessing.py
============================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Unit tests for engine/preprocessing.py.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.preprocessing import (
    PAD_COLOR,
    TARGET_SIZE,
    bilateral_filter,
    ingest,
    load_image,
    pad_and_resize,
    preprocess,
    to_lineart,
    to_rgb,
)

@pytest.fixture
def rgb_image() -> np.ndarray:
    """Synthetic 200×300 RGB image (random pixels)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)

@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Synthetic 150×150 grayscale image."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (150, 150), dtype=np.uint8)

@pytest.fixture
def rgba_image() -> np.ndarray:
    """Synthetic 100×100 RGBA image."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (100, 100, 4), dtype=np.uint8)

@pytest.fixture
def sketch_png(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    """Write a synthetic RGB PNG to a temp file and return its path."""
    path = tmp_path / "sketch.png"
    Image.fromarray(rgb_image).save(path)
    return path

@pytest.fixture
def sketch_jpg(tmp_path: Path, rgb_image: np.ndarray) -> Path:
    """Write a synthetic JPEG to a temp file and return its path."""
    path = tmp_path / "sketch.jpg"
    Image.fromarray(rgb_image).save(path, quality=95)
    return path

@pytest.fixture
def gray_png(tmp_path: Path, grayscale_image: np.ndarray) -> Path:
    """Write a synthetic grayscale PNG to a temp file."""
    path = tmp_path / "gray.png"
    Image.fromarray(grayscale_image, mode="L").save(path)
    return path

@pytest.fixture
def rgba_png(tmp_path: Path, rgba_image: np.ndarray) -> Path:
    """Write a synthetic RGBA PNG to a temp file."""
    path = tmp_path / "rgba.png"
    Image.fromarray(rgba_image, mode="RGBA").save(path)
    return path

@pytest.fixture
def base64_png(sketch_png: Path) -> str:
    """Generate a base64 data-URL representing a drawing canvas payload."""
    with open(sketch_png, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

class TestIngest:
    def test_loads_png(self, sketch_png: Path):
        img = ingest(sketch_png)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.dtype == np.uint8

    def test_loads_jpeg(self, sketch_jpg: Path):
        img = ingest(sketch_jpg)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3

    def test_loads_grayscale_png(self, gray_png: Path):
        img = ingest(gray_png)
        assert img.ndim == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest("nonexistent_sketch.png")

    def test_accepts_path_object(self, sketch_png: Path):
        img = ingest(sketch_png)
        assert img is not None

    def test_accepts_string_path(self, sketch_png: Path):
        img = ingest(str(sketch_png))
        assert img is not None

    def test_accepts_pil_image(self, rgb_image: np.ndarray):
        pil_img = Image.fromarray(rgb_image)
        img = ingest(pil_img)
        assert isinstance(img, np.ndarray)
        np.testing.assert_array_equal(img, rgb_image)

    def test_accepts_raw_bytes(self, sketch_png: Path):
        with open(sketch_png, "rb") as f:
            raw_bytes = f.read()
        img = ingest(raw_bytes)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_accepts_canvas_base64_dataurl(self, base64_png: str):
        img = ingest(base64_png)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3

    def test_accepts_raw_numpy_array(self, rgb_image: np.ndarray):
        img = ingest(rgb_image)
        assert img is rgb_image

    def test_rejects_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            ingest(12345)

    def test_backward_compatible_load_image(self, sketch_png: Path):
        img = load_image(sketch_png)
        assert isinstance(img, np.ndarray)

class TestToRGB:
    def test_rgb_passthrough(self, rgb_image: np.ndarray):
        out = to_rgb(rgb_image)
        assert out.shape == rgb_image.shape
        assert out.shape[2] == 3

    def test_grayscale_to_rgb(self, grayscale_image: np.ndarray):
        out = to_rgb(grayscale_image)
        assert out.ndim == 3
        assert out.shape[2] == 3

        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 1], out[:, :, 2])

    def test_rgba_to_rgb(self, rgba_image: np.ndarray):
        out = to_rgb(rgba_image)
        assert out.ndim == 3
        assert out.shape[2] == 3

    def test_single_channel_to_rgb(self):
        img = np.full((50, 50, 1), 128, dtype=np.uint8)
        out = to_rgb(img)
        assert out.shape == (50, 50, 3)

    def test_output_dtype_is_uint8(self, grayscale_image: np.ndarray):
        out = to_rgb(grayscale_image)
        assert out.dtype == np.uint8

    def test_float_input_normalised(self):
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        out = to_rgb(img)
        assert out.dtype == np.uint8

        assert np.all(out == 127)

    def test_invalid_channels_raises(self):
        img = np.zeros((10, 10, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="channels"):
            to_rgb(img)

class TestPadAndResize:
    def test_output_shape(self, rgb_image: np.ndarray):
        out = pad_and_resize(rgb_image)
        h, w = TARGET_SIZE[1], TARGET_SIZE[0]
        assert out.shape == (h, w, 3)

    def test_custom_target_size(self, rgb_image: np.ndarray):
        out = pad_and_resize(rgb_image, target_size=(256, 256))
        assert out.shape == (256, 256, 3)

    def test_square_input_no_padding(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        out = pad_and_resize(img, target_size=(512, 512))
        assert out.shape == (512, 512, 3)

    def test_landscape_padding(self):
        """Wide image should have top/bottom white padding."""
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        out = pad_and_resize(img, target_size=(512, 512), pad_color=(255, 255, 255))

        assert tuple(out[0, 0]) == (255, 255, 255)

    def test_portrait_padding(self):
        """Tall image should have left/right padding."""
        img = np.zeros((400, 100, 3), dtype=np.uint8)
        out = pad_and_resize(img, target_size=(512, 512), pad_color=(255, 255, 255))
        assert tuple(out[0, 0]) == (255, 255, 255)

    def test_dtype_preserved(self, rgb_image: np.ndarray):
        out = pad_and_resize(rgb_image)
        assert out.dtype == np.uint8

class TestBilateralFilter:
    def test_output_shape_unchanged(self, rgb_image: np.ndarray):
        out = bilateral_filter(rgb_image)
        assert out.shape == rgb_image.shape

    def test_output_dtype_uint8(self, rgb_image: np.ndarray):
        out = bilateral_filter(rgb_image)
        assert out.dtype == np.uint8

    def test_uniform_image_unchanged(self):
        """A flat-colour image should not change under bilateral filtering."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = bilateral_filter(img)
        np.testing.assert_array_equal(out, img)

    def test_pixel_range_preserved(self, rgb_image: np.ndarray):
        out = bilateral_filter(rgb_image)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_custom_sigma_values(self, rgb_image: np.ndarray):
        out = bilateral_filter(rgb_image, d=5, sigma_color=50, sigma_space=50)
        assert out.shape == rgb_image.shape

class TestToLineart:
    def test_lineart_shape_and_binary_values(self, rgb_image: np.ndarray):
        lineart = to_lineart(rgb_image)
        assert lineart.shape == rgb_image.shape[:2]
        assert lineart.dtype == np.uint8

        unique_vals = set(np.unique(lineart))
        assert unique_vals.issubset({0, 255})

    def test_lineart_invert_flag(self, rgb_image: np.ndarray):
        inv = to_lineart(rgb_image, invert=True)
        non_inv = to_lineart(rgb_image, invert=False)
        np.testing.assert_array_equal(inv, 255 - non_inv)

    def test_invalid_block_size_raises(self, rgb_image: np.ndarray):
        with pytest.raises(ValueError, match="odd integer"):
            to_lineart(rgb_image, block_size=10)
        with pytest.raises(ValueError, match="odd integer"):
            to_lineart(rgb_image, block_size=1)

class TestPreprocess:
    def test_returns_ndarray_when_no_tensor(self, sketch_png: Path):
        result = preprocess(sketch_png, return_tensor=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_returns_tensor_shape(self, sketch_png: Path):
        torch = pytest.importorskip("torch")
        result = preprocess(sketch_png, return_tensor=True)
        assert isinstance(result, torch.Tensor)

        assert result.shape == (3, TARGET_SIZE[1], TARGET_SIZE[0])

    def test_tensor_value_range(self, sketch_png: Path):
        torch = pytest.importorskip("torch")
        result = preprocess(sketch_png, return_tensor=True)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_tensor_dtype_float32(self, sketch_png: Path):
        torch = pytest.importorskip("torch")
        result = preprocess(sketch_png, return_tensor=True)
        assert result.dtype == torch.float32

    def test_grayscale_input(self, gray_png: Path):
        result = preprocess(gray_png, return_tensor=False)
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_rgba_input(self, rgba_png: Path):
        result = preprocess(rgba_png, return_tensor=False)
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_drawing_canvas_base64_payload(self, base64_png: str):
        result = preprocess(base64_png, return_tensor=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_drawing_canvas_raw_bytes(self, sketch_png: Path):
        with open(sketch_png, "rb") as f:
            data = f.read()
        result = preprocess(data, return_tensor=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_return_lineart_mask(self, sketch_png: Path):
        mask = preprocess(sketch_png, return_lineart=True, return_tensor=False)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (TARGET_SIZE[1], TARGET_SIZE[0])
        assert set(np.unique(mask)).issubset({0, 255})

    def test_return_lineart_tensor(self, sketch_png: Path):
        torch = pytest.importorskip("torch")
        tensor = preprocess(sketch_png, return_lineart=True, return_tensor=True)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, TARGET_SIZE[1], TARGET_SIZE[0])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            preprocess("does_not_exist.png")
