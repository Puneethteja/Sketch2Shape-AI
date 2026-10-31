"""
utils/image_utils.py
=====================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

Shared image utility helpers used across the engine, UI, and test suite.
These are lightweight, dependency-minimal functions that do not belong to
any single pipeline stage.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("sketch2shape.utils.image_utils")

__all__ = [
    "save_image",
    "load_pil",
    "ndarray_to_pil",
    "pil_to_ndarray",
    "stack_side_by_side",
    "draw_overlay_mask",
]


def save_image(
    img: np.ndarray,
    path: Union[str, Path],
    mkdir: bool = True,
) -> Path:
    """
    Save a NumPy RGB (or grayscale) image to disk via OpenCV.

    Parameters
    ----------
    img : np.ndarray
        Image array.  3-channel arrays are assumed to be in **RGB** order
        and are converted to BGR before writing.  2-D arrays are saved
        as grayscale directly.
    path : str | Path
        Destination file path.  The extension determines the codec
        (e.g. ``.png``, ``.jpg``).
    mkdir : bool
        If ``True`` (default), create parent directories as needed.

    Returns
    -------
    Path
        Resolved absolute path of the saved file.

    Raises
    ------
    RuntimeError
        If ``cv2.imwrite`` fails.
    """
    out = Path(path).resolve()
    if mkdir:
        out.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 3 and img.shape[2] == 3:
        write_arr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        write_arr = img

    ok = cv2.imwrite(str(out), write_arr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for path: {out}")

    logger.debug("save_image: wrote %s  shape=%s", out.name, img.shape)
    return out


def load_pil(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from disk as a Pillow ``Image`` (RGB mode).

    Parameters
    ----------
    path : str | Path
        Path to any PIL-supported image file.

    Returns
    -------
    PIL.Image.Image
        Loaded image in RGB mode.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    from PIL import ImageOps
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def ndarray_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert a ``uint8`` NumPy array (H×W×3 RGB or H×W grayscale) to a
    Pillow Image.

    Parameters
    ----------
    arr : np.ndarray
        Source array.

    Returns
    -------
    PIL.Image.Image
    """
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    raise ValueError(f"ndarray_to_pil: unsupported array shape {arr.shape}")


def pil_to_ndarray(img: Image.Image) -> np.ndarray:
    """
    Convert a Pillow Image to a ``uint8`` NumPy array.

    Parameters
    ----------
    img : PIL.Image.Image
        Source Pillow image (any mode).

    Returns
    -------
    np.ndarray
        Shape (H, W) for grayscale or (H, W, C) for colour.
    """
    return np.array(img)


def stack_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    gap: int = 8,
    gap_color: Tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """
    Concatenate two same-height RGB images horizontally with an optional gap.

    Useful for generating before/after comparison panels (e.g. raw sketch
    vs. preprocessed lineart) during debugging and demo mode.

    Parameters
    ----------
    left, right : np.ndarray
        RGB ``uint8`` arrays of identical height.  If heights differ, the
        shorter image is centre-padded with white.
    gap : int
        Width in pixels of the separator column.  Default: ``8``.
    gap_color : tuple[int, int, int]
        RGB colour of the separator.  Default: light grey.

    Returns
    -------
    np.ndarray
        Concatenated image of shape ``(max_h, left_w + gap + right_w, 3)``.
    """
    # Handle grayscale inputs — convert to 3-ch for consistent output
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return np.stack([img] * 3, axis=-1)
        return img

    left  = _ensure_rgb(left)
    right = _ensure_rgb(right)

    h_l, h_r = left.shape[0], right.shape[0]
    max_h     = max(h_l, h_r)

    def _vpad(img: np.ndarray, target_h: int) -> np.ndarray:
        pad = target_h - img.shape[0]
        if pad == 0:
            return img
        top = pad // 2
        bot = pad - top
        return np.pad(
            img, ((top, bot), (0, 0), (0, 0)),
            constant_values=255,
        )

    left  = _vpad(left,  max_h)
    right = _vpad(right, max_h)

    separator = np.full((max_h, gap, 3), gap_color, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def draw_overlay_mask(
    base: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 80, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a binary mask on top of a base RGB image with transparency.

    Useful for visualising lineart masks over the original sketch during
    debugging.

    Parameters
    ----------
    base : np.ndarray
        RGB ``uint8`` image, shape ``(H, W, 3)``.
    mask : np.ndarray
        Binary mask, shape ``(H, W)``, values ``{0, 255}``.
        Non-zero pixels are painted with *color*.
    color : tuple[int, int, int]
        RGB overlay colour.  Default: red ``(255, 80, 80)``.
    alpha : float
        Opacity of the overlay, in ``[0, 1]``.  Default: ``0.45``.

    Returns
    -------
    np.ndarray
        Blended RGB image, same shape and dtype as *base*.
    """
    overlay     = base.copy()
    colour_layer = np.zeros_like(base)
    colour_layer[mask > 0] = color
    overlay     = cv2.addWeighted(base, 1 - alpha, colour_layer, alpha, 0)
    return overlay
