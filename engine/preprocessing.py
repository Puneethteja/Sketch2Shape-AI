"""
engine/preprocessing.py
========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

Production-grade preprocessing pipeline for raw sketch images and drawing
canvas payloads.

Pipeline Stages
───────────────
1.  ingest()            – Universal ingestion: file path, bytes, base64,
                          PIL.Image, or raw np.ndarray  → uint8 RGB array
2.  to_rgb()            – Normalise any channel count → 3-ch RGB uint8
3.  pad_and_resize()    – Letterbox-pad + resize to TARGET_SIZE (512×512)
                          while preserving aspect ratio
4.  bilateral_filter()  – Edge-preserving bilateral denoising
5.  to_lineart()        – Adaptive threshold + morphological cleanup
                          → clean binary lineart / scribble mask
6.  preprocess()        – End-to-end convenience wrapper

Quick Usage
───────────
>>> from engine.preprocessing import preprocess
>>> # From a file path
>>> tensor = preprocess("sketch.png")                        # (3,512,512) float32
>>> # From raw bytes (e.g. HTTP upload)
>>> tensor = preprocess(open("sketch.png","rb").read())
>>> # From a base64 data-URL (drawing canvas payload)
>>> tensor = preprocess("data:image/png;base64,iVBOR...")
>>> # Get the lineart mask instead
>>> mask = preprocess("sketch.png", return_lineart=True, return_tensor=False)
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    import torch

import cv2
import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.preprocessing")

__all__ = [
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

TARGET_SIZE: Tuple[int, int] = (512, 512)
PAD_COLOR: Tuple[int, int, int] = (255, 255, 255)

BILATERAL_D: int = 9
BILATERAL_SIGMA_COLOR: float = 75.0
BILATERAL_SIGMA_SPACE: float = 75.0

LINEART_BLOCK_SIZE: int = 11
LINEART_C: int = 2
LINEART_MORPH_KERNEL: int = 2

ImageInput = Union[str, Path, bytes, np.ndarray, Image.Image]

def ingest(source: ImageInput) -> np.ndarray:
    """
    Ingest an image from any supported source and return a raw NumPy array.

    Accepted *source* types
    -----------------------
    * ``str`` / ``Path``  — file-system path (``.png``, ``.jpg``, ``.bmp``, …)
    * ``str``             — base64 data-URL  (``data:image/png;base64,…``)
    * ``bytes``           — raw image bytes (e.g. from an HTTP multipart upload
                            or a drawing-canvas ``Blob``)
    * ``np.ndarray``      — already-decoded array (passed through untouched)
    * ``PIL.Image.Image`` — Pillow image object

    Parameters
    ----------
    source : ImageInput
        The raw image payload in any of the formats listed above.

    Returns
    -------
    np.ndarray
        Shape (H, W) or (H, W, C), dtype ``uint8``.
        Channel order is **RGB** when C == 3, **RGBA** when C == 4.

    Raises
    ------
    FileNotFoundError
        If *source* is a file path that does not exist on disk.
    ValueError
        If *source* cannot be decoded as a valid image, or is an unsupported
        type.
    """

    if isinstance(source, np.ndarray):
        logger.debug("ingest: received np.ndarray directly — passthrough")
        return source

    if isinstance(source, Image.Image):
        logger.debug("ingest: received PIL.Image — converting to ndarray")
        return np.array(source)

    if isinstance(source, (bytes, bytearray)):
        logger.info("ingest: decoding raw image bytes (%d B)", len(source))
        return _pil_to_array(Image.open(io.BytesIO(source)))

    if isinstance(source, (str, Path)):
        src_str = str(source)

        if src_str.startswith("data:"):
            logger.info("ingest: detected base64 data-URL — decoding")
            return _decode_dataurl(src_str)

        path = Path(src_str)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path.resolve()}")
        logger.info("ingest: loading file '%s'", path.name)
        return _pil_to_array(Image.open(path))

    raise ValueError(
        f"Unsupported source type: {type(source).__name__}. "
        "Expected str, Path, bytes, np.ndarray, or PIL.Image."
    )

def _pil_to_array(pil_img: Image.Image) -> np.ndarray:
    """Force-decode a Pillow image and return it as a uint8 NumPy array."""
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img.load()
    arr = np.array(pil_img)
    logger.debug(
        "_pil_to_array: shape=%s  dtype=%s  mode=%s",
        arr.shape, arr.dtype, pil_img.mode,
    )
    return arr

def _decode_dataurl(data_url: str) -> np.ndarray:
    """
    Decode a base64 data-URL string into a NumPy array.

    Handles both ``data:image/png;base64,…`` and plain base64 strings
    (no ``data:`` prefix) as a fallback.
    """
    try:

        header, b64_data = data_url.split(",", 1)
    except ValueError:
        b64_data = data_url

    raw_bytes = base64.b64decode(b64_data)
    return _pil_to_array(Image.open(io.BytesIO(raw_bytes)))

load_image = ingest

def to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert *img* to a 3-channel ``uint8`` RGB array regardless of source
    mode (grayscale, RGBA, palette, float, …).

    Parameters
    ----------
    img : np.ndarray
        Raw image array as returned by :func:`ingest`.

    Returns
    -------
    np.ndarray
        Shape (H, W, 3), dtype ``uint8``, channel order **RGB**.

    Raises
    ------
    ValueError
        For arrays with an unsupported number of dimensions or channels.
    """

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) \
            if img.max() <= 1.0 else img.astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
        logger.debug("to_rgb: grayscale → RGB")
        return img

    if img.ndim != 3:
        raise ValueError(
            f"to_rgb: unsupported array ndim={img.ndim}. Expected 2 or 3."
        )

    channels = img.shape[2]

    if channels == 1:

        img = np.concatenate([img] * 3, axis=-1)
        logger.debug("to_rgb: single-channel → RGB")

    elif channels == 3:
        pass

    elif channels == 4:

        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb   = img[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        img   = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
        logger.debug("to_rgb: RGBA composited onto white → RGB")

    else:
        raise ValueError(
            f"to_rgb: unexpected channel count={channels}. "
            "Expected 1, 3, or 4 channels."
        )

    return img

def pad_and_resize(
    img: np.ndarray,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_color: Tuple[int, int, int] = PAD_COLOR,
) -> np.ndarray:
    """
    Resize *img* to *target_size* while preserving its aspect ratio by
    centre-placing it on a padded canvas (letterboxing).

    Algorithm
    ---------
    1. Compute a uniform scale factor so the longest side fits the target box.
    2. Resize with ``INTER_AREA`` (downscale) or ``INTER_CUBIC`` (upscale).
    3. Place the scaled image on a ``target_size`` canvas filled with
       *pad_color*, centred both horizontally and vertically.

    Parameters
    ----------
    img : np.ndarray
        3-channel RGB ``uint8`` array.
    target_size : tuple[int, int]
        Desired ``(width, height)`` of the output.  Default: ``(512, 512)``.
    pad_color : tuple[int, int, int]
        RGB fill colour for padding regions.  Default: white ``(255,255,255)``.

    Returns
    -------
    np.ndarray
        Shape ``(target_size[1], target_size[0], 3)``, dtype ``uint8``.
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale  = min(target_w / w, target_h / h)
    new_w  = int(round(w * scale))
    new_h  = int(round(h * scale))

    interp  = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    canvas   = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    pad_top  = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

    logger.debug(
        "pad_and_resize: (%d×%d)→(%d×%d) scale=%.3f pad_top=%d pad_left=%d",
        w, h, new_w, new_h, scale, pad_top, pad_left,
    )
    return canvas

def bilateral_filter(
    img: np.ndarray,
    d: int = BILATERAL_D,
    sigma_color: float = BILATERAL_SIGMA_COLOR,
    sigma_space: float = BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """
    Apply OpenCV's bilateral filter to *img*, reducing noise while keeping
    sketch edges sharp.

    The bilateral filter blends nearby pixels only when their colours are
    similar, making it ideal for cleaning pencil-sketch or scribble inputs
    before diffusion-model inference.

    Parameters
    ----------
    img : np.ndarray
        3-channel RGB ``uint8`` image.
    d : int
        Diameter of each pixel neighbourhood.  Use ``-1`` to auto-compute
        from *sigma_space*.  Default: ``9``.
    sigma_color : float
        Sigma in colour space — larger values blend more colours together.
        Default: ``75``.
    sigma_space : float
        Sigma in coordinate space — larger values allow more spatially
        distant pixels to influence each other.  Default: ``75``.

    Returns
    -------
    np.ndarray
        Filtered image; same shape and dtype as *img*.

    Notes
    -----
    ``cv2.bilateralFilter`` operates in BGR internally; RGB↔BGR conversion
    is handled transparently within this function.
    """
    bgr          = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    logger.debug(
        "bilateral_filter: d=%d  σ_color=%.1f  σ_space=%.1f",
        d, sigma_color, sigma_space,
    )
    return filtered_rgb

def to_lineart(
    img: np.ndarray,
    block_size: int = LINEART_BLOCK_SIZE,
    c: int = LINEART_C,
    morph_kernel_size: int = LINEART_MORPH_KERNEL,
    invert: bool = True,
) -> np.ndarray:
    """
    Convert a (bilaterally filtered) RGB sketch into a clean binary lineart
    mask using adaptive thresholding and morphological post-processing.

    This stage is designed to produce the crisp black-on-white (or
    white-on-black) stroke map expected by ControlNet scribble/lineart
    conditioning pipelines.

    Algorithm
    ---------
    1. Convert RGB → grayscale.
    2. Apply **Gaussian adaptive thresholding** (local neighbourhood
       *block_size* × *block_size*, constant offset *c*) to handle uneven
       illumination across the canvas.
    3. Apply a morphological **opening** (erosion → dilation) to eliminate
       isolated noise pixels while preserving stroke continuity.
    4. Optionally invert so strokes are *white* on a *black* background —
       the format expected by most ControlNet scribble models.

    Parameters
    ----------
    img : np.ndarray
        3-channel RGB ``uint8`` image (typically the bilateral-filtered
        output of :func:`bilateral_filter`).
    block_size : int
        Size of the local neighbourhood for adaptive thresholding.  Must be
        an **odd** integer ≥ 3.  Default: ``11``.
    c : int
        Constant subtracted from the weighted mean in adaptive thresholding.
        Positive values raise the threshold (fewer pixels marked as
        foreground).  Default: ``2``.
    morph_kernel_size : int
        Side length (pixels) of the square structuring element used in the
        morphological opening pass.  Set to ``0`` or ``1`` to skip.
        Default: ``2``.
    invert : bool
        If ``True`` (default), invert the binary mask so strokes are white
        on black — the ControlNet scribble convention.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype ``uint8``, values in ``{0, 255}``.
        White pixels represent sketch strokes; black pixels are background
        (when *invert* is ``True``).

    Raises
    ------
    ValueError
        If *block_size* is even or less than 3.
    """
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(
            f"to_lineart: block_size must be an odd integer ≥ 3, got {block_size}."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if invert:
        binary = cv2.bitwise_not(binary)

    logger.debug(
        "to_lineart: block_size=%d  c=%d  morph_k=%d  invert=%s",
        block_size, c, morph_kernel_size, invert,
    )
    return binary

def preprocess(
    source: ImageInput,
    target_size: Tuple[int, int] = TARGET_SIZE,
    pad_color: Tuple[int, int, int] = PAD_COLOR,

    bilateral_d: int = BILATERAL_D,
    bilateral_sigma_color: float = BILATERAL_SIGMA_COLOR,
    bilateral_sigma_space: float = BILATERAL_SIGMA_SPACE,

    return_lineart: bool = False,
    lineart_block_size: int = LINEART_BLOCK_SIZE,
    lineart_c: int = LINEART_C,
    lineart_morph_kernel: int = LINEART_MORPH_KERNEL,
    lineart_invert: bool = True,

    return_tensor: bool = True,
) -> "Union[np.ndarray, torch.Tensor]":
    """
    Full end-to-end preprocessing pipeline.

    Flow
    ----
    ``source`` → :func:`ingest` → :func:`to_rgb` → :func:`pad_and_resize`
    → :func:`bilateral_filter` → [ :func:`to_lineart` ] → tensor / ndarray

    Parameters
    ----------
    source : ImageInput
        Any accepted image input — file path, base64 data-URL, raw bytes,
        ``np.ndarray``, or ``PIL.Image``.
    target_size : tuple[int, int]
        Output ``(width, height)``.  Default: ``(512, 512)``.
    pad_color : tuple[int, int, int]
        Letterbox padding colour.  Default: white.
    bilateral_d : int
        Bilateral filter neighbourhood diameter.
    bilateral_sigma_color : float
        Bilateral colour sigma.
    bilateral_sigma_space : float
        Bilateral spatial sigma.
    return_lineart : bool
        If ``True``, run :func:`to_lineart` and return the binary stroke mask
        instead of the colour image.  Default: ``False``.
    lineart_block_size : int
        Adaptive threshold block size (must be odd).
    lineart_c : int
        Adaptive threshold constant offset.
    lineart_morph_kernel : int
        Morphological opening kernel size.
    lineart_invert : bool
        Invert lineart mask (white strokes on black).
    return_tensor : bool
        When ``True`` (default), return a ``torch.Tensor`` of shape
        ``(C, H, W)`` in float32 with values in ``[0, 1]``.
        ``C = 3`` for colour output; ``C = 1`` for lineart.
        When ``False``, return a ``np.ndarray``.

    Returns
    -------
    torch.Tensor or np.ndarray
        Preprocessed image or lineart mask, ready for model ingestion.
    """
    logger.info("── preprocess(): ingesting source type=%s ──", type(source).__name__)

    img: np.ndarray = ingest(source)
    img = to_rgb(img)
    img = pad_and_resize(img, target_size=target_size, pad_color=pad_color)
    img = bilateral_filter(
        img,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    if return_lineart:
        mask: np.ndarray = to_lineart(
            img,
            block_size=lineart_block_size,
            c=lineart_c,
            morph_kernel_size=lineart_morph_kernel,
            invert=lineart_invert,
        )
        logger.info("preprocess(): lineart mask shape=%s", mask.shape)
        if return_tensor:
            return _to_tensor(mask[:, :, np.newaxis])
        return mask

    logger.info("preprocess(): colour output shape=%s", img.shape)
    if return_tensor:
        return _to_tensor(img)
    return img

def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
    """
    Convert a ``uint8`` NumPy array ``(H, W, C)`` to a float32 torch Tensor
    ``(C, H, W)`` with values normalised to ``[0, 1]``.

    Raises ``ImportError`` (with a helpful message) if PyTorch is not
    installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor output. "
            "Install it with: pip install torch  (see requirements.txt for "
            "the CUDA-enabled wheel command)."
        ) from exc

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    logger.debug("_to_tensor: %s → Tensor%s", arr.shape, tuple(tensor.shape))
    return tensor

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python engine/preprocessing.py <path_to_image> [--lineart]")
        sys.exit(1)

    src_path    = sys.argv[1]
    do_lineart  = "--lineart" in sys.argv

    result = preprocess(src_path, return_lineart=do_lineart, return_tensor=False)

    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    stem     = Path(src_path).stem
    suffix   = "_lineart.png" if do_lineart else "_preprocessed.png"
    out_path = out_dir / (stem + suffix)

    if do_lineart:
        cv2.imwrite(str(out_path), result)
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"✔  Saved → {out_path}")
