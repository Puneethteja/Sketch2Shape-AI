"""
utils/segmentation.py
=====================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

Foreground Isolation & Background Removal Utilities.
Integrates rembg (U²-Net / IS-Net models) to extract foreground subjects,
generate high-resolution transparent alpha masks, and crop/center the isolated
subject on a neutral background to prevent downstream 3D reconstruction distortion.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from utils.preprocessing import ImageInput, ingest, to_rgb

logger = logging.getLogger("sketch2shape.segmentation")

_SESSION_CACHE: Dict[str, Any] = {}

SUPPORTED_SEGMENTATION_MODELS: Tuple[str, ...] = (
    "u2net",
    "u2netp",
    "isnet-general-use",
    "isnet-anime",
    "silueta",
)

DEFAULT_MODEL: str = "u2net"
DEFAULT_TARGET_SIZE: Tuple[int, int] = (512, 512)
DEFAULT_PADDING_RATIO: float = 0.1

def get_rembg_session(model_name: str = DEFAULT_MODEL) -> Any:
    """
    Retrieve or initialize a cached rembg ONNX inference session.

    Parameters
    ----------
    model_name : str
        Name of the segmentation model backbone (default: 'u2net').

    Returns
    -------
    rembg.sessions.BaseSession or None
        The initialized session instance, or None if rembg cannot be loaded.
    """
    if model_name in _SESSION_CACHE:
        return _SESSION_CACHE[model_name]

    try:
        from rembg import new_session
        logger.info("Initializing rembg session for model '%s'", model_name)
        session = new_session(model_name)
        _SESSION_CACHE[model_name] = session
        return session
    except ImportError as exc:
        logger.warning("rembg is not installed: %s. Using heuristic fallback.", exc)
        return None
    except Exception as exc:
        logger.warning("Could not initialize rembg session for '%s': %s", model_name, exc)
        return None

def remove_background(
    image: ImageInput,
    model_name: str = DEFAULT_MODEL,
    session: Optional[Any] = None,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    only_mask: bool = False,
) -> Image.Image:
    """
    Extract the foreground subject from the background using rembg.

    Parameters
    ----------
    image : ImageInput
        Source image (file path, raw bytes, base64 data-URL, PIL.Image, or ndarray).
    model_name : str
        Model architecture to employ (e.g. 'u2net', 'isnet-general-use').
    session : Any, optional
        Pre-existing rembg session. If None, uses cached session from get_rembg_session.
    alpha_matting : bool
        Whether to refine soft boundary edges (feathering, hair/fur).
    alpha_matting_foreground_threshold : int
        Threshold for confident foreground in alpha matting.
    alpha_matting_background_threshold : int
        Threshold for confident background in alpha matting.
    alpha_matting_erode_size : int
        Erosion kernel size for alpha matting trimap generation.
    only_mask : bool
        If True, returns only the single-channel alpha mask.

    Returns
    -------
    PIL.Image.Image
        RGBA image with transparent background, or single-channel 'L' mask if only_mask=True.
    """

    raw_array = ingest(image)
    if raw_array.ndim == 2:
        pil_input = Image.fromarray(raw_array, mode="L").convert("RGB")
    elif raw_array.shape[2] == 4:
        pil_input = Image.fromarray(raw_array, mode="RGBA")
    else:
        pil_input = Image.fromarray(to_rgb(raw_array), mode="RGB")

    active_session = session or get_rembg_session(model_name)

    if active_session is not None:
        try:
            from rembg import remove
            output_img = remove(
                pil_input,
                session=active_session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                only_mask=only_mask,
            )
            return output_img
        except Exception as exc:
            logger.warning("rembg inference failed (%s); falling back to heuristic segmentation", exc)

    return _heuristic_segmentation_fallback(pil_input, only_mask=only_mask)

def _heuristic_segmentation_fallback(
    pil_img: Image.Image,
    only_mask: bool = False,
) -> Image.Image:
    """
    Fast offline heuristic segmentation when neural models are unavailable.
    Segments subject assuming light/white paper canvas or uniform perimeter background.
    """
    rgb_arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    if only_mask:
        return Image.fromarray(mask, mode="L")

    rgba = np.dstack([rgb_arr, mask])
    return Image.fromarray(rgba, mode="RGBA")

def extract_alpha_mask(
    image: ImageInput,
    model_name: str = DEFAULT_MODEL,
    session: Optional[Any] = None,
    threshold: int = 10,
) -> np.ndarray:
    """
    Extract a high-resolution 2D binary/alpha mask array from an image.

    Parameters
    ----------
    image : ImageInput
        Source image payload.
    model_name : str
        Segmentation model backbone.
    session : Any, optional
        Rembg session instance.
    threshold : int
        Minimum opacity value [0-255] considered foreground.

    Returns
    -------
    np.ndarray
        2D uint8 NumPy array of shape (H, W) where 255 = foreground, 0 = background.
    """
    mask_img = remove_background(
        image,
        model_name=model_name,
        session=session,
        only_mask=True,
    )
    mask_arr = np.array(mask_img.convert("L"))

    if threshold > 0:
        mask_arr = np.where(mask_arr >= threshold, np.uint8(255), np.uint8(0))

    return mask_arr

def get_foreground_bbox(
    alpha_mask: np.ndarray,
    min_opacity: int = 10,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate the tight bounding box of non-zero foreground pixels.

    Parameters
    ----------
    alpha_mask : np.ndarray
        Single-channel 2D alpha mask (H, W).
    min_opacity : int
        Minimum alpha value considered active foreground.

    Returns
    -------
    tuple[int, int, int, int] or None
        (x_min, y_min, x_max, y_max), or None if no foreground pixels exist.
    """
    active_y, active_x = np.where(alpha_mask >= min_opacity)
    if len(active_x) == 0 or len(active_y) == 0:
        return None

    x_min, x_max = int(np.min(active_x)), int(np.max(active_x))
    y_min, y_max = int(np.min(active_y)), int(np.max(active_y))
    return x_min, y_min, x_max, y_max

def crop_and_center_subject(
    rgba_image: Union[Image.Image, np.ndarray],
    alpha_mask: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
    return_bbox: bool = False,
) -> Union[Image.Image, Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]]:
    """
    Crop the isolated subject to its tight bounding box and center it on a neutral
    canvas preserving aspect ratio with controlled padding.

    Crucial for 3D reconstruction backbones (InstantMesh / Unique3D) which require
    the subject to be uniformly centered within the 512x512 canonical coordinate frame.

    Parameters
    ----------
    rgba_image : PIL.Image or np.ndarray
        The transparent foreground-isolated image (RGBA).
    alpha_mask : np.ndarray, optional
        Pre-extracted alpha mask. If None, derived from the alpha channel of rgba_image.
    target_size : tuple[int, int]
        Desired output resolution (width, height), default: (512, 512).
    padding_ratio : float
        Proportional margin around the subject (default: 0.1, or 10% border).
    background_color : tuple[int, int, int, int]
        RGBA background fill color for the canvas. Default: transparent (255, 255, 255, 0).
    return_bbox : bool
        If True, returns a tuple of (centered_image, bbox).

    Returns
    -------
    PIL.Image.Image or tuple[PIL.Image.Image, Optional[tuple]]
        Cleanly centered RGBA image.
    """
    if isinstance(rgba_image, np.ndarray):
        pil_rgba = Image.fromarray(rgba_image, mode="RGBA")
    else:
        pil_rgba = rgba_image.convert("RGBA")

    w_in, h_in = pil_rgba.size

    if alpha_mask is None:
        alpha_arr = np.array(pil_rgba.split()[-1])
    else:
        alpha_arr = alpha_mask

    bbox = get_foreground_bbox(alpha_arr)
    target_w, target_h = target_size

    if bbox is None:
        logger.warning("crop_and_center_subject: No foreground detected. Returning blank canvas.")
        canvas = Image.new("RGBA", target_size, background_color)
        return (canvas, None) if return_bbox else canvas

    x_min, y_min, x_max, y_max = bbox
    crop_w = max(1, x_max - x_min + 1)
    crop_h = max(1, y_max - y_min + 1)

    cropped = pil_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))

    avail_w = max(1, int(target_w * (1.0 - 2.0 * padding_ratio)))
    avail_h = max(1, int(target_h * (1.0 - 2.0 * padding_ratio)))

    scale = min(avail_w / crop_w, avail_h / crop_h)
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))

    resample_method = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    scaled_subject = cropped.resize((new_w, new_h), resample=resample_method)

    canvas = Image.new("RGBA", target_size, background_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    canvas.paste(scaled_subject, (paste_x, paste_y), mask=scaled_subject)

    logger.debug(
        "crop_and_center: bbox=(%d,%d,%d,%d) crop=(%dx%d) scaled=(%dx%d) paste=(%d,%d)",
        x_min, y_min, x_max, y_max, crop_w, crop_h, new_w, new_h, paste_x, paste_y,
    )

    if return_bbox:
        return canvas, bbox
    return canvas

def isolate_and_center(
    source: ImageInput,
    model_name: str = DEFAULT_MODEL,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    neutral_background: bool = True,
    background_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    End-to-end subject segmentation and centering pipeline.

    Flow:
    1. Ingest input source.
    2. Remove background with rembg / U²-Net.
    3. Calculate tight foreground bounding box.
    4. Rescale and center onto a canonical 512x512 canvas.
    5. Optionally composite over a neutral solid background (e.g. clean white)
       ready for direct input into 3D reconstruction models.

    Parameters
    ----------
    source : ImageInput
        Input image (file path, bytes, base64, PIL, ndarray).
    model_name : str
        Rembg model backbone.
    target_size : tuple[int, int]
        Output resolution (width, height).
    padding_ratio : float
        Border padding ratio (default: 0.1).
    neutral_background : bool
        If True, composites transparent areas over solid background_rgb.
        If False, returns transparent RGBA.
    background_rgb : tuple[int, int, int]
        Background color for compositing (default: white 255, 255, 255).

    Returns
    -------
    PIL.Image.Image
        Standardized 512x512 preprocessed image optimized for 3D reconstruction.
    """
    logger.info("Executing isolate_and_center segmentation pipeline")

    segmented_rgba = remove_background(source, model_name=model_name)

    centered_rgba = crop_and_center_subject(
        segmented_rgba,
        target_size=target_size,
        padding_ratio=padding_ratio,
        background_color=(0, 0, 0, 0),
    )

    if not neutral_background:
        return centered_rgba

    solid_bg = Image.new("RGBA", target_size, (*background_rgb, 255))
    solid_bg.paste(centered_rgba, (0, 0), mask=centered_rgba)
    return solid_bg.convert("RGB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sketch2Shape Foreground Isolation Utility")
    parser.add_argument("input", help="Path to input image file")
    parser.add_argument("-o", "--output", default="outputs/isolated_subject.png", help="Output path")
    parser.add_argument("-m", "--model", default="u2net", help="rembg model name (u2net, isnet-general-use)")
    parser.add_argument("--transparent", action="store_true", help="Output transparent RGBA instead of solid background")
    parser.add_argument("--pad", type=float, default=0.1, help="Padding ratio (default: 0.1)")

    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    result = isolate_and_center(
        args.input,
        model_name=args.model,
        padding_ratio=args.pad,
        neutral_background=not args.transparent,
    )
    result.save(out_p)
    print(f"Saved foreground-isolated image to: {out_p.resolve()}")
