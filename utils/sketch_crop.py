from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def prepare_sketch(image: Image.Image, output_size: int = 512) -> Tuple[Image.Image, Dict[str, Any]]:
    """Remove a near-uniform border and center the primary drawing when clear.

    Returns a 512-square white-background preview and crop metadata. The function
    declines automatic cropping when the foreground fills the image or detection
    is ambiguous; callers can then request an explicit user crop.
    """
    import cv2

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    band = max(2, min(height, width) // 32)
    border = np.concatenate((rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
                             rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    border_spread = float(np.median(np.linalg.norm(border.astype(np.float32) - background, axis=1)))
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    mask = (distance > max(20.0, border_spread * 3.0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, width // 64 * 2 + 1), max(3, height // 64 * 2 + 1)))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    candidates = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= max(12, int(width * height * 0.0003))]
    candidates.sort(reverse=True)
    uncertain = False
    reason = None
    bbox = (0, 0, width, height)
    if not candidates:
        uncertain, reason = True, "No clear foreground drawing was found."
    else:
        primary = candidates[0][1]
        ys, xs = np.where(labels == primary)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        # Include nearby disconnected strokes (chair legs, bag handles, leaves)
        # instead of cropping to the single largest connected contour.
        margin_x = max(24, (x1 - x0) // 4)
        margin_y = max(24, (y1 - y0) // 4)
        expanded = (max(0, x0 - margin_x), max(0, y0 - margin_y),
                    min(width, x1 + margin_x), min(height, y1 + margin_y))
        for area, component_id in candidates[1:]:
            cy, cx = np.where(labels == component_id)
            bx0, bx1 = int(cx.min()), int(cx.max()) + 1
            by0, by1 = int(cy.min()), int(cy.max()) + 1
            nearby = bx1 >= expanded[0] and bx0 <= expanded[2] and by1 >= expanded[1] and by0 <= expanded[3]
            if nearby:
                x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
            elif area >= candidates[0][0] * 0.12:
                uncertain, reason = True, "A substantial separate region may be part of the object; include it in the manual crop."
        box_area = (x1 - x0) * (y1 - y0)
        occupancy = box_area / float(width * height)
        second_ratio = candidates[1][0] / candidates[0][0] if len(candidates) > 1 else 0.0
        if uncertain:
            pass
        elif occupancy > 0.70:
            uncertain, reason = True, "The detected object or page content spans most of the image; review the crop manually to avoid cutting it."
        elif second_ratio > 0.72:
            uncertain, reason = True, "Several similarly prominent regions were found; choose the object crop manually."
        else:
            pad = max(8, int(max(x1 - x0, y1 - y0) * 0.08))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
            bbox = (x0, y0, x1, y1)
    # Normalize near-background pixels to white without erasing internal lines.
    cleaned = rgb.copy()
    bg = np.all(np.abs(rgb.astype(np.int16) - background.astype(np.int16)) <= 18, axis=2)
    cleaned[bg] = 255
    if not uncertain:
        x0, y0, x1, y1 = bbox
        cleaned = cleaned[y0:y1, x0:x1]
    crop = Image.fromarray(cleaned, "RGB")
    crop.thumbnail((output_size - 32, output_size - 32), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (output_size, output_size), "white")
    canvas.paste(crop, ((output_size - crop.width) // 2, (output_size - crop.height) // 2))
    return canvas, {"uncertain": uncertain, "reason": reason, "bbox": list(bbox), "source_size": [width, height]}
