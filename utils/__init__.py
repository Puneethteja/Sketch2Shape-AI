"""
Sketch2Shape AI -- utils package
TechZephyr 4.0 Hackathon · IIT Bhubaneswar
"""

from utils.image_utils import (
    draw_overlay_mask,
    load_pil,
    ndarray_to_pil,
    pil_to_ndarray,
    save_image,
    stack_side_by_side,
)
from utils.mesh_utils import (
    export_mesh,
    inspect_mesh,
    is_watertight,
    load_mesh,
    repair_mesh,
    strip_isolated_components,
)
from utils.preprocessing import (
    bilateral_filter,
    ingest,
    load_image,
    pad_and_resize,
    preprocess,
    to_lineart,
    to_rgb,
)
from utils.segmentation import (
    crop_and_center_subject,
    extract_alpha_mask,
    get_foreground_bbox,
    isolate_and_center,
    remove_background,
)

__all__ = [
    # Image I/O & Helpers
    "save_image",
    "load_pil",
    "ndarray_to_pil",
    "pil_to_ndarray",
    "stack_side_by_side",
    "draw_overlay_mask",
    # Preprocessing
    "ingest",
    "load_image",
    "to_rgb",
    "pad_and_resize",
    "bilateral_filter",
    "to_lineart",
    "preprocess",
    # Segmentation & Centering
    "remove_background",
    "extract_alpha_mask",
    "get_foreground_bbox",
    "crop_and_center_subject",
    "isolate_and_center",
    # Mesh Validation, Repair & Export
    "load_mesh",
    "inspect_mesh",
    "is_watertight",
    "repair_mesh",
    "strip_isolated_components",
    "export_mesh",
]
