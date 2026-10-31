"""
utils/preprocessing.py
======================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

Core preprocessing utilities for ingesting raw sketch files or canvas payloads,
color-space normalisation, letterbox resizing, bilateral noise filtering,
and binary lineart/scribble generation.

Exposes the complete preprocessing engine API under utils.preprocessing.
"""

from engine.preprocessing import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    LINEART_BLOCK_SIZE,
    LINEART_C,
    LINEART_MORPH_KERNEL,
    PAD_COLOR,
    TARGET_SIZE,
    ImageInput,
    bilateral_filter,
    ingest,
    load_image,
    pad_and_resize,
    preprocess,
    to_lineart,
    to_rgb,
)

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
    "BILATERAL_D",
    "BILATERAL_SIGMA_COLOR",
    "BILATERAL_SIGMA_SPACE",
    "LINEART_BLOCK_SIZE",
    "LINEART_C",
    "LINEART_MORPH_KERNEL",
    "ImageInput",
]
