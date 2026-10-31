"""
engine/ai_engine.py
Public API interface for Contract B connecting AI Engine with the Gradio UI layer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.pipeline import GenerationResult, Sketch2ShapePipeline

logger = logging.getLogger("sketch2shape.ai_engine")

__all__ = ["GenerationResult", "generate"]

_GLOBAL_PIPELINE: Optional[Sketch2ShapePipeline] = None


def get_pipeline(device: Optional[str] = None) -> Sketch2ShapePipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = Sketch2ShapePipeline(device=device)
    return _GLOBAL_PIPELINE


def generate(
    source: Any,
    prompt: str = "",
    controlnet_scale: float = 0.85,
    num_inference_steps: int = 30,
    seed: int = -1,
    device: Optional[str] = None,
) -> GenerationResult:
    pipeline = get_pipeline(device=device)
    return pipeline.run(
        sketch_source=source,
        prompt=prompt,
        controlnet_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
