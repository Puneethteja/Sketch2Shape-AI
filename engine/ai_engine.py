"""
engine/ai_engine.py
===================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Public API interface for Contract B connecting AI Engine with the Gradio UI layer (ui/app.py).
Conforms strictly to Technical Architecture Specification §4.2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.pipeline import GenerationResult, Sketch2ShapePipeline

logger = logging.getLogger("sketch2shape.ai_engine")

__all__ = ["GenerationResult", "generate"]

# Module-level singleton pipeline instance
_GLOBAL_PIPELINE: Optional[Sketch2ShapePipeline] = None


def get_pipeline(device: Optional[str] = None) -> Sketch2ShapePipeline:
    """Returns or initializes the global pipeline instance."""
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
    """
    Contract B Producer Function (§4.2).
    Generates a 3D asset from an input sketch and returns a GenerationResult.

    Parameters:
        source: Preprocessed lineart tensor, PIL.Image, or NumPy array.
        prompt: Optional text description for materials and styling.
        controlnet_scale: ControlNet conditioning weight (calibrated 0.7 - 0.95).
        num_inference_steps: Diffusion denoising steps (default: 30).
        seed: RNG seed for reproducible generation (-1 = random).
        device: Target execution device ('cuda' or 'cpu').

    Returns:
        GenerationResult dataclass:
            depth_map: shape (512, 512), float32, range [0, 1]
            normal_map: shape (512, 512, 3), float32, range [0, 1]
            mesh: watertight trimesh.Trimesh
            seed_used: concrete integer
            vram_peak_mb: peak VRAM in MB during inference
    """
    pipeline = get_pipeline(device=device)
    return pipeline.run(
        sketch_source=source,
        prompt=prompt,
        controlnet_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
