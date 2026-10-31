"""
Sketch2Shape AI — Core Engine Package
====================================
TechZephyr 4.0 Hackathon · IIT Bhubaneswar
"""

from engine.ai_engine import GenerationResult, generate
from engine.generator import DiffusionConfig, SketchDiffusionGenerator
from engine.linker import StageHandoff, StageLinker
from engine.mesh_utils import TrimeshFallback, repair_mesh
from engine.pipeline import Sketch2ShapePipeline, StageTiming
from engine.reconstruction import MeshReconstructor, ReconstructionConfig

try:
    from engine.preprocessing import preprocess
except ImportError:
    preprocess = None

__all__ = [
    "preprocess",
    "SketchDiffusionGenerator",
    "DiffusionConfig",
    "MeshReconstructor",
    "ReconstructionConfig",
    "TrimeshFallback",
    "repair_mesh",
    "StageLinker",
    "StageHandoff",
    "Sketch2ShapePipeline",
    "StageTiming",
    "GenerationResult",
    "generate",
]
