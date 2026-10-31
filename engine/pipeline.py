"""
engine/pipeline.py
==================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Unified end-to-end inference pipeline orchestrator with strict FP16 execution
and sequential VRAM memory handover to eliminate Out-Of-Memory (OOM) risks
on 16GB VRAM GPUs (NVIDIA T4 / V100) and consumer cards.

Sequential Execution Flow:
1. Ingest sketch input and normalize to 512x512 lineart tensor (Puneeth, Contract A).
2. Initialize SketchDiffusionGenerator in FP16 -> Generate 2.5D asset (~5.2 GB peak).
3. Sequential Handover: Evacuate generator weights & clear CUDA cache (0 GB active).
4. Initialize MeshReconstructor in FP16 -> Predict depth, normals, & 3D mesh (~6.5 GB peak).
5. Deterministic watertight repair sequence (§3.4).
6. Return GenerationResult (Contract B) and optionally export to OBJ/GLB/STL/PLY.
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Reconfigure stdout to utf-8 if supported to prevent Windows charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image

# Ensure project root is in sys.path for direct CLI execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.generator import DiffusionConfig, SketchDiffusionGenerator
from engine.mesh_utils import repair_mesh
from engine.reconstruction import MeshReconstructor, ReconstructionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.pipeline")


# ── Contract B Dataclass Definition (§4.2) ───────────────────────────────────

@dataclass
class GenerationResult:
    """
    Return type for engine.pipeline and engine.ai_engine.generate().
    All fields are on CPU and ready for UI consumption. Conforms to §4.2.
    """
    depth_map: np.ndarray               # shape (512, 512), float32, range [0, 1]
    normal_map: Optional[np.ndarray]    # shape (512, 512, 3), float32, range [0, 1]
    mesh: Any                           # watertight trimesh.Trimesh after §3.4 repair
    seed_used: int                      # actual concrete seed (never -1)
    vram_peak_mb: float                 # peak VRAM during inference in MB
    generated_image: Optional[Image.Image] = None  # 2.5D diffusion output
    latency_seconds: float = 0.0        # total pipeline latency in seconds


# ── Unified Pipeline Orchestrator ────────────────────────────────────────────

class Sketch2ShapePipeline:
    """
    End-to-end pipeline executing sketch-guided generation followed by
    3D mesh reconstruction with strict sequential VRAM management.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        force_cpu: bool = False,
    ) -> None:
        self.device = "cpu" if force_cpu else self._resolve_device(device)
        logger.info(f"Initialized Sketch2ShapePipeline on device: {self.device}")

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        if requested_device:
            return requested_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _get_peak_vram_mb(self) -> float:
        """Reads peak CUDA memory allocated in MB."""
        try:
            import torch
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        except Exception:
            pass
        return 0.0

    def _reset_vram_peak(self) -> None:
        """Resets CUDA peak memory tracking."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def run(
        self,
        sketch_source: Union[str, Path, Image.Image, np.ndarray, bytes],
        prompt: Optional[str] = None,
        controlnet_scale: float = 0.85,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = -1,
        export_path: Optional[Union[str, Path]] = None,
        export_format: str = "obj",
    ) -> GenerationResult:
        """
        Executes end-to-end generation from 2D sketch to textured watertight 3D mesh.

        Parameters:
            sketch_source: File path, PIL.Image, NumPy array, or bytes.
            prompt: Optional text description for materials, styling, colors.
            controlnet_scale: ControlNet conditioning weight (calibrated 0.7 - 0.95).
            num_inference_steps: Diffusion denoising steps (default: 30).
            guidance_scale: Classifier-free guidance (default: 7.5).
            seed: RNG seed for reproducible generation (-1 for random).
            export_path: Optional output path to auto-export mesh file.
            export_format: Mesh format ('obj', 'glb', 'stl', 'ply').
        """
        pipeline_t0 = time.perf_counter()
        self._reset_vram_peak()

        logger.info("=" * 70)
        logger.info("Starting Sketch2Shape AI End-to-End Inference Pipeline")
        logger.info(f"Target Device: {self.device} | FP16 Execution: {self.device.startswith('cuda')}")
        logger.info("=" * 70)

        # ── Stage 1: Universal Preprocessing ─────────────────────────────────
        stage1_t0 = time.perf_counter()
        preprocessed_img = self._preprocess_input(sketch_source)
        logger.info(f"[Stage 1] Preprocessing completed in {time.perf_counter() - stage1_t0:.3f}s")

        # ── Stage 2: Sketch-Guided Diffusion Generation (FP16) ───────────────
        stage2_t0 = time.perf_counter()
        logger.info("[Stage 2] Initializing Diffusion Generator in FP16...")

        diff_config = DiffusionConfig(
            device=self.device,
            controlnet_scale=controlnet_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        generator = SketchDiffusionGenerator(config=diff_config)

        gen_image, concrete_seed, gen_elapsed = generator.generate(
            sketch_input=preprocessed_img,
            prompt=prompt,
            controlnet_scale=controlnet_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        logger.info(f"[Stage 2] Diffusion pass completed in {gen_elapsed:.2f}s | Seed: {concrete_seed}")

        # ── Sequential Handover: Evacuate Generator from VRAM ────────────────
        logger.info("[Sequential Handover] Evacuating Stage 2 model weights from VRAM...")
        generator.unload()
        del generator
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                vram_curr = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"[VRAM Reset] CUDA cache cleared. Active VRAM: {vram_curr:.1f} MB")
        except Exception:
            pass

        # ── Stage 3: Pretrained 3D Mesh Reconstruction (FP16) ────────────────
        stage3_t0 = time.perf_counter()
        logger.info("[Stage 3] Initializing 3D Reconstruction Backbone in FP16...")

        recon_config = ReconstructionConfig(
            device=self.device,
        )
        reconstructor = MeshReconstructor(config=recon_config)

        watertight_mesh, depth_map, normal_map, recon_elapsed = reconstructor.reconstruct(
            image_input=gen_image,
        )
        logger.info(
            f"[Stage 3] 3D Reconstruction completed in {recon_elapsed:.2f}s | "
            f"Vertices: {len(watertight_mesh.vertices)} | Faces: {len(watertight_mesh.faces)} | "
            f"Watertight: {watertight_mesh.is_watertight}"
        )

        # Evacuate Reconstructor from VRAM
        reconstructor.unload()
        del reconstructor
        gc.collect()

        total_elapsed = round(time.perf_counter() - pipeline_t0, 3)
        peak_vram = self._get_peak_vram_mb()

        # ── Stage 4: Optional Serialization & Export ─────────────────────────
        if export_path is not None:
            out_file = Path(export_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            watertight_mesh.export(str(out_file), file_type=export_format)
            logger.info(f"[Stage 4] Exported 3D asset to {out_file} (format={export_format.upper()})")

        logger.info("=" * 70)
        logger.info(
            f"Pipeline Success | Total Time: {total_elapsed:.2f}s | "
            f"VRAM Peak: {peak_vram:.1f} MB | Watertight: {watertight_mesh.is_watertight}"
        )
        logger.info("=" * 70)

        return GenerationResult(
            depth_map=depth_map,
            normal_map=normal_map,
            mesh=watertight_mesh,
            seed_used=concrete_seed,
            vram_peak_mb=peak_vram,
            generated_image=gen_image,
            latency_seconds=total_elapsed,
        )

    def _preprocess_input(self, source: Any) -> Image.Image:
        """
        Integrates Puneeth's engine.preprocessing pipeline if available,
        falling back to PIL normalisation if unimported.
        """
        try:
            from engine.preprocessing import preprocess
            # Ingest and letterbox resize to 512x512
            preprocessed_np = preprocess(source, return_tensor=False)
            return Image.fromarray(preprocessed_np)
        except Exception as exc:
            logger.debug(f"Preprocessing through engine.preprocessing skipped ({exc}), using standard PIL.")

        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
        elif isinstance(source, Image.Image):
            img = source.convert("RGB")
        elif isinstance(source, np.ndarray):
            if source.ndim == 2:
                img = Image.fromarray(source).convert("RGB")
            elif source.shape[-1] == 4:
                rgba = Image.fromarray(source)
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                img = bg
            else:
                img = Image.fromarray(source.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Unsupported source format: {type(source)}")

        if img.size != (512, 512):
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
        return img


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sketch2Shape AI - End-to-End Generative 3D Reconstruction Pipeline"
    )
    parser.add_argument("input", type=str, help="Path to input sketch image.")
    parser.add_argument("--prompt", type=str, default="", help="Optional text description for styling and materials.")
    parser.add_argument("--scale", type=float, default=0.85, help="ControlNet conditioning scale (0.0 - 2.0). Default: 0.85.")
    parser.add_argument("--steps", type=int, default=30, help="Number of diffusion inference steps. Default: 30.")
    parser.add_argument("--seed", type=int, default=-1, help="RNG seed (-1 for random).")
    parser.add_argument("--output", type=str, default="outputs/generated_mesh.obj", help="Path to save reconstructed mesh.")
    parser.add_argument("--format", type=str, default="obj", choices=["obj", "glb", "stl", "ply"], help="Mesh export format.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution mode.")
    args = parser.parse_args()

    pipeline = Sketch2ShapePipeline(force_cpu=args.cpu)
    result = pipeline.run(
        sketch_source=args.input,
        prompt=args.prompt,
        controlnet_scale=args.scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        export_path=args.output,
        export_format=args.format,
    )

    print("\n" + "=" * 60)
    print(" [Sketch2Shape AI] Generation Finished")
    print(f"  * Seed Used       : {result.seed_used}")
    print(f"  * Total Latency   : {result.latency_seconds:.2f}s")
    print(f"  * Peak VRAM       : {result.vram_peak_mb:.1f} MB")
    print(f"  * Output Mesh     : {args.output}")
    print(f"  * Watertight Hull : {result.mesh.is_watertight}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
