"""
engine/pipeline.py
Unified end-to-end inference pipeline orchestrator.
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.generator import DiffusionConfig, SketchDiffusionGenerator
from engine.reconstruction import MeshReconstructor, ReconstructionConfig
from utils.mesh_utils import export_mesh, repair_mesh
from utils.segmentation import isolate_and_center

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.pipeline")


@dataclass
class GenerationResult:
    depth_map: np.ndarray
    normal_map: Optional[np.ndarray]
    mesh: Any
    seed_used: int
    vram_peak_mb: float
    generated_image: Optional[Image.Image] = None
    latency_seconds: float = 0.0
    glb_path: Optional[str] = None


class Sketch2ShapePipeline:
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
        try:
            import torch
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        except Exception:
            pass
        return 0.0

    def _reset_vram_peak(self) -> None:
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
        pipeline_t0 = time.perf_counter()
        self._reset_vram_peak()

        logger.info("=" * 70)
        logger.info("Starting Sketch2Shape AI End-to-End Inference Pipeline")
        logger.info(f"Target Device: {self.device} | FP16 Execution: {self.device.startswith('cuda')}")
        logger.info("=" * 70)

        stage1_t0 = time.perf_counter()
        preprocessed_img = self._preprocess_input(sketch_source)
        logger.info(f"[Stage 1] Preprocessing completed in {time.perf_counter() - stage1_t0:.3f}s")

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

        try:
            segmented_gen_image = isolate_and_center(gen_image, neutral_background=True)
        except Exception as exc:
            logger.warning(f"Foreground segmentation fallback: {exc}")
            segmented_gen_image = gen_image

        stage3_t0 = time.perf_counter()
        logger.info("[Stage 3] Initializing 3D Reconstruction Backbone in FP16...")

        recon_config = ReconstructionConfig(
            device=self.device,
        )
        reconstructor = MeshReconstructor(config=recon_config)

        raw_mesh, depth_map, normal_map, recon_elapsed = reconstructor.reconstruct(
            image_input=segmented_gen_image,
        )
        logger.info(
            f"[Stage 3] 3D Reconstruction completed in {recon_elapsed:.2f}s | "
            f"Vertices: {len(raw_mesh.vertices)} | Faces: {len(raw_mesh.faces)}"
        )

        reconstructor.unload()
        del reconstructor
        gc.collect()

        logger.info("[Stage 4] Intercepting mesh with non-manifold repair sequence...")
        try:
            watertight_mesh = repair_mesh(
                raw_mesh,
                strip_floating=True,
                fill_holes=True,
                center_and_scale=True,
            )
        except Exception as exc:
            logger.warning(f"Mesh repair interceptor warning ({exc}); using reconstructed mesh")
            watertight_mesh = raw_mesh

        total_elapsed = round(time.perf_counter() - pipeline_t0, 3)
        peak_vram = self._get_peak_vram_mb()

        glb_path_exported: Optional[str] = None
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        default_glb = outputs_dir / f"generated_mesh_{concrete_seed}.glb"
        try:
            export_mesh(watertight_mesh, default_glb, export_format="glb")
            glb_path_exported = str(default_glb.resolve())
            logger.info(f"[Stage 5] Serialized clean binary GLB to {default_glb}")
        except Exception as exc:
            logger.warning(f"Default GLB export warning: {exc}")

        if export_path is not None:
            out_file = Path(export_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                export_mesh(watertight_mesh, out_file, export_format=export_format)
            except Exception:
                watertight_mesh.export(str(out_file), file_type=export_format)
            logger.info(f"[Stage 5] Exported 3D asset to {out_file} (format={export_format.upper()})")

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
            glb_path=glb_path_exported,
        )

    def _preprocess_input(self, source: Any) -> Image.Image:
        try:
            from engine.preprocessing import preprocess
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
