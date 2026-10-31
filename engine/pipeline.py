"""
engine/pipeline.py
Unified end-to-end inference pipeline orchestrator with strict FP16 execution,
StageLinker tensor handoff, automated VRAM clearing, and sequential CPU offloading.
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
from engine.linker import StageHandoff, StageLinker
from engine.reconstruction import MeshReconstructor, ReconstructionConfig

try:
    from utils.mesh_utils import export_mesh, repair_mesh
except Exception:
    from engine.mesh_utils import repair_mesh as _engine_repair_mesh
    def repair_mesh(mesh: Any, **kwargs: Any) -> Any:
        return _engine_repair_mesh(mesh, verbose=kwargs.get("verbose", False))
    def export_mesh(mesh: Any, path: Union[str, Path], export_format: str = "obj") -> str:
        mesh.export(str(path), file_type=export_format)
        return str(path)

try:
    from utils.segmentation import isolate_and_center
except Exception:
    isolate_and_center = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.pipeline")


@dataclass
class StageTiming:
    preprocess: float = 0.0
    generation: float = 0.0
    linker_extraction: float = 0.0
    reconstruction: float = 0.0
    export: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.preprocess + self.generation + self.linker_extraction + self.reconstruction + self.export,
            3,
        )


@dataclass
class GenerationResult:
    depth_map: np.ndarray
    normal_map: Optional[np.ndarray]
    mesh: Any
    seed_used: int
    vram_peak_mb: float
    generated_image: Optional[Image.Image] = None
    latency_seconds: float = 0.0
    stage_timing: Optional[StageTiming] = None
    handoff_metadata: Optional[Dict[str, Any]] = None
    glb_path: Optional[str] = None

    @property
    def is_watertight(self) -> bool:
        return bool(getattr(self.mesh, "is_watertight", False))

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices) if self.mesh is not None else 0

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces) if self.mesh is not None else 0

    @property
    def depth_image(self) -> Image.Image:
        depth_uint8 = np.clip(self.depth_map * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(depth_uint8, mode="L").convert("RGB")

    @property
    def elapsed_total(self) -> float:
        return self.latency_seconds

    @property
    def seed(self) -> int:
        return self.seed_used


class Sketch2ShapePipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        force_cpu: bool = False,
    ) -> None:
        self.device = "cpu" if force_cpu else self._resolve_device(device)
        self._linker = StageLinker(device=self.device)
        logger.info(f"Initialized Sketch2ShapePipeline on device: {self.device} (StageLinker active)")

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

    def _evacuate_vram(self, model: Optional[Any] = None) -> None:
        if model is not None:
            if hasattr(model, "pipeline") and model.pipeline is not None:
                try:
                    model.pipeline.to("cpu")
                except Exception:
                    pass
            if hasattr(model, "model") and model.model is not None:
                try:
                    if hasattr(model.model, "to"):
                        model.model.to("cpu")
                except Exception:
                    pass
            if hasattr(model, "unload"):
                try:
                    model.unload()
                except Exception:
                    pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                active_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"[VRAM Reset] Cache cleared and offloaded to CPU. Active VRAM: {active_mb:.1f} MB")
        except Exception:
            pass

    def run(
        self,
        sketch_source: Union[str, Path, Image.Image, np.ndarray, bytes],
        prompt: Optional[str] = None,
        controlnet_scale: float = 0.80,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        seed: int = -1,
        export_path: Optional[Union[str, Path]] = None,
        export_format: str = "obj",
    ) -> GenerationResult:
        pipeline_t0 = time.perf_counter()
        self._reset_vram_peak()
        timing = StageTiming()

        logger.info("=" * 70)
        logger.info("Starting Sketch2Shape AI End-to-End Inference Pipeline")
        logger.info(f"Target Device: {self.device} | FP16 Execution: {self.device.startswith('cuda')}")
        logger.info("=" * 70)

        stage1_t0 = time.perf_counter()
        preprocessed_img = self._preprocess_input(sketch_source)
        timing.preprocess = round(time.perf_counter() - stage1_t0, 3)
        logger.info(f"[Stage 1] Preprocessing completed in {timing.preprocess:.3f}s")

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
        timing.generation = gen_elapsed
        logger.info(f"[Stage 2] Diffusion pass completed in {gen_elapsed:.2f}s | Seed: {concrete_seed}")

        linker_t0 = time.perf_counter()
        logger.info("[Stage 2->3 Linker] Extracting depth priors and shading tensors...")
        handoff: StageHandoff = self._linker.extract_handoff_tensors(
            generated_image=gen_image,
            seed=concrete_seed,
            gen_elapsed=gen_elapsed,
            prompt=prompt or "",
        )
        timing.linker_extraction = round(time.perf_counter() - linker_t0, 4)
        logger.info(
            f"[Stage 2->3 Linker] Tensor extraction completed in {timing.linker_extraction:.4f}s | "
            f"FG pixels: {handoff.metadata.get('fg_pixel_count', 0)} | "
            f"Depth prior max: {handoff.metadata.get('depth_max', 0.0):.3f}"
        )

        logger.info("[Sequential Handover] Evacuating Stage 2 model weights from VRAM...")
        self._evacuate_vram(generator)
        del generator

        if isolate_and_center is not None:
            try:
                segmented_gen_image = isolate_and_center(gen_image, neutral_background=True)
                handoff.generated_image = segmented_gen_image
            except Exception as exc:
                logger.warning(f"Foreground segmentation fallback: {exc}")

        stage3_t0 = time.perf_counter()
        logger.info("[Stage 3] Initializing 3D Reconstruction Backbone in FP16...")

        recon_config = ReconstructionConfig(device=self.device)
        reconstructor = MeshReconstructor(config=recon_config)

        raw_mesh, depth_map, normal_map, recon_elapsed = self._linker.linked_reconstruct(
            reconstructor=reconstructor,
            handoff=handoff,
        )
        timing.reconstruction = recon_elapsed
        logger.info(
            f"[Stage 3] Linked 3D Reconstruction completed in {recon_elapsed:.2f}s | "
            f"Vertices: {len(raw_mesh.vertices)} | "
            f"Faces: {len(raw_mesh.faces)}"
        )

        logger.info("[Sequential Handover] Evacuating Stage 3 model weights from VRAM...")
        self._evacuate_vram(reconstructor)
        del reconstructor

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
            export_t0 = time.perf_counter()
            out_file = Path(export_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                export_mesh(watertight_mesh, out_file, export_format=export_format)
            except Exception:
                watertight_mesh.export(str(out_file), file_type=export_format)
            timing.export = round(time.perf_counter() - export_t0, 3)
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
            stage_timing=timing,
            handoff_metadata=handoff.metadata,
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
    parser.add_argument("--scale", type=float, default=0.80, help="ControlNet conditioning scale (0.0 - 2.0). Default: 0.80.")
    parser.add_argument("--steps", type=int, default=20, help="Number of diffusion inference steps. Default: 20.")
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

    t = result.stage_timing
    print("\n" + "=" * 60)
    print(" [Sketch2Shape AI] Pipeline Finished")
    print(f"  * Seed Used           : {result.seed_used}")
    print(f"  * Total Latency       : {result.latency_seconds:.2f}s")
    if t:
        print(f"    - Preprocess        : {t.preprocess:.3f}s")
        print(f"    - Generation        : {t.generation:.3f}s")
        print(f"    - Linker Extraction : {t.linker_extraction:.4f}s")
        print(f"    - Reconstruction    : {t.reconstruction:.3f}s")
        if t.export > 0:
            print(f"    - Export            : {t.export:.3f}s")
    print(f"  * Peak VRAM           : {result.vram_peak_mb:.1f} MB")
    print(f"  * Output Mesh         : {args.output}")
    print(f"  * Vertices / Faces    : {result.vertex_count} / {result.face_count}")
    print(f"  * Watertight Hull     : {result.is_watertight}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
