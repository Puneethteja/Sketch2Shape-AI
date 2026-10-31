"""
engine/generator.py
===================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Production generative diffusion pipeline conditioning Stable Diffusion v1.5
on hand-drawn sketches via ControlNet (Scribble/Lineart) adapters.

Features:
1. Dynamic initialization of Stable Diffusion v1.5 + ControlNet Scribble in FP16.
2. Text-prompt conditioning logic blending optional user descriptions with
   calibrated 3D quality booster and negative prompt templates.
3. Parameter calibration: ControlNet conditioning scale (0.7 - 0.95) and CFG (7.0 - 8.5).
4. GPU VRAM optimization: Attention Slicing, VAE Slicing, and PyTorch 2.x SDPA.
5. High-fidelity synthetic fallback generator for CPU and offline test environments.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Reconfigure stdout to utf-8 if supported to prevent Windows charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.generator")

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"


# ── Prompt Engineering Templates ─────────────────────────────────────────────

# Quality booster tokens engineered specifically for clean 3D mesh reconstruction
PROMPT_3D_BOOSTER = (
    "3d asset, single centered object, isolated on clean neutral white background, "
    "studio lighting, highly detailed, photorealistic, sharp focus, clean geometry, "
    "solid material, product render, 8k resolution"
)

# Negative prompt suppresses background noise, multiple objects, and diffusion artifacts
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, bad anatomy, deformed, watermark, "
    "text, signature, logo, shadows on background, messy background, cluttered, "
    "multiple objects, cropped, out of frame, dark background, noisy, grainy"
)


# ── Configuration Dataclass ──────────────────────────────────────────────────

@dataclass
class DiffusionConfig:
    """Hyperparameter configuration for sketch-guided diffusion generation."""
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_scribble"
    controlnet_scale: float = 0.85          # Calibrated in target 0.70 - 0.95 range
    guidance_scale: float = 7.5             # Classifier-Free Guidance (CFG) scale
    num_inference_steps: int = 30           # 20-30 for interactive latency; 50 for final
    target_size: Tuple[int, int] = (512, 512)
    device: str = "cuda"
    torch_dtype: str = "float16"            # "float16" for GPU low VRAM, "float32" for CPU
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False        # For ultra-low VRAM (< 6 GB) environments


# ── Generator Class ──────────────────────────────────────────────────────────

class SketchDiffusionGenerator:
    """
    Orchestrates sketch-guided 2.5D generation using Stable Diffusion v1.5
    and ControlNet Scribble in FP16 half-precision.
    """

    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        self.config = config or DiffusionConfig()
        self.device = self._resolve_device(self.config.device)
        self.pipeline: Optional[Any] = None
        self._is_mock = False

        self._initialize_pipeline()

    def _resolve_device(self, requested_device: str) -> str:
        """Resolves target execution device based on CUDA availability."""
        try:
            import torch
            if requested_device.startswith("cuda") and torch.cuda.is_available():
                return requested_device
        except ImportError:
            pass
        return "cpu"

    def _initialize_pipeline(self) -> None:
        """
        Attempts to load Hugging Face Diffusers pipeline in FP16.
        Falls back to high-fidelity mock generator if dependencies or checkpoints are absent.
        """
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                DPMSolverMultistepScheduler,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

            # Locate local cached weights if present
            local_sd = CHECKPOINTS_DIR / "stable_diffusion"
            local_cn = CHECKPOINTS_DIR / "controlnet" / "scribble"

            sd_id = str(local_sd) if (local_sd / "model_index.json").exists() else self.config.sd_model_id
            cn_id = str(local_cn) if (local_cn / "config.json").exists() else self.config.controlnet_model_id

            logger.info(f"Loading ControlNet adapter: {cn_id} (dtype={dtype}) ...")
            controlnet = ControlNetModel.from_pretrained(
                cn_id,
                torch_dtype=dtype,
            )

            logger.info(f"Loading Stable Diffusion v1.5 pipeline: {sd_id} ...")
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                sd_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )

            # Fast multi-step scheduler for lower latency (15-30 steps instead of 50)
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

            # Memory optimization guards for NVIDIA T4 / V100
            if self.device.startswith("cuda"):
                if self.config.enable_cpu_offload:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Sequential model CPU offload enabled.")
                else:
                    self.pipeline.to(self.device)

                if self.config.enable_attention_slicing:
                    self.pipeline.enable_attention_slicing("auto")

                if self.config.enable_vae_slicing:
                    self.pipeline.enable_vae_slicing()

            else:
                self.pipeline.to("cpu")

            logger.info("SketchDiffusionGenerator pipeline initialized successfully.")
            self._is_mock = False

        except Exception as exc:
            logger.warning(
                f"Diffusers pipeline initialization unavailable ({exc}). "
                "Operating in high-fidelity synthetic mock generator mode."
            )
            self.pipeline = None
            self._is_mock = True

    def build_conditioned_prompt(self, user_prompt: Optional[str]) -> Tuple[str, str]:
        """
        Synthesizes user description with 3D quality booster and negative prompt.
        """
        clean_user_prompt = (user_prompt or "").strip()
        if clean_user_prompt:
            # Emphasize user prompt while appending 3D reconstruction cues
            full_prompt = f"{clean_user_prompt}, {PROMPT_3D_BOOSTER}"
        else:
            full_prompt = f"a clean 3D physical object shape, {PROMPT_3D_BOOSTER}"

        return full_prompt, DEFAULT_NEGATIVE_PROMPT

    def _prepare_control_image(self, sketch_input: Union[Image.Image, np.ndarray, Any]) -> Image.Image:
        """
        Converts varied input types (PIL, NumPy array, or PyTorch tensor)
        into a 512x512 RGB PIL Image for ControlNet Scribble conditioning.
        """
        if isinstance(sketch_input, Image.Image):
            pil_img = sketch_input.convert("RGB")
        elif isinstance(sketch_input, np.ndarray):
            # Handle grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4)
            if sketch_input.ndim == 2:
                pil_img = Image.fromarray(sketch_input).convert("RGB")
            elif sketch_input.shape[-1] == 4:
                # Alpha composite over white
                img_rgba = Image.fromarray(sketch_input)
                bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
                bg.paste(img_rgba, mask=img_rgba.split()[3])
                pil_img = bg
            else:
                pil_img = Image.fromarray(sketch_input.astype(np.uint8)).convert("RGB")
        elif hasattr(sketch_input, "cpu") and hasattr(sketch_input, "numpy"):
            # PyTorch Tensor: (C, H, W) or (1, C, H, W)
            tensor_cpu = sketch_input.detach().cpu()
            if tensor_cpu.ndim == 4:
                tensor_cpu = tensor_cpu.squeeze(0)
            arr = tensor_cpu.numpy()
            if arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            pil_img = Image.fromarray(arr).convert("RGB")
        else:
            raise ValueError(f"Unsupported sketch input format: {type(sketch_input)}")

        # Ensure exact 512x512 dimensions
        if pil_img.size != self.config.target_size:
            pil_img = pil_img.resize(self.config.target_size, Image.Resampling.BILINEAR)

        return pil_img

    def generate(
        self,
        sketch_input: Union[Image.Image, np.ndarray, Any],
        prompt: Optional[str] = None,
        controlnet_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: int = -1,
    ) -> Tuple[Image.Image, int, float]:
        """
        Executes sketch-conditioned diffusion generation.

        Parameters:
            sketch_input: Hand-drawn sketch image or binary lineart tensor.
            prompt: Optional text prompt specifying style, colors, materials.
            controlnet_scale: Conditioning scale (0.0 - 2.0). Default calibrated: 0.85.
            guidance_scale: Classifier-free guidance scale. Default: 7.5.
            num_inference_steps: Denoising steps. Default: 30.
            seed: RNG seed for deterministic reproduction (-1 for random).

        Returns:
            (generated_image, concrete_seed_used, elapsed_seconds)
        """
        t0 = time.perf_counter()

        # Resolve RNG seed
        seed_used = int(seed if seed >= 0 else np.random.randint(0, 2**31 - 1))

        # Prepare prompts
        pos_prompt, neg_prompt = self.build_conditioned_prompt(prompt)

        # Calibrated parameters
        cn_scale = float(
            controlnet_scale if controlnet_scale is not None else self.config.controlnet_scale
        )
        # Enforce safe bounds documented in §3.2 (0.0 to 2.0; recommended 0.70 - 0.95)
        cn_scale = max(0.0, min(2.0, cn_scale))

        cfg_scale = float(
            guidance_scale if guidance_scale is not None else self.config.guidance_scale
        )
        steps = int(
            num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps
        )

        control_img = self._prepare_control_image(sketch_input)

        if not self._is_mock and self.pipeline is not None:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed_used)

            logger.info(
                f"Running Diffusion Generation | prompt={pos_prompt[:40]!r}... | "
                f"cn_scale={cn_scale:.2f} | cfg={cfg_scale:.1f} | steps={steps} | seed={seed_used}"
            )

            result = self.pipeline(
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                image=control_img,
                controlnet_conditioning_scale=cn_scale,
                guidance_scale=cfg_scale,
                num_inference_steps=steps,
                generator=generator,
            )
            out_img = result.images[0]
        else:
            # High-fidelity synthetic fallback generator for CPU / test environments
            out_img = self._generate_synthetic_conditioned_image(control_img, pos_prompt, seed_used)

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"Generation completed in {elapsed:.2f}s | Output size: {out_img.size}")
        return out_img, seed_used, elapsed

    def _generate_synthetic_conditioned_image(
        self,
        control_img: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        """
        Creates a synthetic shaded 2.5D visual asset adhering to sketch contours.
        Used for offline testing, CI, and CPU fallback environments.
        """
        rng = np.random.RandomState(seed % (2**31 - 1))
        ctrl_arr = np.array(control_img.convert("L")).astype(np.float32)

        # Invert if sketch is black stroke on white canvas
        if ctrl_arr.mean() > 128:
            strokes = (255.0 - ctrl_arr) / 255.0
        else:
            strokes = ctrl_arr / 255.0

        # Create smooth distance transform / shading mask
        from scipy.ndimage import gaussian_filter
        shading = gaussian_filter(strokes, sigma=8.0)
        norm_shading = (shading - shading.min()) / (shading.max() - shading.min() + 1e-6)

        # Pick base tone based on prompt keywords
        p_lower = prompt.lower()
        if "ceramic" in p_lower or "white" in p_lower:
            base_color = np.array([230, 232, 235], dtype=np.float32)
        elif "wood" in p_lower:
            base_color = np.array([170, 115, 75], dtype=np.float32)
        elif "metal" in p_lower or "gold" in p_lower:
            base_color = np.array([210, 180, 110], dtype=np.float32)
        elif "blue" in p_lower:
            base_color = np.array([80, 140, 220], dtype=np.float32)
        else:
            base_color = np.array([180, 185, 195], dtype=np.float32)

        # 3D shading model (ambient + directional light)
        H, W = ctrl_arr.shape
        Y, X = np.ogrid[:H, :W]
        light_dir = np.array([-0.5, 0.5, 0.7])
        light_dir /= np.linalg.norm(light_dir)

        grad_y, grad_x = np.gradient(norm_shading)
        grad_z = np.ones_like(norm_shading) * 0.15
        normals = np.stack([grad_x, grad_y, grad_z], axis=-1)
        norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        normals /= norm_mag

        diffuse = np.clip(np.sum(normals * light_dir, axis=-1), 0.2, 1.0)
        diffuse = np.expand_dims(diffuse, -1)

        # Composite shaded object onto clean white background
        object_mask = np.expand_dims(norm_shading > 0.05, -1)
        canvas = np.ones((H, W, 3), dtype=np.float32) * 255.0
        shaded_object = base_color * diffuse
        composed = np.where(object_mask, shaded_object, canvas)

        return Image.fromarray(np.clip(composed, 0, 255).astype(np.uint8))

    def unload(self) -> None:
        """
        Releases pipeline weights and executes CUDA cache cleanup.
        Essential for sequential execution on 16GB GPUs (NVIDIA T4).
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Evacuated SketchDiffusionGenerator weights from CUDA VRAM.")
        except ImportError:
            pass
