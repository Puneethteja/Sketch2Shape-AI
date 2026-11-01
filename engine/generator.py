"""
engine/generator.py
Sketch2Shape AI - Generative diffusion pipeline conditioning Stable Diffusion v1.5
on hand-drawn sketches via ControlNet adapters with calibrated parameters.
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

PROMPT_3D_BOOSTER = (
    "3d asset, single centered object, isolated on clean neutral white background, "
    "studio lighting, highly detailed, photorealistic, sharp focus, clean geometry, "
    "solid material, product render, 8k resolution"
)

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, bad anatomy, deformed, watermark, "
    "text, signature, logo, shadows on background, messy background, cluttered, "
    "multiple objects, cropped, out of frame, dark background, noisy, grainy"
)


@dataclass
class DiffusionConfig:
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_scribble"
    depth_controlnet_id: str = "lllyasviel/control_v11f1p_sd15_depth"
    normal_controlnet_id: str = "lllyasviel/control_v11p_sd15_normalbae"
    controlnet_scale: float = 0.80
    guidance_scale: float = 7.0
    num_inference_steps: int = 20
    target_size: Tuple[int, int] = (512, 512)
    device: str = "cuda"
    torch_dtype: str = "float16"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False


class SketchDiffusionGenerator:
    def __init__(self, config: Optional[DiffusionConfig] = None) -> None:
        self.config = config or DiffusionConfig()
        self.device = self._resolve_device(self.config.device)
        self.pipeline: Optional[Any] = None
        self._is_mock = False

        self._initialize_pipeline()

    def _resolve_device(self, requested_device: str) -> str:
        try:
            import torch
            if requested_device.startswith("cuda") and torch.cuda.is_available():
                return requested_device
        except ImportError:
            pass
        return "cpu"

    def _initialize_pipeline(self) -> None:
        try:
            import torch
            from diffusers import (
                ControlNetModel,
                DPMSolverMultistepScheduler,
                StableDiffusionControlNetPipeline,
                UniPCMultistepScheduler,
            )

            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

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

            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

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
                "AI generation is unavailable; refusing to return a synthetic stand-in."
            )
            self.pipeline = None
            self._is_mock = False

    def build_conditioned_prompt(self, user_prompt: Optional[str]) -> Tuple[str, str]:
        clean_user_prompt = (user_prompt or "").strip()
        if clean_user_prompt:
            full_prompt = f"{clean_user_prompt}, {PROMPT_3D_BOOSTER}"
        else:
            full_prompt = f"a clean 3D physical object shape, {PROMPT_3D_BOOSTER}"

        return full_prompt, DEFAULT_NEGATIVE_PROMPT

    def _prepare_control_image(self, sketch_input: Union[Image.Image, np.ndarray, Any]) -> Image.Image:
        if isinstance(sketch_input, Image.Image):
            pil_img = sketch_input.convert("RGB")
        elif isinstance(sketch_input, np.ndarray):
            if sketch_input.ndim == 2:
                pil_img = Image.fromarray(sketch_input).convert("RGB")
            elif sketch_input.shape[-1] == 4:
                img_rgba = Image.fromarray(sketch_input)
                bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
                bg.paste(img_rgba, mask=img_rgba.split()[3])
                pil_img = bg
            else:
                pil_img = Image.fromarray(sketch_input.astype(np.uint8)).convert("RGB")
        elif hasattr(sketch_input, "cpu") and hasattr(sketch_input, "numpy"):
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
        t0 = time.perf_counter()

        seed_used = int(seed if seed >= 0 else np.random.randint(0, 2**31 - 1))

        pos_prompt, neg_prompt = self.build_conditioned_prompt(prompt)

        cn_scale = float(
            controlnet_scale if controlnet_scale is not None else self.config.controlnet_scale
        )
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
            raise RuntimeError("Diffusers/ControlNet weights are unavailable. Configure Tencent TokenHub HY-3D or install the local model weights; no placeholder result was generated.")

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"Generation completed in {elapsed:.2f}s | Output size: {out_img.size}")
        return out_img, seed_used, elapsed

    def _generate_synthetic_conditioned_image(
        self,
        control_img: Image.Image,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        ctrl_rgb = np.array(control_img.convert("RGB")).astype(np.float32)
        H, W, _ = ctrl_rgb.shape

        ctrl_gray = np.dot(ctrl_rgb[..., :3], [0.2989, 0.5870, 0.1140])
        fg_mask = ctrl_gray < 245.0
        if fg_mask.sum() < 30:
            fg_mask = np.ones((H, W), dtype=bool)

        from scipy.ndimage import distance_transform_edt, gaussian_filter

        # Compute stroke and distance transform depth structure
        if ctrl_gray.mean() > 128:
            strokes = (255.0 - ctrl_gray) / 255.0
        else:
            strokes = ctrl_gray / 255.0

        dist = distance_transform_edt(fg_mask).astype(np.float32)
        norm_depth = dist / (dist.max() + 1e-6)
        shading = gaussian_filter(0.7 * norm_depth + 0.3 * strokes, sigma=4.0)
        norm_shading = (shading - shading.min()) / (shading.max() - shading.min() + 1e-6)

        # Check if the input image contains native RGB color information (not pure gray/sketch)
        # Channel std > 6.0 indicates chromatic color (R, G, B are distinct)
        channel_std = np.std(ctrl_rgb[fg_mask, :3], axis=-1) if fg_mask.any() else np.array([0.0])
        has_native_color = float(np.mean(channel_std)) > 6.0

        p_lower = prompt.lower()

        if has_native_color:
            # Color Preservation Pass: preserve exact user RGB colors and surface saturation
            native_albedo = ctrl_rgb.copy()
            # Slight contrast boost to prevent desaturation
            native_albedo = np.clip((native_albedo - 128.0) * 1.10 + 128.0, 0.0, 255.0)
            base_albedo = native_albedo
        else:
            # Map prompt material keywords to vibrant, saturated color palettes (no dull gray!)
            if "ceramic" in p_lower or "porcelain" in p_lower or "white" in p_lower:
                base_col = np.array([242, 244, 248], dtype=np.float32)
            elif "gold" in p_lower:
                base_col = np.array([238, 198, 68], dtype=np.float32)
            elif "titanium" in p_lower:
                base_col = np.array([178, 188, 198], dtype=np.float32)
            elif "wood" in p_lower or "walnut" in p_lower:
                base_col = np.array([142, 78, 42], dtype=np.float32)
            elif "terracotta" in p_lower or "clay" in p_lower:
                base_col = np.array([212, 108, 68], dtype=np.float32)
            elif "glass" in p_lower:
                base_col = np.array([168, 228, 238], dtype=np.float32)
            elif "obsidian" in p_lower:
                base_col = np.array([36, 40, 52], dtype=np.float32)
            elif "coral" in p_lower:
                base_col = np.array([248, 128, 112], dtype=np.float32)
            elif "blue" in p_lower:
                base_col = np.array([55, 125, 235], dtype=np.float32)
            elif "red" in p_lower or "apple" in p_lower:
                base_col = np.array([225, 38, 38], dtype=np.float32)
            elif "green" in p_lower:
                base_col = np.array([48, 185, 88], dtype=np.float32)
            else:
                base_col = np.array([225, 228, 236], dtype=np.float32)

            base_albedo = np.tile(base_col.reshape(1, 1, 3), (H, W, 1))

        # Surface Normals from depth gradient
        grad_y, grad_x = np.gradient(norm_shading * 2.0)
        grad_z = np.ones_like(norm_shading) * 0.20
        normals = np.stack([-grad_x, -grad_y, grad_z], axis=-1)
        norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
        normals /= norm_mag

        # Key studio directional light
        light_dir = np.array([-0.4, 0.6, 0.7], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        # Diffuse component (Lambertian with wrap lighting to avoid harsh dark spots)
        n_dot_l = np.sum(normals * light_dir, axis=-1, keepdims=True)
        diffuse = np.clip(0.65 + 0.35 * n_dot_l, 0.40, 1.0)

        # Specular highlight component (Blinn-Phong)
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vec = light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)
        n_dot_h = np.clip(np.sum(normals * half_vec, axis=-1, keepdims=True), 0.0, 1.0)
        specular = (n_dot_h ** 20) * 45.0  # Crisp specular sheen

        # Composite shaded object
        object_mask = np.expand_dims(fg_mask & (norm_shading > 0.03), -1)
        shaded_object = np.clip(base_albedo * diffuse + specular, 0.0, 255.0)

        canvas = np.ones((H, W, 3), dtype=np.float32) * 255.0
        composed = np.where(object_mask, shaded_object, canvas)

        return Image.fromarray(composed.astype(np.uint8))

    def unload(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
            del self.pipeline
            self.pipeline = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                logger.info("Evacuated SketchDiffusionGenerator weights from CUDA VRAM.")
        except ImportError:
            pass
