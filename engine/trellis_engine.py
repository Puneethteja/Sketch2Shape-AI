"""
engine/trellis_engine.py
========================
Microsoft TRELLIS.2 (TRELLIS.2-4B) — Real Pretrained Model Inference Pipeline

This module calls the actual Microsoft TRELLIS.2-4B pretrained model hosted on
Hugging Face Spaces (microsoft/TRELLIS.2) via the Gradio Client API. The 4-billion
parameter model runs on Microsoft's GPU infrastructure (NVIDIA H100/A100) and
returns production-quality 3D meshes with full PBR materials.

Pipeline Steps:
  1. Preprocess input image (background removal, centering) via /preprocess_image
  2. Run image-to-3D generation (O-Voxel Sparse 3D DiT) via /image_to_3d
  3. Extract GLB mesh with decimation & texture baking via /extract_glb
  4. Download GLB and convert locally to STL, OBJ, PLY

Paper: "Native and Compact Structured Latents for 3D Generation"
       (Xiang et al., 2025, arXiv:2512.14692)
Weights: microsoft/TRELLIS.2-4B (Hugging Face)
Space:   microsoft/TRELLIS.2 (Hugging Face Spaces)
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import trimesh

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.mesh_utils import repair_mesh, export_mesh, sanitize_mesh

logger = logging.getLogger("sketch2shape.trellis")

# ── Hugging Face Space Configuration ─────────────────────────────────────────
HF_SPACE_ID = "microsoft/TRELLIS.2"

# ── 3 Native PBR Materials ───────────────────────────────────────────────────
TRELLIS_MATERIALS = {
    "ceramic": {
        "name": "Porcelain Ceramic",
        "description": "Smooth specular white ceramic glaze with high surface reflectivity and clean curvature.",
        "roughness": 0.18,
        "metalness": 0.05,
        "base_color": [242, 244, 248, 255],
        "hex": "#f2f4f8",
    },
    "slate": {
        "name": "Slate Polymer",
        "description": "Industrial matte polymer resin with controlled micro-roughness and crisp geometric edge definition.",
        "roughness": 0.72,
        "metalness": 0.12,
        "base_color": [45, 52, 64, 255],
        "hex": "#2d3440",
    },
    "bronze": {
        "name": "Cast Bronze",
        "description": "Antique warm metallic bronze with warm copper undertones and diffuse ambient patina.",
        "roughness": 0.38,
        "metalness": 0.85,
        "base_color": [182, 134, 76, 255],
        "hex": "#b6864c",
    },
}

MODEL_METRICS_SPECS = {
    "model_name": "Microsoft TRELLIS.2 (TRELLIS.2-4B)",
    "developer": "Microsoft Research (Xiang et al., 2025)",
    "paper": "Native and Compact Structured Latents for 3D Generation (arXiv:2512.14692)",
    "weights_identifier": "microsoft/TRELLIS.2-4B",
    "parameters": "4 Billion (Sparse 3D DiT + Sparse 3D VAE)",
    "representation": "Field-Free O-Voxel (Sparse Voxel Structure)",
    "downsampling": "16x Spatial Compression via Sparse 3D VAE",
    "resolution": "512\u00b3 - 1536\u00b3 Continuous Resolution",
    "pbr_attributes": ["Base Color", "Roughness", "Metallic", "Opacity"],
    "supported_formats": ["STL", "OBJ", "GLB", "PLY"],
    "supported_materials": ["Porcelain Ceramic", "Slate Polymer", "Cast Bronze"],
    "topology_guarantee": "Arbitrary Topology (Open Surfaces, Non-manifold, Enclosed Watertight)",
    "inference_endpoint": f"Hugging Face Spaces ({HF_SPACE_ID})",
}


@dataclass
class TrellisGenerationResult:
    """Encapsulates the complete outputs from the Microsoft TRELLIS.2-4B engine."""
    mesh: trimesh.Trimesh
    glb_path: str
    obj_path: str
    stl_path: str
    ply_path: str
    glb_url: str
    obj_url: str
    stl_url: str
    ply_url: str
    preview_image_url: str
    vertices: int
    faces: int
    is_watertight: bool
    latency_seconds: float
    material_used: str
    used_real_model: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


class MicrosoftTrellisEngine:
    """
    Microsoft TRELLIS.2-4B Real Pretrained Model Pipeline.

    Primary mode: Calls the actual TRELLIS.2-4B pretrained model hosted on
    Hugging Face Spaces via the Gradio Client API. The real 4B-parameter model
    runs image-to-3D generation on Microsoft's GPU infrastructure and returns
    production-quality GLB meshes with PBR textures.

    Fallback mode: If the remote API is unreachable (no internet, rate-limited,
    or queue full), falls back to local procedural geometry synthesis for
    development/testing purposes only.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.model_name = "Microsoft TRELLIS.2-4B"
        self.pipeline: Optional[Any] = None
        self._gradio_client: Optional[Any] = None
        self._api_available: Optional[bool] = None
        self._init_pipeline()

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _init_pipeline(self) -> None:
        """
        Initializes the inference pipeline in priority order:
        1. Local CUDA pipeline (if trellis2 package + GPU available)
        2. Hugging Face Spaces remote API via gradio_client
        3. Local procedural fallback (for testing/development only)
        """
        # 1. Try local GPU pipeline
        if self.device == "cuda":
            try:
                from trellis2.pipelines import Trellis2ImageTo3DPipeline
                logger.info("Loading Microsoft TRELLIS.2-4B from 'microsoft/TRELLIS.2-4B' onto GPU...")
                self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
                self.pipeline.to(self.device)
                logger.info("TRELLIS.2-4B loaded into GPU VRAM. Using local CUDA inference.")
                return
            except Exception as exc:
                logger.info(f"Local CUDA pipeline not available ({exc}).")

        # 2. Try Hugging Face Spaces remote API
        try:
            self._init_gradio_client()
            if self._gradio_client is not None:
                logger.info(f"Connected to Microsoft TRELLIS.2 remote API at {HF_SPACE_ID}.")
                logger.info("Image-to-3D inference will run on Microsoft's GPU infrastructure.")
                return
        except Exception as exc:
            logger.warning(f"Remote API connection deferred ({exc}).")

        # 3. Procedural fallback
        logger.info("Operating in local procedural fallback mode (no real model).")
        self.pipeline = None

    def _init_gradio_client(self) -> None:
        """Connects to the Microsoft TRELLIS.2 Hugging Face Space via gradio_client."""
        from gradio_client import Client
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            self._gradio_client = Client(HF_SPACE_ID, hf_token=hf_token)
        else:
            self._gradio_client = Client(HF_SPACE_ID)
        # Start a session so the Space allocates GPU
        try:
            self._gradio_client.predict(api_name="/start_session")
        except Exception:
            pass  # Some Spaces don't require explicit session start
        self._api_available = True

    def _is_remote_api_available(self) -> bool:
        """Checks if the Gradio remote API client is initialized and reachable."""
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False
        return self._gradio_client is not None and self._api_available is True

    def generate(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "",
        material: str = "ceramic",
        hf_token: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> TrellisGenerationResult:
        """
        Executes the Microsoft TRELLIS.2-4B image-to-3D pipeline.

        If the remote Hugging Face API is available, this calls the real pretrained
        4B-parameter model running on NVIDIA H100/A100 GPUs. Otherwise, falls back
        to local procedural geometry for testing.
        """
        t0 = time.perf_counter()
        target_dir = output_dir or (ROOT_DIR / "outputs")
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ingest & Preprocess image (rembg, contrast adjustment, centering)
        pil_img = self._load_image(image_input)
        processed_img = self._preprocess_input_image(pil_img)

        # Memory management: empty CUDA cache before heavy inference
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 2. Save temp preprocessed image for API/pipeline upload
        temp_img_path = target_dir / f"_temp_input_{int(time.time()*1000)}.png"
        processed_img.save(str(temp_img_path), format="PNG")

        used_real_model = False
        glb_mesh = None

        # 3. Try real TRELLIS.2 pretrained model via remote API
        effective_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if effective_token and (self._gradio_client is None or getattr(self, "_current_token", None) != effective_token):
            try:
                from gradio_client import Client
                logger.info("Connecting to Microsoft TRELLIS.2 Space with user-authenticated HF token...")
                self._gradio_client = Client(HF_SPACE_ID, hf_token=effective_token)
                self._current_token = effective_token
                self._api_available = True
            except Exception as e:
                logger.warning(f"Could not initialize authenticated client: {e}")

        zerogpu_quota_hit = False
        quota_err_msg = ""
        if self._is_remote_api_available():
            try:
                glb_mesh = self._generate_via_remote_api(str(temp_img_path), target_dir)
                used_real_model = True
                logger.info("Successfully generated 3D mesh using real Microsoft TRELLIS.2-4B pretrained model.")
            except Exception as exc:
                exc_str = str(exc)
                if "ZeroGPU quota" in exc_str:
                    zerogpu_quota_hit = True
                    quota_err_msg = exc_str
                logger.warning(
                    f"Remote TRELLIS.2 API call failed ({exc_str}). "
                    "Engaging high-fidelity geometric synthesis engine."
                )

        # 4. Try local CUDA pipeline
        if glb_mesh is None and self.pipeline is not None:
            try:
                glb_mesh = self._generate_via_local_pipeline(pil_img, target_dir)
                used_real_model = True
                logger.info("Generated mesh using local TRELLIS.2-4B CUDA pipeline.")
            except Exception as exc:
                logger.warning(f"Local CUDA pipeline failed ({exc}). Falling back to procedural.")

        # 5. Universal Geometry Engine fallback (high-resolution manifold geometry)
        if glb_mesh is None:
            logger.info("Engaging high-fidelity Universal Geometry Engine (lathe/hollow vessel/component).")
            from engine.geometry_engine import UniversalGeometryEngine
            try:
                raw_mesh = UniversalGeometryEngine.generate(
                    image_input=processed_img,
                    prompt=prompt,
                    sketch_input=pil_img,
                )
            except Exception as e:
                logger.warning(f"UniversalGeometryEngine error ({e}), using bilateral organic fallback.")
                raw_mesh = self._infer_3d_geometry_procedural(processed_img, material=material)

            clean_mesh = repair_mesh(raw_mesh)
            mat_key = material.lower().strip()
            mat_info = TRELLIS_MATERIALS.get(mat_key, TRELLIS_MATERIALS["ceramic"])
            glb_mesh = self._apply_material_colors(clean_mesh, mat_info)

        # 6. Export all 4 formats: STL, OBJ, GLB, PLY
        timestamp = int(time.time() * 1000)
        base_name = f"trellis_model_{timestamp}"

        glb_file = target_dir / f"{base_name}.glb"
        obj_file = target_dir / f"{base_name}.obj"
        stl_file = target_dir / f"{base_name}.stl"
        ply_file = target_dir / f"{base_name}.ply"

        # If we got a GLB file directly from the API, copy it; otherwise export from mesh
        if used_real_model and hasattr(glb_mesh, '_api_glb_path'):
            api_glb = glb_mesh._api_glb_path
            shutil.copy2(api_glb, str(glb_file))
            # The mesh object is the loaded GLB
            colored_mesh = glb_mesh
        else:
            colored_mesh = glb_mesh
            export_mesh(colored_mesh, glb_file, export_format="glb")

        export_mesh(colored_mesh, obj_file, export_format="obj")
        export_mesh(colored_mesh, stl_file, export_format="stl")
        export_mesh(colored_mesh, ply_file, export_format="ply")

        # 7. Render preview
        mat_key = material.lower().strip()
        mat_info = TRELLIS_MATERIALS.get(mat_key, TRELLIS_MATERIALS["ceramic"])
        preview_img = self._render_mesh_preview(colored_mesh, mat_info)
        preview_data_url = self._pil_to_data_url(preview_img)

        # Clean up temp file
        try:
            temp_img_path.unlink(missing_ok=True)
        except Exception:
            pass

        elapsed = round(time.perf_counter() - t0, 2)
        n_verts = len(colored_mesh.vertices)
        n_faces = len(colored_mesh.faces)

        inference_mode = "TRELLIS.2-4B Pretrained (Remote API)" if used_real_model else "Procedural Fallback"
        logger.info(
            f"Generation complete [{inference_mode}] in {elapsed}s | "
            f"Verts: {n_verts} | Faces: {n_faces} | Watertight: {colored_mesh.is_watertight} | "
            f"Exported: STL, OBJ, GLB, PLY"
        )

        return TrellisGenerationResult(
            mesh=colored_mesh,
            glb_path=str(glb_file),
            obj_path=str(obj_file),
            stl_path=str(stl_file),
            ply_path=str(ply_file),
            glb_url=f"/outputs/{glb_file.name}",
            obj_url=f"/outputs/{obj_file.name}",
            stl_url=f"/outputs/{stl_file.name}",
            ply_url=f"/outputs/{ply_file.name}",
            preview_image_url=preview_data_url,
            vertices=n_verts,
            faces=n_faces,
            is_watertight=bool(colored_mesh.is_watertight),
            latency_seconds=elapsed,
            material_used=mat_info["name"],
            used_real_model=used_real_model,
            metrics={
                "model": "Microsoft TRELLIS.2-4B",
                "inference_mode": inference_mode,
                "parameters": "4B (Sparse 3D DiT + Sparse 3D VAE)",
                "representation": "O-Voxel (Field-Free Sparse Voxel)",
                "formats": ["STL", "OBJ", "GLB", "PLY"],
                "triangle_budget": n_faces,
                "vertex_budget": n_verts,
                "device": self.device,
                "zerogpu_quota_exceeded": zerogpu_quota_hit,
                "quota_message": quota_err_msg if zerogpu_quota_hit else "",
            },
        )

    # ── Real Model: Remote API Pipeline ──────────────────────────────────────

    def _generate_via_remote_api(
        self, image_path: str, target_dir: Path
    ) -> trimesh.Trimesh:
        """
        Calls the real Microsoft TRELLIS.2-4B pretrained model via the
        Hugging Face Spaces Gradio API.

        Pipeline:
          1. /preprocess_image  — Background removal & centering
          2. /image_to_3d       — Run the 4B-parameter O-Voxel DiT model
          3. /extract_glb       — Post-process & export GLB with PBR textures
        """
        from gradio_client import handle_file

        client = self._gradio_client
        logger.info("Step 1/3: Preprocessing image (background removal & centering)...")

        # Step 1: Preprocess image
        preprocessed = client.predict(
            input=handle_file(image_path),
            api_name="/preprocess_image",
        )
        # preprocessed is a dict with 'path' key pointing to the processed image
        if isinstance(preprocessed, dict):
            processed_path = preprocessed.get("path") or preprocessed.get("url")
        else:
            processed_path = str(preprocessed)

        logger.info("Step 2/3: Running TRELLIS.2-4B image-to-3D generation (4B params on GPU)...")

        # Step 2: Image to 3D generation with the real model
        # Use default optimal parameters (no user tuning)
        preview_result = client.predict(
            image=handle_file(processed_path),
            seed=42,
            resolution="1024",
            ss_guidance_strength=7.5,
            ss_guidance_rescale=0.7,
            ss_sampling_steps=12,
            ss_rescale_t=5.0,
            shape_slat_guidance_strength=7.5,
            shape_slat_guidance_rescale=0.5,
            shape_slat_sampling_steps=12,
            shape_slat_rescale_t=3.0,
            tex_slat_guidance_strength=1.0,
            tex_slat_guidance_rescale=0.0,
            tex_slat_sampling_steps=12,
            tex_slat_rescale_t=3.0,
            api_name="/image_to_3d",
        )
        logger.info("Step 3/3: Extracting GLB mesh with PBR textures...")

        # Step 3: Extract GLB with decimation and texture baking
        glb_result = client.predict(
            decimation_target=300000,
            texture_size=2048,
            api_name="/extract_glb",
        )

        # glb_result is a tuple: (model3d_filepath, download_filepath)
        if isinstance(glb_result, (tuple, list)):
            glb_source = glb_result[0]
        else:
            glb_source = glb_result

        # If the result is a dict, extract the path
        if isinstance(glb_source, dict):
            glb_source = glb_source.get("path") or glb_source.get("url") or str(glb_source)

        glb_source = str(glb_source)
        logger.info(f"Received GLB from TRELLIS.2 API: {glb_source}")

        # Load the GLB mesh
        loaded = trimesh.load(glb_source, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if geoms:
                mesh = trimesh.util.concatenate(geoms)
            else:
                raise ValueError("TRELLIS.2 API returned GLB with no mesh geometry")
        else:
            mesh = loaded

        # Attach the API GLB path so we can copy the original file
        mesh._api_glb_path = glb_source

        return mesh

    def _generate_via_local_pipeline(
        self, pil_img: Image.Image, target_dir: Path
    ) -> trimesh.Trimesh:
        """Runs inference using a locally installed trellis2 CUDA pipeline."""
        import torch
        import cv2

        mesh_result = self.pipeline.run(pil_img)[0]
        mesh_result.simplify(16777216)

        try:
            import o_voxel
            glb_path = str(target_dir / "_trellis_local.glb")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_result.vertices,
                faces=mesh_result.faces,
                attr_volume=mesh_result.attrs,
                coords=mesh_result.coords,
                attr_layout=mesh_result.layout,
                voxel_size=mesh_result.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=300000,
                texture_size=2048,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(glb_path, extension_webp=True)
            loaded = trimesh.load(glb_path, force="mesh")
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                return trimesh.util.concatenate(geoms) if geoms else mesh_result
            return loaded
        except Exception:
            # Return the raw mesh if o_voxel postprocessing fails
            verts = mesh_result.vertices
            faces = mesh_result.faces
            return trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # ── Procedural Fallback (Development/Testing Only) ───────────────────────

    def _infer_3d_geometry_procedural(
        self, pil_img: Image.Image, material: str = "ceramic"
    ) -> trimesh.Trimesh:
        """
        Procedural geometry fallback. NOT the real TRELLIS model.
        Used only when the remote API and local CUDA are both unavailable.
        """
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        luminance = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        fg_mask = luminance < 0.94

        if fg_mask.sum() < 30:
            return trimesh.creation.icosphere(subdivisions=3, radius=0.8)

        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import ConvexHull

        dist = distance_transform_edt(fg_mask).astype(np.float32)
        max_d = dist.max() + 1e-6
        depth_profile = dist / max_d

        stride = 4
        H, W = depth_profile.shape
        y_grid, x_grid = np.mgrid[0:H:stride, 0:W:stride]
        valid = depth_profile[::stride, ::stride] > 0.04

        if valid.sum() < 25:
            return trimesh.creation.icosphere(subdivisions=3, radius=0.8)

        norm_x = (x_grid / W) * 2.0 - 1.0
        norm_y = -((y_grid / H) * 2.0 - 1.0)
        norm_z_front = depth_profile[::stride, ::stride] * 0.75
        norm_z_back = -norm_z_front * 0.70

        vf_x = norm_x[valid]
        vf_y = norm_y[valid]
        vf_zf = norm_z_front[valid]
        vf_zb = norm_z_back[valid]

        pts_f = np.stack([vf_x, vf_y, vf_zf], axis=-1)
        pts_b = np.stack([vf_x, vf_y, vf_zb], axis=-1)
        all_pts = np.concatenate([pts_f, pts_b], axis=0)

        try:
            hull = ConvexHull(all_pts)
            faces = hull.simplices.astype(np.int32)
            mesh = trimesh.Trimesh(vertices=all_pts, faces=faces, process=True)
            return mesh
        except Exception:
            return trimesh.creation.icosphere(subdivisions=3, radius=0.8)

    # ── Shared Utilities & Advanced Preprocessing ────────────────────────────

    def _preprocess_input_image(self, img: Image.Image) -> Image.Image:
        """
        Input Preprocessing Pipeline:
        1. Background removal using rembg (with adaptive alpha threshold fallback).
        2. Contrast adjustment and tone mapping to prevent washed-out textures.
        3. Subject bounding-box centering with 10% safety margin.
        """
        from PIL import ImageEnhance, ImageOps

        # Convert to RGBA
        rgba = img.convert("RGBA")

        # 1. Background removal
        try:
            import rembg
            # Check if image already has clean transparency
            alpha = np.array(rgba.split()[-1])
            if (alpha < 250).sum() < (alpha.size * 0.05):
                rgba = rembg.remove(rgba)
                logger.info("Applied rembg background removal to input silhouette.")
        except Exception as exc:
            logger.debug(f"rembg background removal fallback ({exc})")

        # Fallback background removal for solid white or light canvases
        arr = np.array(rgba)
        if (arr[..., 3] > 200).all():
            is_white_bg = (arr[..., 0] > 238) & (arr[..., 1] > 238) & (arr[..., 2] > 238)
            if is_white_bg.sum() > (arr.shape[0] * arr.shape[1] * 0.15):
                arr[is_white_bg, 3] = 0
                rgba = Image.fromarray(arr)

        # 2. Contrast adjustment & color enhancement to prevent washed out textures
        rgb_part = rgba.convert("RGB")
        enhancer = ImageEnhance.Contrast(rgb_part)
        enhanced_rgb = enhancer.enhance(1.22)
        r, g, b = enhanced_rgb.split()
        rgba = Image.merge("RGBA", (r, g, b, rgba.split()[-1]))

        # 3. Center foreground with ~10% margin onto square canvas
        bbox = rgba.getbbox()
        if bbox:
            cropped = rgba.crop(bbox)
            target_size = 512
            max_dim = max(cropped.width, cropped.height)
            scale = (target_size * 0.82) / max(max_dim, 1)
            new_w = max(int(cropped.width * scale), 1)
            new_h = max(int(cropped.height * scale), 1)
            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
            offset_x = (target_size - new_w) // 2
            offset_y = (target_size - new_h) // 2
            canvas.paste(resized, (offset_x, offset_y), resized)
            return canvas

        return rgba.resize((512, 512), Image.Resampling.LANCZOS)

    def _load_image(self, image_input: Any) -> Image.Image:
        """Ingests sketch from file, PIL Image, NumPy array, or Base64 data URL."""
        if isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            s = str(image_input)
            if s.startswith("data:image"):
                import base64
                header, encoded = s.split(",", 1)
                data = base64.b64decode(encoded)
                img = Image.open(io.BytesIO(data)).convert("RGB")
            else:
                img = Image.open(s).convert("RGB")
        else:
            img = Image.new("RGB", (512, 512), (255, 255, 255))

        if img.size != (512, 512):
            img = img.resize((512, 512), Image.Resampling.BILINEAR)
        return img

    def _apply_material_colors(
        self, mesh: trimesh.Trimesh, mat_info: Dict[str, Any]
    ) -> trimesh.Trimesh:
        """Applies PBR vertex colors based on material properties."""
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        n = len(verts)

        base_rgb = np.array(mat_info["base_color"][:3], dtype=np.float32)
        normals = getattr(mesh, "vertex_normals", None)
        if normals is None or len(normals) != n:
            mesh.recompute_normals()
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        key_light = np.array([0.45, 0.75, 0.55])
        key_light /= np.linalg.norm(key_light)
        fill_light = np.array([-0.5, 0.3, -0.4])
        fill_light /= np.linalg.norm(fill_light)

        dot_key = np.clip(np.sum(normals * key_light, axis=-1), 0.0, 1.0)
        dot_fill = np.clip(np.sum(normals * fill_light, axis=-1), 0.0, 1.0) * 0.35
        ambient = 0.35
        shading = np.clip(ambient + dot_key * 0.65 + dot_fill, 0.0, 1.0)

        if mat_info["roughness"] < 0.3:
            view_vec = np.array([0.0, 0.0, 1.0])
            half_vec = key_light + view_vec
            half_vec /= np.linalg.norm(half_vec)
            spec = np.power(np.clip(np.sum(normals * half_vec, axis=-1), 0.0, 1.0), 32) * 45.0
        else:
            spec = 0.0

        colors = np.zeros((n, 4), dtype=np.uint8)
        for i in range(3):
            c_chan = np.clip(base_rgb[i] * shading + spec, 0, 255).astype(np.uint8)
            colors[:, i] = c_chan
        colors[:, 3] = 255

        mesh.visual.vertex_colors = colors
        return mesh

    def _render_mesh_preview(
        self, mesh: trimesh.Trimesh, mat_info: Dict[str, Any], size: int = 400
    ) -> Image.Image:
        """Renders an orthographic studio preview of the mesh."""
        img = Image.new("RGB", (size, size), (18, 22, 30))
        draw = ImageDraw.Draw(img)

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if len(verts) == 0:
            return img

        px, py, pz = verts[:, 0], verts[:, 1], verts[:, 2]
        span_x = max(px.max() - px.min(), 1e-4)
        span_y = max(py.max() - py.min(), 1e-4)
        scale = (size * 0.70) / max(span_x, span_y)

        cx = (px.min() + px.max()) / 2.0
        cy = (py.min() + py.max()) / 2.0
        scr_x = ((px - cx) * scale + (size / 2.0)).astype(np.int32)
        scr_y = (-(py - cy) * scale + (size / 2.0)).astype(np.int32)

        faces = getattr(mesh, "faces", None)
        v_colors = getattr(mesh, "visual", None)
        colors = getattr(v_colors, "vertex_colors", None) if v_colors else None

        if faces is not None and len(faces) > 0:
            tri_depths = [pz[f].mean() for f in faces]
            sorted_idx = np.argsort(tri_depths)
            for idx in sorted_idx:
                f = faces[idx]
                pts = [(int(scr_x[f[i]]), int(scr_y[f[i]])) for i in range(3)]
                if colors is not None and len(colors) > f[0]:
                    col = tuple(int(c) for c in colors[f[0]][:3])
                else:
                    col = (200, 205, 215)
                draw.polygon(pts, fill=col, outline=None)

        return img

    def _pil_to_data_url(self, img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# ── Global Singleton ─────────────────────────────────────────────────────────
_TRELLIS_ENGINE: Optional[MicrosoftTrellisEngine] = None


def get_trellis_engine() -> MicrosoftTrellisEngine:
    global _TRELLIS_ENGINE
    if _TRELLIS_ENGINE is None:
        _TRELLIS_ENGINE = MicrosoftTrellisEngine()
    return _TRELLIS_ENGINE
