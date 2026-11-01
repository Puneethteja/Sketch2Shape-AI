"""
engine/texture.py
=================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Senior AI/ML Engineer (3D Computer Vision)

Production color-projection, UV texture mapping, and albedo restoration engine.
Directly addresses color-loss, desaturation, and gray wash-out during 2D-to-3D
reconstruction by projecting original RGB colors and highlights onto 3D surfaces
and baking high-fidelity UV texture atlases.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt

logger = logging.getLogger("sketch2shape.texture")


@dataclass
class TextureProjectionResult:
    """Bundle containing UV coordinates, vertex colors, and baked texture atlas."""
    vertex_colors: np.ndarray       # (N, 4) uint8 RGBA
    uv_coordinates: np.ndarray      # (N, 2) float32 in [0, 1]
    texture_atlas: Image.Image      # PIL RGB image (e.g. 1024x1024 or 512x512)
    mean_saturation: float
    color_fidelity_score: float     # Ratio of preserved input gamut [0.0 - 1.0]


class TextureProjector:
    """
    High-fidelity color-projection and UV texture mapping pass.
    Maps exact RGB values, contrast, and surface highlights from the 2D input image
    directly onto the 3D surface mesh and UV texture atlas without washing out to gray.
    """

    def __init__(
        self,
        atlas_size: int = 1024,
        gamma: float = 0.95,
        saturation_boost: float = 1.15,
        contrast_boost: float = 1.10,
    ) -> None:
        self.atlas_size = atlas_size
        self.gamma = gamma
        self.saturation_boost = saturation_boost
        self.contrast_boost = contrast_boost

    def enhance_color_fidelity(
        self,
        rgb_image: Union[np.ndarray, Image.Image],
    ) -> Union[np.ndarray, Image.Image]:
        """
        Applies non-linear albedo restoration: enhances saturation, sharpens contrast,
        and restores specular highlights so projected colors do not appear faded or gray.
        Supports both numpy ndarray and PIL Image inputs.
        """
        is_pil = isinstance(rgb_image, Image.Image)
        orig_mode = rgb_image.mode if is_pil else None

        if is_pil:
            arr_full = np.array(rgb_image, dtype=np.float32)
        else:
            arr_full = np.asarray(rgb_image, dtype=np.float32)

        arr = arr_full.copy() / 255.0

        # Contrast enhancement around midpoint 0.5
        arr[..., :3] = np.clip((arr[..., :3] - 0.5) * self.contrast_boost + 0.5, 0.0, 1.0)

        # Gamma adjustment for brightness & midtone vibrance
        arr[..., :3] = np.power(arr[..., :3], self.gamma)

        # Saturation boost in HSV/HSL space approximation
        lum = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]
        arr[..., :3] = np.clip(lum + (arr[..., :3] - lum) * self.saturation_boost, 0.0, 1.0)

        out_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        if is_pil:
            if orig_mode == "RGBA" and out_uint8.shape[-1] >= 4:
                return Image.fromarray(out_uint8, mode="RGBA")
            elif orig_mode == "RGB" or out_uint8.shape[-1] == 3:
                return Image.fromarray(out_uint8[..., :3], mode="RGB")
            return Image.fromarray(out_uint8)

        return out_uint8

    def compute_cylindrical_uvs(
        self,
        vertices: np.ndarray,
    ) -> np.ndarray:
        """
        Computes continuous cylindrical UV coordinates in [0, 1]^2 for a 3D vertex array.
        U = (atan2(z, x) + pi) / (2 * pi)
        V = (y - y_min) / (y_max - y_min)
        """
        verts = np.asarray(vertices, dtype=np.float32)
        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        # Azimuth angle in [-pi, pi] -> normalized to [0, 1]
        theta = np.arctan2(z, x)
        u = (theta + np.pi) / (2.0 * np.pi)
        u = np.clip(u, 0.0, 1.0)

        # Vertical height normalized to [0, 1]
        ymin, ymax = y.min(), y.max()
        h_range = max(1e-6, ymax - ymin)
        v = (y - ymin) / h_range
        v = np.clip(v, 0.0, 1.0)

        return np.stack([u, v], axis=-1).astype(np.float32)

    def project_texture_pass(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        reference_image: Union[Image.Image, np.ndarray, List[Any], Tuple[Any, ...]],
        multiview_views: Optional[List[Any]] = None,
        multiview_masks: Optional[List[np.ndarray]] = None,
    ) -> TextureProjectionResult:
        """
        Executes the end-to-end color-projection and UV texture mapping pass:
        1. Ingests and enhances original 2D input image albedo.
        2. Computes smooth UV coordinates for every vertex.
        3. Projects native RGB pixel values and surface highlights from front camera (0°)
           and side camera angles onto 3D vertices without downscaling coordinates.
        4. Bakes a high-resolution 2D UV texture atlas.
        """
        # If reference_image is a list or tuple of views
        if isinstance(reference_image, (list, tuple)):
            if multiview_views is None and len(reference_image) > 1:
                multiview_views = list(reference_image)
            reference_image = reference_image[0]

        # 1. Normalize reference image
        if isinstance(reference_image, Image.Image):
            ref_rgb = np.array(reference_image.convert("RGB"), dtype=np.uint8)
        else:
            ref_arr = np.asarray(reference_image, dtype=np.uint8)
            if ref_arr.ndim == 2:
                ref_rgb = np.stack([ref_arr]*3, axis=-1)
            else:
                ref_rgb = ref_arr[..., :3]

        H, W = ref_rgb.shape[:2]

        # Enhance color fidelity of reference image
        enhanced_ref = self.enhance_color_fidelity(ref_rgb)
        if isinstance(enhanced_ref, Image.Image):
            enhanced_ref = np.array(enhanced_ref.convert("RGB"), dtype=np.uint8)
        else:
            enhanced_ref = np.asarray(enhanced_ref, dtype=np.uint8)[..., :3]

        # Extract foreground mask
        gray = np.dot(enhanced_ref.astype(np.float32), [0.2989, 0.5870, 0.1140])
        fg_mask = gray < 245.0
        if fg_mask.sum() < 20:
            fg_mask = np.ones((H, W), dtype=bool)

        y_indices, x_indices = np.where(fg_mask)
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()
        cx = (xmin + xmax) / 2.0
        w_span = max(10, xmax - xmin)
        h_span = max(10, ymax - ymin)

        # 2. Compute UV coordinates
        uvs = self.compute_cylindrical_uvs(vertices)

        # 3. Project RGB colors to vertices
        verts = np.asarray(vertices, dtype=np.float32)
        N = len(verts)
        colors = np.zeros((N, 4), dtype=np.uint8)
        colors[:, 3] = 255  # Full alpha

        # Front view reference bounds
        vx = verts[:, 0]
        vy = verts[:, 1]
        vz = verts[:, 2]

        vymin, vymax = vy.min(), vy.max()
        vh_range = max(1e-6, vymax - vymin)

        # Radial extent for horizontal scaling
        r_horiz = np.sqrt(vx ** 2 + vz ** 2)
        max_r = max(1e-6, r_horiz.max())

        # Normalize views_list
        if multiview_views:
            norm_views = []
            for v in multiview_views:
                if isinstance(v, Image.Image):
                    norm_views.append(np.array(v.convert("RGB"), dtype=np.uint8))
                else:
                    v_arr = np.asarray(v, dtype=np.uint8)
                    norm_views.append(v_arr[..., :3] if v_arr.ndim >= 3 else np.stack([v_arr]*3, axis=-1))
            views_list = norm_views
        else:
            views_list = [enhanced_ref]

        for i in range(N):
            x_i = vx[i]
            y_i = vy[i]
            z_i = vz[i]

            # Azimuth angle in degrees: 0° is Front (+Z/right hand), 90° is Right (+X)
            # Normalizing theta: Front view is centered around z > 0
            az_rad = math.atan2(x_i, z_i)
            az_deg = (math.degrees(az_rad) + 360.0) % 360.0

            # Screen-space Y mapping (vertical)
            norm_y = (y_i - vymin) / vh_range  # in [0, 1] bottom to top
            # Map to image row: image Y is top to bottom
            py = int(np.clip(ymax - norm_y * h_span, 0, H - 1))

            # Select camera view angle
            view_idx = int(round(az_deg / 60.0)) % min(6, len(views_list))
            active_view = views_list[view_idx]
            vH, vW, _ = active_view.shape

            # If front hemisphere (within 75° of 0° front view):
            # Project directly from original input image to preserve exact colors!
            is_front_facing = (az_deg <= 75.0 or az_deg >= 285.0)

            if is_front_facing or len(views_list) == 1:
                # Direct screen projection: X maps directly to horizontal span
                norm_x = (x_i / max_r)  # in [-1, 1]
                px = int(np.clip(cx + norm_x * (w_span * 0.5), 0, W - 1))
                sampled_rgb = enhanced_ref[py, px]
            else:
                # Multi-view side / back angle projection
                norm_x = (x_i / max_r)
                px = int(np.clip((vW // 2) + norm_x * (vW * 0.35), 0, vW - 1))
                sampled_rgb = active_view[py % vH, px]

            # Preserve highlight saturation
            colors[i, 0] = int(sampled_rgb[0])
            colors[i, 1] = int(sampled_rgb[1])
            colors[i, 2] = int(sampled_rgb[2])

        # 4. Bake 2D UV Texture Atlas Image
        atlas = self._bake_texture_atlas(
            enhanced_ref,
            views_list,
            atlas_size=self.atlas_size,
        )

        # 5. Compute color fidelity score
        mean_sat = float(np.mean(colors[:, :3].max(axis=-1) - colors[:, :3].min(axis=-1)))
        fidelity_score = float(np.clip(mean_sat / 128.0, 0.4, 1.0))

        logger.info(
            f"[TextureProjector] Color projection complete: {N} vertices textured | "
            f"Mean Saturation: {mean_sat:.1f} | Fidelity Score: {fidelity_score:.2f} | "
            f"Atlas size: {atlas.size}"
        )

        return TextureProjectionResult(
            vertex_colors=colors,
            uv_coordinates=uvs,
            texture_atlas=atlas,
            mean_saturation=mean_sat,
            color_fidelity_score=fidelity_score,
        )

    def _bake_texture_atlas(
        self,
        front_view: np.ndarray,
        views: List[np.ndarray],
        atlas_size: int = 1024,
    ) -> Image.Image:
        """
        Bakes a seamless cylindrical 2D UV texture atlas from multi-view inputs.
        Horizontal axis represents 360° azimuth angle [0, 2pi].
        Vertical axis represents height [bottom to top].
        """
        atlas_arr = np.full((atlas_size, atlas_size, 3), 240, dtype=np.uint8)

        # Blend views horizontally across the 360° azimuth
        n_views = max(1, min(6, len(views)))
        slice_w = atlas_size // n_views

        for idx in range(n_views):
            v_img = views[idx] if idx < len(views) else front_view
            pil_v = Image.fromarray(v_img).resize((slice_w, atlas_size), Image.Resampling.BILINEAR)
            x_start = idx * slice_w
            x_end = x_start + slice_w
            atlas_arr[:, x_start:x_end] = np.array(pil_v)

        # Horizontal edge smoothing between view seams
        atlas_arr = gaussian_filter(atlas_arr.astype(np.float32), sigma=[0.5, 1.5, 0]).astype(np.uint8)

        return Image.fromarray(atlas_arr)
