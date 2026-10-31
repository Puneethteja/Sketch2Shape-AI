from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import trimesh
from PIL import Image

from utils.mesh_utils import (
    create_fallback_mesh,
    export_mesh,
    inspect_mesh,
    is_watertight,
    load_mesh,
    repair_mesh,
    sanitize_mesh,
    strip_isolated_components,
)
from utils.preprocessing import preprocess, to_lineart
from utils.segmentation import isolate_and_center


class TestPreprocessingStress:
    def test_preprocess_noisy_gaussian_sketch(self) -> None:
        clean_sketch = np.full((512, 512, 3), 255, dtype=np.uint8)
        clean_sketch[150:350, 250:260] = 0
        noise = np.random.normal(0, 35, clean_sketch.shape).astype(np.int16)
        noisy_sketch = np.clip(clean_sketch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        processed_np = preprocess(noisy_sketch, return_tensor=False)
        assert isinstance(processed_np, np.ndarray)
        assert processed_np.shape == (512, 512, 3)
        assert processed_np.dtype == np.uint8

        tensor_out = preprocess(noisy_sketch, return_tensor=True)
        assert tuple(tensor_out.shape) in ((1, 3, 512, 512), (3, 512, 512))
        assert float(tensor_out.min()) >= 0.0
        assert float(tensor_out.max()) <= 1.0

    def test_preprocess_salt_and_pepper_noise(self) -> None:
        base = np.full((256, 256, 3), 240, dtype=np.uint8)
        base[50:200, 50:200] = 20
        rnd = np.random.rand(256, 256)
        base[rnd < 0.08] = 0
        base[rnd > 0.92] = 255

        processed = preprocess(base, return_tensor=False)
        assert processed.shape == (512, 512, 3)
        lineart = to_lineart(processed)
        assert lineart.shape == (512, 512)
        assert set(np.unique(lineart)).issubset({0, 255})

    def test_preprocess_low_resolution_inputs(self) -> None:
        resolutions = [(16, 16), (32, 32), (64, 48), (128, 96)]
        for h, w in resolutions:
            low_res = np.full((h, w, 3), 255, dtype=np.uint8)
            low_res[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 0

            out = preprocess(low_res, return_tensor=False)
            assert out.shape == (512, 512, 3)
            assert out.dtype == np.uint8

    def test_preprocess_high_contrast_and_inverted_inputs(self) -> None:
        inverted = np.zeros((300, 300, 3), dtype=np.uint8)
        inverted[100:200, 100:200] = 255

        out_normal = preprocess(inverted, return_tensor=False)
        assert out_normal.shape == (512, 512, 3)

        high_contrast = np.where(inverted > 128, 255, 0).astype(np.uint8)
        lineart = to_lineart(high_contrast, invert=True)
        assert lineart.shape == (300, 300)

    def test_preprocess_incomplete_faint_sketch(self) -> None:
        faint = np.full((400, 400, 3), 255, dtype=np.uint8)
        faint[100:300, 195:205] = 220

        out = preprocess(faint, return_tensor=False)
        assert out.shape == (512, 512, 3)
        assert np.isfinite(out).all()

    def test_preprocess_extreme_aspect_ratios(self) -> None:
        wide_banner = np.full((64, 800, 3), 255, dtype=np.uint8)
        wide_banner[:, 380:420] = 10
        out_wide = preprocess(wide_banner, return_tensor=False)
        assert out_wide.shape == (512, 512, 3)

        tall_strip = np.full((800, 64, 3), 255, dtype=np.uint8)
        tall_strip[380:420, :] = 10
        out_tall = preprocess(tall_strip, return_tensor=False)
        assert out_tall.shape == (512, 512, 3)

    def test_segmentation_stress_edge_cases(self) -> None:
        blank_white = np.full((256, 256, 3), 255, dtype=np.uint8)
        centered_blank = isolate_and_center(blank_white)
        assert isinstance(centered_blank, Image.Image)
        assert centered_blank.size == (512, 512)

        tiny_dot = np.full((256, 256, 3), 255, dtype=np.uint8)
        tiny_dot[128, 128] = 0
        centered_dot = isolate_and_center(tiny_dot)
        assert isinstance(centered_dot, Image.Image)
        assert centered_dot.size == (512, 512)


class TestMeshResilienceStress:
    def test_mesh_repair_non_watertight_open_annulus(self) -> None:
        open_annulus = trimesh.creation.annulus(r_min=0.3, r_max=1.0, height=0.0)
        assert is_watertight(open_annulus) is False

        repaired = repair_mesh(open_annulus, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert is_watertight(repaired) is True
        assert len(repaired.faces) > 0
        assert len(repaired.vertices) >= 4

    def test_mesh_repair_open_cylinder_without_caps(self) -> None:
        cylinder = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=16)
        open_cyl = trimesh.Trimesh(
            vertices=cylinder.vertices,
            faces=cylinder.faces[:-4],
            process=False,
        )
        assert is_watertight(open_cyl) is False

        repaired = repair_mesh(open_cyl, fill_holes=True, enforce_watertight=True)
        assert is_watertight(repaired) is True
        assert repaired.volume > 0.0

    def test_mesh_repair_self_intersecting_crossed_geometry(self) -> None:
        v_cross = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, -0.6],
            [0.5, 0.5, 0.6],
        ], dtype=np.float64)
        f_cross = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
            [0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 0, 5],
        ], dtype=np.int64)

        cross_mesh = trimesh.Trimesh(vertices=v_cross, faces=f_cross, process=False)
        repaired = repair_mesh(cross_mesh, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert is_watertight(repaired) is True
        assert len(repaired.faces) >= 4

    def test_mesh_repair_degenerate_triangles_and_colinear_faces(self) -> None:
        v_deg = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        f_deg = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [0, 6, 7],
            [0, 6, 7],
        ], dtype=np.int64)

        deg_mesh = trimesh.Trimesh(vertices=v_deg, faces=f_deg, process=False)
        repaired = repair_mesh(deg_mesh, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert is_watertight(repaired) is True
        assert len(repaired.vertices) >= 4

    def test_mesh_repair_unattached_floating_fragments(self) -> None:
        main_sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        fragments = [
            trimesh.creation.box(extents=(0.02, 0.02, 0.02)).apply_translation([5.0 + i, 0.0, 0.0])
            for i in range(30)
        ]
        composite = trimesh.util.concatenate([main_sphere] + fragments)

        metrics_before = inspect_mesh(composite)
        assert metrics_before["body_count"] > 1

        cleaned = strip_isolated_components(composite, min_face_ratio=0.05)
        metrics_after = inspect_mesh(cleaned)
        assert metrics_after["body_count"] == 1

        repaired = repair_mesh(composite, strip_floating=True, enforce_watertight=True)
        assert is_watertight(repaired) is True
        assert len(repaired.faces) < len(composite.faces)

    def test_mesh_repair_nan_and_inf_vertex_coordinates(self) -> None:
        v_corrupt = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [np.nan, 0.0, 0.0],
            [np.inf, np.inf, 0.0],
            [-np.inf, 0.0, 1.0],
        ], dtype=np.float64)
        f_corrupt = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 5, 6],
        ], dtype=np.int64)

        corrupt_mesh = trimesh.Trimesh(vertices=v_corrupt, faces=f_corrupt, process=False)
        repaired = repair_mesh(corrupt_mesh, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert np.isfinite(repaired.vertices).all()
        assert is_watertight(repaired) is True

    def test_mesh_repair_completely_empty_mesh(self) -> None:
        empty_mesh = trimesh.Trimesh()
        repaired = repair_mesh(empty_mesh, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert len(repaired.faces) > 0
        assert is_watertight(repaired) is True

        none_repaired = repair_mesh(None, enforce_watertight=True)
        assert isinstance(none_repaired, trimesh.Trimesh)
        assert is_watertight(none_repaired) is True

    def test_mesh_repair_single_triangle_surface(self) -> None:
        v_tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        f_tri = np.array([[0, 1, 2]], dtype=np.int64)
        single_tri = trimesh.Trimesh(vertices=v_tri, faces=f_tri, process=False)

        repaired = repair_mesh(single_tri, enforce_watertight=True)
        assert isinstance(repaired, trimesh.Trimesh)
        assert is_watertight(repaired) is True
        assert repaired.volume > 0.0

    def test_mesh_export_corrupt_mesh_to_glb_and_obj(self, tmp_path: Path) -> None:
        empty_mesh = trimesh.Trimesh()

        glb_path = tmp_path / "corrupt_fallback.glb"
        exported_glb = export_mesh(empty_mesh, glb_path)
        assert exported_glb.exists()
        assert exported_glb.stat().st_size > 0

        obj_path = tmp_path / "corrupt_fallback.obj"
        exported_obj = export_mesh(empty_mesh, obj_path)
        assert exported_obj.exists()
        assert exported_obj.stat().st_size > 0

        reloaded = load_mesh(exported_glb)
        assert len(reloaded.faces) > 0
        assert len(reloaded.vertices) > 0

    def test_inspect_mesh_on_malformed_geometry(self) -> None:
        empty_metrics = inspect_mesh(trimesh.Trimesh())
        assert isinstance(empty_metrics, dict)
        assert empty_metrics["vertex_count"] == 0
        assert empty_metrics["face_count"] == 0
        assert empty_metrics["is_watertight"] is False
        assert empty_metrics["volume"] == 0.0

        none_metrics = inspect_mesh(None)
        assert isinstance(none_metrics, dict)
        assert none_metrics["vertex_count"] == 0
