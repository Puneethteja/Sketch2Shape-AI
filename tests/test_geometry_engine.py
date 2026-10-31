"""
tests/test_geometry_engine.py
==============================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Unit & validation test suite for the extensible geometry generation architecture
(engine/geometry_engine.py), focusing on:
  - Rotational lathe geometry with hollow interior, rim, and wall thickness
  - Bounded proportions matching 2D sketch silhouettes (bulbous body, narrow neck, flared opening)
  - Watertight 2-manifold topological guarantees
  - Multi-category dispatch (vases, cups, bowls, furniture, organic)
  - GLB and OBJ binary export validation
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from engine.geometry_engine import (
    GeometryCategory,
    RotationalVesselEngine,
    HollowVesselEngine,
    ComponentGeometryEngine,
    UniversalGeometryEngine,
    classify_geometry,
)
from engine.mesh_utils import TrimeshFallback


class TestGeometryClassification(unittest.TestCase):
    """Verifies semantic prompt and visual silhouette classification."""

    def test_prompt_semantic_routing(self):
        dummy_img = Image.new("L", (100, 100), 255)
        self.assertEqual(classify_geometry(dummy_img, "ceramic vase"), GeometryCategory.ROTATIONAL_VESSEL)
        self.assertEqual(classify_geometry(dummy_img, "glass water bottle"), GeometryCategory.ROTATIONAL_VESSEL)
        self.assertEqual(classify_geometry(dummy_img, "terracotta flower pot"), GeometryCategory.ROTATIONAL_VESSEL)
        self.assertEqual(classify_geometry(dummy_img, "coffee mug"), GeometryCategory.HOLLOW_VESSEL)
        self.assertEqual(classify_geometry(dummy_img, "wooden bowl"), GeometryCategory.HOLLOW_VESSEL)
        self.assertEqual(classify_geometry(dummy_img, "modern dining chair"), GeometryCategory.COMPONENT_BASED)
        self.assertEqual(classify_geometry(dummy_img, "wooden table"), GeometryCategory.COMPONENT_BASED)
        self.assertEqual(classify_geometry(dummy_img, "sports car"), GeometryCategory.ORGANIC_BILATERAL)

    def test_visual_symmetry_routing_without_prompt(self):
        vase_path = Path("outputs/samples/sample_01_vase/input_sketch.png")
        if vase_path.exists():
            img = Image.open(vase_path)
            # Empty prompt triggers visual silhouette analysis
            cat = classify_geometry(img, prompt="")
            self.assertEqual(cat, GeometryCategory.ROTATIONAL_VESSEL)


class TestRotationalVesselEngine(unittest.TestCase):
    """Verifies lathe geometry reconstruction on actual hand-drawn vase sketches."""

    @classmethod
    def setUpClass(cls):
        cls.vase_path = Path("outputs/samples/sample_01_vase/input_sketch.png")
        if cls.vase_path.exists():
            cls.vase_img = Image.open(cls.vase_path).convert("RGB")
        else:
            # Synthetic vase sketch fallback
            cls.vase_img = Image.new("RGB", (512, 512), (255, 255, 255))

    def test_profile_extraction_proportions(self):
        r_outer, z_outer, x_axis, aspect = RotationalVesselEngine.extract_profile_from_image(
            self.vase_img,
            num_vertical_samples=48,
        )
        self.assertEqual(len(r_outer), 48)
        self.assertEqual(len(z_outer), 48)

        # Base to top heights: z_outer should be strictly increasing
        self.assertTrue(np.all(np.diff(z_outer) > 0))

        # Check proportions: bulbous body vs narrow neck
        # Body is in lower-middle section (around index 12-20 out of 48)
        body_r = np.max(r_outer[8:24])
        # Neck is in upper section (around index 32-40 out of 48)
        neck_r = np.min(r_outer[28:42])
        # Rim is at the very top (index 47)
        rim_r = r_outer[-1]

        # Bulbous body must be significantly wider than narrow neck (at least 1.8x)
        self.assertGreater(body_r, 1.8 * neck_r)

        # Flared opening at top: rim radius must be wider than the narrow neck
        self.assertGreater(rim_r, neck_r)

    def test_hollow_mesh_topology_and_watertightness(self):
        mesh = RotationalVesselEngine.generate_mesh(
            self.vase_img,
            num_vertical=48,
            num_radial=48,
            wall_thickness_ratio=0.08,
            base_floor_ratio=0.08,
            prompt="ceramic vase",
        )
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(mesh.vertices), 1000)
        self.assertGreater(len(mesh.faces), 2000)

        # Check edge sharing: every edge in a watertight 2-manifold must belong to exactly 2 faces
        from collections import Counter
        faces = np.asarray(mesh.faces, dtype=np.int32)
        edges = []
        for f in faces:
            edges.append(tuple(sorted([f[0], f[1]])))
            edges.append(tuple(sorted([f[1], f[2]])))
            edges.append(tuple(sorted([f[2], f[0]])))
        edge_counts = Counter(edges)
        non_two = [e for e, c in edge_counts.items() if c != 2]
        self.assertEqual(len(non_two), 0, f"Found {len(non_two)} non-manifold or boundary edges!")

        # Euler characteristic: V - E + F == 2 (sphere topology)
        euler = len(mesh.vertices) - len(edge_counts) + len(mesh.faces)
        self.assertEqual(euler, 2)

    def test_hollow_interior_and_wall_thickness(self):
        mesh = RotationalVesselEngine.generate_mesh(
            self.vase_img,
            num_vertical=48,
            num_radial=48,
            wall_thickness_ratio=0.08,
            base_floor_ratio=0.08,
        )
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        y_max = verts[:, 1].max()
        y_min = verts[:, 1].min()

        # At the top rim level (y near y_max)
        top_verts = verts[verts[:, 1] > (y_max - 0.05)]
        top_radii = np.sqrt(top_verts[:, 0] ** 2 + top_verts[:, 2] ** 2)

        outer_rim_r = float(top_radii.max())
        inner_rim_r = float(top_radii.min())
        wall_t = outer_rim_r - inner_rim_r

        # Must have realistic wall thickness
        self.assertGreater(wall_t, 0.015)
        # Inner opening must be clearly hollow (inner cavity diameter > 0.15)
        self.assertGreater(inner_rim_r * 2.0, 0.15)

    def test_mesh_export_obj_and_glb(self):
        mesh = RotationalVesselEngine.generate_mesh(self.vase_img, prompt="porcelain vase")
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "vase.obj"
            glb_path = Path(tmpdir) / "vase.glb"
            mesh.export(str(obj_path), file_type="obj")
            mesh.export(str(glb_path), file_type="glb")

            self.assertTrue(obj_path.exists())
            self.assertTrue(glb_path.exists())
            self.assertGreater(obj_path.stat().st_size, 1000)
            self.assertGreater(glb_path.stat().st_size, 1000)


class TestUniversalDispatcher(unittest.TestCase):
    """Verifies universal routing across diverse categories."""

    def test_universal_dispatch_categories(self):
        dummy_img = Image.new("RGB", (256, 256), (255, 255, 255))

        # 1. Rotational
        v_mesh = UniversalGeometryEngine.generate(dummy_img, prompt="ceramic vase")
        self.assertTrue(v_mesh.is_watertight)
        self.assertGreater(len(v_mesh.vertices), 500)

        # 2. Hollow bowl
        b_mesh = UniversalGeometryEngine.generate(dummy_img, prompt="wooden salad bowl")
        self.assertTrue(b_mesh.is_watertight)

        # 3. Component chair
        c_mesh = UniversalGeometryEngine.generate(dummy_img, prompt="office chair")
        self.assertTrue(c_mesh.is_watertight)


if __name__ == "__main__":
    unittest.main()
