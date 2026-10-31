"""
tests/test_usp.py
Unit tests for Sketch2Shape AI Four Unique Selling Propositions (USPs):
1. Sketch Fidelity Score
2. Multi-View Intelligence
3. Component-Level Editing
4. AI Design Variations
"""

import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from engine.usp import (
    compute_sketch_fidelity,
    generate_multiview_projections,
    ComponentEditor,
    generate_design_variations,
    CAMERA_PRESETS,
    COMPONENT_PRESETS,
    MATERIAL_PALETTES,
    DESIGN_STYLE_PRESETS,
)
from engine.mesh_utils import TrimeshFallback


def _build_test_mesh():
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ], dtype=np.int32)
    return TrimeshFallback(vertices=verts, faces=faces, is_watertight=True)


class TestSketchFidelity(unittest.TestCase):
    def setUp(self):
        self.mesh = _build_test_mesh()
        self.sketch = np.ones((512, 512, 3), dtype=np.uint8) * 255
        self.sketch[150:350, 150:350] = 30

    def test_compute_fidelity_returns_valid_report(self):
        report = compute_sketch_fidelity(self.sketch, self.mesh, view="front")
        self.assertGreaterEqual(report.iou_score, 0.0)
        self.assertLessEqual(report.iou_score, 1.0)
        self.assertGreaterEqual(report.percentage, 0.0)
        self.assertLessEqual(report.percentage, 100.0)
        self.assertIn(report.rating, ["Excellent Fidelity", "Good Fidelity", "Moderate Fidelity"])
        self.assertIsInstance(report.explanation, str)
        self.assertIsInstance(report.comparison_image, Image.Image)
        self.assertEqual(report.comparison_image.size[0], 512)

    def test_compute_fidelity_with_pil_input(self):
        pil_img = Image.fromarray(self.sketch)
        report = compute_sketch_fidelity(pil_img, self.mesh)
        self.assertGreater(report.percentage, 0.0)


class TestMultiViewIntelligence(unittest.TestCase):
    def setUp(self):
        self.mesh = _build_test_mesh()

    def test_generate_multiview_composite(self):
        composite = generate_multiview_projections(self.mesh, tile_size=128)
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.size, (128 * 2 + 12, 128 * 2 + 12))

    def test_camera_presets_structure(self):
        self.assertIn("Front", CAMERA_PRESETS)
        self.assertIn("Back", CAMERA_PRESETS)
        self.assertIn("Right Side", CAMERA_PRESETS)
        self.assertIn("Top", CAMERA_PRESETS)
        self.assertIn("Isometric 360", CAMERA_PRESETS)
        for name, preset in CAMERA_PRESETS.items():
            self.assertIn("pos", preset)
            self.assertIn("target", preset)
            self.assertEqual(len(preset["pos"]), 3)


class TestComponentLevelEditing(unittest.TestCase):
    def setUp(self):
        self.mesh = _build_test_mesh()

    def test_presets_availability(self):
        comps = ComponentEditor.get_components()
        self.assertGreaterEqual(len(comps), 3)
        self.assertIn("Base / Stand / Legs", comps)

        mats = ComponentEditor.get_materials()
        self.assertIn("Polished Walnut Wood", mats)
        self.assertIn("Brushed Titanium", mats)

    def test_apply_component_edit(self):
        edited = ComponentEditor.apply_edit(
            self.mesh,
            "Base / Stand / Legs",
            "Polished Walnut Wood",
            custom_color="#784b2d",
        )
        self.assertIsNotNone(edited)
        self.assertTrue(getattr(edited, "is_watertight", False))
        self.assertEqual(len(edited.vertices), len(self.mesh.vertices))


class TestAIDesignVariations(unittest.TestCase):
    def setUp(self):
        self.sketch = np.ones((512, 512, 3), dtype=np.uint8) * 255
        self.sketch[180:320, 180:320] = 40

    def test_generate_variations(self):
        styles = ["Minimalist Porcelain", "Brushed Titanium Metal"]
        vars_out = generate_design_variations(
            self.sketch,
            user_prompt="modern desk accessory",
            selected_styles=styles,
        )
        self.assertEqual(len(vars_out), 2)
        for style_name, img, prompt in vars_out:
            self.assertIn(style_name, styles)
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.size, (512, 512))
            self.assertIn("modern desk accessory", prompt)


if __name__ == "__main__":
    unittest.main()
