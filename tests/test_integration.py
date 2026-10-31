"""
tests/test_integration.py
=========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Phase 3 Integration Test Suite  ·  Mona (Generative AI & 3D Reconstruction)

End-to-end integration tests validating the full Stage 2->3 tensor-linked pipeline:
  - StageLinker tensor extraction contracts
  - Depth prior fusion correctness
  - Linked reconstruct mesh geometry contracts
  - Full pipeline run with synthetic sketch inputs (4 shape variants)
  - GenerationResult field contracts (Contract B SS4.2)
  - Stage timing breakdown presence and validity
  - Handoff metadata propagation
  - ai_engine.generate() facade integration
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_synthetic_sketch(variant: str = "ellipse", size: int = 512) -> np.ndarray:
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    cx, cy = size // 2, size // 2

    if variant == "ellipse":
        Y, X = np.ogrid[:size, :size]
        mask = ((X - cx) / (size * 0.30)) ** 2 + ((Y - cy) / (size * 0.22)) ** 2 <= 1.0
        canvas[mask] = [60, 80, 100]

    elif variant == "rect":
        m = size // 4
        canvas[m: size - m, m: size - m] = [120, 80, 50]

    elif variant == "circle":
        Y, X = np.ogrid[:size, :size]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (size * 0.28) ** 2
        canvas[mask] = [200, 80, 60]

    elif variant == "triangle":
        from PIL import ImageDraw
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        pts = [(cx, cy - size // 3), (cx - size // 3, cy + size // 3), (cx + size // 3, cy + size // 3)]
        draw.polygon(pts, fill=(50, 150, 200))
        canvas = np.array(img)

    return canvas


class TestStageLinkTensorExtraction(unittest.TestCase):

    def setUp(self):
        from engine.linker import StageLinker
        self.linker = StageLinker(device="cpu")
        self.sketch_np = _make_synthetic_sketch("ellipse")
        self.sketch_pil = Image.fromarray(self.sketch_np)

    def test_handoff_has_all_required_fields(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=42)
        self.assertIsNotNone(handoff.generated_image)
        self.assertIsNotNone(handoff.image_np)
        self.assertIsNotNone(handoff.depth_prior)
        self.assertIsNotNone(handoff.shading_map)
        self.assertIsNotNone(handoff.luminance)
        self.assertIsNotNone(handoff.foreground_mask)

    def test_depth_prior_range_is_normalised(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=7)
        prior = handoff.depth_prior
        self.assertGreaterEqual(float(prior.min()), 0.0)
        self.assertLessEqual(float(prior.max()), 1.0 + 1e-5)

    def test_shading_map_range_is_normalised(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=7)
        shading = handoff.shading_map
        self.assertGreaterEqual(float(shading.min()), 0.0)
        self.assertLessEqual(float(shading.max()), 1.0 + 1e-5)

    def test_foreground_mask_is_boolean_array(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=9)
        mask = handoff.foreground_mask
        self.assertEqual(mask.dtype, bool)
        self.assertGreater(int(mask.sum()), 100)

    def test_luminance_matches_image_shape(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        self.assertEqual(handoff.luminance.shape, (512, 512))

    def test_handoff_metadata_contains_expected_keys(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        for key in ("fg_pixel_count", "depth_min", "depth_max", "shading_mean"):
            self.assertIn(key, handoff.metadata)

    def test_seed_forwarded_in_handoff(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=1234)
        self.assertEqual(handoff.seed, 1234)

    def test_prompt_forwarded_in_handoff(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, prompt="ceramic vase")
        self.assertEqual(handoff.prompt, "ceramic vase")


class TestDepthFusion(unittest.TestCase):

    def setUp(self):
        from engine.linker import StageLinker
        from engine.reconstruction import MeshReconstructor, ReconstructionConfig
        self.linker = StageLinker(device="cpu")
        self.reconstructor = MeshReconstructor(config=ReconstructionConfig(device="cpu"))
        self.sketch_pil = Image.fromarray(_make_synthetic_sketch("circle"))

    def test_inject_depth_prior_returns_correct_shapes(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        fused_depth, normal_map = self.linker.inject_depth_prior(self.reconstructor, handoff)
        self.assertEqual(fused_depth.shape, (512, 512))
        self.assertEqual(normal_map.shape, (512, 512, 3))

    def test_fused_depth_range_is_valid(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        fused_depth, _ = self.linker.inject_depth_prior(self.reconstructor, handoff)
        self.assertGreaterEqual(float(fused_depth.min()), 0.0)
        self.assertLessEqual(float(fused_depth.max()), 1.0 + 1e-5)

    def test_fused_depth_differs_from_standalone(self):
        img_np = np.array(self.sketch_pil, dtype=np.uint8)
        standalone_depth, _ = self.reconstructor.predict_depth_and_normals(img_np)
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        fused_depth, _ = self.linker.inject_depth_prior(self.reconstructor, handoff)
        max_diff = float(np.abs(fused_depth - standalone_depth).max())
        self.assertGreater(max_diff, 1e-4)

    def test_normal_map_range_is_valid(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil)
        _, normal_map = self.linker.inject_depth_prior(self.reconstructor, handoff)
        self.assertGreaterEqual(float(normal_map.min()), 0.0)
        self.assertLessEqual(float(normal_map.max()), 1.0 + 1e-5)


class TestLinkedReconstruct(unittest.TestCase):

    def setUp(self):
        from engine.linker import StageLinker
        from engine.reconstruction import MeshReconstructor, ReconstructionConfig
        self.linker = StageLinker(device="cpu")
        self.reconstructor = MeshReconstructor(config=ReconstructionConfig(device="cpu"))

    def _run_linked(self, variant: str):
        sketch_pil = Image.fromarray(_make_synthetic_sketch(variant))
        handoff = self.linker.extract_handoff_tensors(sketch_pil, seed=42)
        return self.linker.linked_reconstruct(self.reconstructor, handoff)

    def test_linked_reconstruct_returns_watertight_mesh(self):
        mesh, depth, normals, elapsed = self._run_linked("ellipse")
        self.assertTrue(mesh.is_watertight)

    def test_linked_reconstruct_mesh_has_vertices_and_faces(self):
        mesh, _, _, _ = self._run_linked("rect")
        self.assertGreater(len(mesh.vertices), 10)
        self.assertGreater(len(mesh.faces), 4)

    def test_linked_reconstruct_elapsed_is_positive(self):
        _, _, _, elapsed = self._run_linked("circle")
        self.assertGreater(elapsed, 0.0)

    def test_linked_reconstruct_depth_shape(self):
        _, depth, _, _ = self._run_linked("ellipse")
        self.assertEqual(depth.shape, (512, 512))

    def test_linked_reconstruct_normals_shape(self):
        _, _, normals, _ = self._run_linked("ellipse")
        self.assertEqual(normals.shape, (512, 512, 3))


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        from engine.pipeline import Sketch2ShapePipeline
        self.pipeline = Sketch2ShapePipeline(force_cpu=True)

    def _run(self, variant: str, prompt: str = "", seed: int = 42):
        sketch_np = _make_synthetic_sketch(variant)
        return self.pipeline.run(
            sketch_source=sketch_np,
            prompt=prompt,
            seed=seed,
        )

    def test_e2e_ellipse_sketch_produces_watertight_mesh(self):
        result = self._run("ellipse", prompt="smooth ceramic bowl")
        self.assertTrue(result.is_watertight)

    def test_e2e_rect_sketch_produces_valid_mesh(self):
        result = self._run("rect", prompt="wooden box")
        self.assertGreater(result.vertex_count, 10)
        self.assertGreater(result.face_count, 4)

    def test_e2e_circle_sketch_produces_valid_mesh(self):
        result = self._run("circle", prompt="red sphere")
        self.assertGreater(result.vertex_count, 10)

    def test_e2e_triangle_sketch_produces_valid_mesh(self):
        result = self._run("triangle", prompt="blue prism")
        self.assertGreater(result.face_count, 4)

    def test_e2e_result_depth_map_shape(self):
        result = self._run("ellipse")
        self.assertEqual(result.depth_map.shape, (512, 512))

    def test_e2e_result_normal_map_shape(self):
        result = self._run("ellipse")
        self.assertEqual(result.normal_map.shape, (512, 512, 3))

    def test_e2e_result_depth_map_range(self):
        result = self._run("circle")
        self.assertGreaterEqual(float(result.depth_map.min()), 0.0)
        self.assertLessEqual(float(result.depth_map.max()), 1.0 + 1e-5)

    def test_e2e_result_seed_used_is_positive(self):
        result = self._run("ellipse", seed=99)
        self.assertEqual(result.seed_used, 99)

    def test_e2e_result_seed_random_when_minus_one(self):
        result = self._run("ellipse", seed=-1)
        self.assertGreater(result.seed_used, 0)

    def test_e2e_result_latency_is_positive(self):
        result = self._run("ellipse")
        self.assertGreater(result.latency_seconds, 0.0)

    def test_e2e_result_generated_image_is_pil(self):
        result = self._run("rect")
        self.assertIsInstance(result.generated_image, Image.Image)

    def test_e2e_result_depth_image_property_is_pil(self):
        result = self._run("ellipse")
        depth_img = result.depth_image
        self.assertIsInstance(depth_img, Image.Image)
        self.assertEqual(depth_img.size, (512, 512))


class TestStageTiming(unittest.TestCase):

    def setUp(self):
        from engine.pipeline import Sketch2ShapePipeline
        self.pipeline = Sketch2ShapePipeline(force_cpu=True)

    def test_stage_timing_present_in_result(self):
        sketch_np = _make_synthetic_sketch("ellipse")
        result = self.pipeline.run(sketch_source=sketch_np, seed=1)
        self.assertIsNotNone(result.stage_timing)

    def test_stage_timing_all_components_positive(self):
        sketch_np = _make_synthetic_sketch("circle")
        result = self.pipeline.run(sketch_source=sketch_np, seed=2)
        t = result.stage_timing
        self.assertGreater(t.preprocess, 0.0)
        self.assertGreater(t.generation, 0.0)
        self.assertGreater(t.linker_extraction, 0.0)
        self.assertGreater(t.reconstruction, 0.0)

    def test_stage_timing_total_matches_latency_approx(self):
        sketch_np = _make_synthetic_sketch("rect")
        result = self.pipeline.run(sketch_source=sketch_np, seed=3)
        t = result.stage_timing
        self.assertAlmostEqual(result.latency_seconds, t.total, delta=0.5)


class TestHandoffMetadataPropagation(unittest.TestCase):

    def setUp(self):
        from engine.pipeline import Sketch2ShapePipeline
        self.pipeline = Sketch2ShapePipeline(force_cpu=True)

    def test_handoff_metadata_propagated_to_result(self):
        sketch_np = _make_synthetic_sketch("ellipse")
        result = self.pipeline.run(sketch_source=sketch_np, seed=42)
        self.assertIsNotNone(result.handoff_metadata)
        self.assertIn("fg_pixel_count", result.handoff_metadata)
        self.assertIn("depth_max", result.handoff_metadata)

    def test_handoff_metadata_fg_pixel_count_positive(self):
        sketch_np = _make_synthetic_sketch("circle")
        result = self.pipeline.run(sketch_source=sketch_np, seed=5)
        self.assertGreater(result.handoff_metadata["fg_pixel_count"], 100)


class TestAIEngineFacadePhase3(unittest.TestCase):

    def test_ai_engine_generate_returns_generation_result(self):
        import engine.ai_engine as ai
        from engine.pipeline import GenerationResult
        sketch_np = _make_synthetic_sketch("ellipse")
        result = ai.generate(sketch_input=sketch_np, prompt="ceramic sphere", seed=42)
        self.assertIsInstance(result, GenerationResult)

    def test_ai_engine_generate_exposes_stage_timing(self):
        import engine.ai_engine as ai
        sketch_np = _make_synthetic_sketch("rect")
        result = ai.generate(sketch_input=sketch_np, seed=7)
        self.assertIsNotNone(result.stage_timing)
        self.assertGreater(result.stage_timing.linker_extraction, 0.0)

    def test_ai_engine_generate_mesh_is_watertight(self):
        import engine.ai_engine as ai
        sketch_np = _make_synthetic_sketch("circle")
        result = ai.generate(sketch_input=sketch_np, seed=99)
        self.assertTrue(result.is_watertight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
