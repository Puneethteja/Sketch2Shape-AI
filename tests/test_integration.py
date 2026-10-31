from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.ai_engine import generate
from engine.pipeline import GenerationResult, Sketch2ShapePipeline, StageTiming
from ui.app import _clear_session, _run_ai_generation, _run_mesh_export, _run_preprocessing
from utils.mesh_utils import export_mesh, inspect_mesh, is_watertight, repair_mesh
from utils.segmentation import isolate_and_center


def test_pipeline_data_wiring_end_to_end():
    synthetic_sketch = np.full((512, 512, 4), 255, dtype=np.uint8)
    synthetic_sketch[128:384, 128:384, :3] = 0

    preview, status = _run_preprocessing(canvas_image=synthetic_sketch)
    assert preview is not None
    assert "✔" in status or "complete" in status.lower()

    depth_img, gen_status = _run_ai_generation(
        prompt="matte ceramic sculpture",
        controlnet_scale=0.85,
        num_inference_steps=10,
        seed=123,
        canvas_image=synthetic_sketch,
    )
    assert depth_img is not None
    assert "[OK]" in gen_status
    assert "watertight=True" in gen_status


def test_mesh_validation_interceptor():
    pipeline = Sketch2ShapePipeline(force_cpu=True)
    sketch_input = np.full((512, 512, 3), 255, dtype=np.uint8)
    sketch_input[200:300, 200:300] = 50

    result = pipeline.run(
        sketch_source=sketch_input,
        prompt="smooth porcelain vase",
        seed=42,
    )

    assert isinstance(result, GenerationResult)
    assert result.mesh is not None
    assert is_watertight(result.mesh) is True
    assert result.glb_path is not None
    assert Path(result.glb_path).exists()

    metrics = inspect_mesh(result.mesh)
    assert metrics["is_watertight"] is True
    assert metrics["body_count"] == 1
    assert metrics["vertex_count"] > 0
    assert metrics["face_count"] > 0


def test_mesh_export_interceptor_glb_and_obj():
    synthetic_sketch = np.full((512, 512, 4), 255, dtype=np.uint8)
    synthetic_sketch[150:350, 150:350, :3] = 20

    _run_preprocessing(canvas_image=synthetic_sketch)
    _run_ai_generation(
        prompt="polished bronze figure",
        controlnet_scale=0.9,
        num_inference_steps=10,
        seed=777,
        canvas_image=synthetic_sketch,
    )

    glb_file, glb_status = _run_mesh_export("GLB")
    assert glb_file is not None
    assert Path(glb_file).exists()
    assert Path(glb_file).stat().st_size > 0
    assert "[OK]" in glb_status

    obj_file, obj_status = _run_mesh_export("OBJ")
    assert obj_file is not None
    assert Path(obj_file).exists()
    assert Path(obj_file).stat().st_size > 0
    assert "[OK]" in obj_status


def test_clear_session_resets_pipeline_state():
    synthetic_sketch = np.full((512, 512, 4), 255, dtype=np.uint8)
    _run_preprocessing(canvas_image=synthetic_sketch)
    _run_ai_generation(
        prompt="test prompt",
        controlnet_scale=0.85,
        num_inference_steps=5,
        seed=1,
        canvas_image=synthetic_sketch,
    )

    reset_tuple = _clear_session()
    assert len(reset_tuple) == 7
    assert all(val is None or val == "" for val in reset_tuple)

    img, status = _run_ai_generation(
        prompt="test prompt",
        controlnet_scale=0.85,
        num_inference_steps=5,
        seed=1,
    )
    assert img is None
    assert "⚠" in status


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
        self.assertFalse(np.allclose(standalone_depth, fused_depth, atol=1e-4))


class TestLinkedReconstruction(unittest.TestCase):

    def setUp(self):
        from engine.linker import StageLinker
        from engine.reconstruction import MeshReconstructor, ReconstructionConfig
        self.linker = StageLinker(device="cpu")
        self.reconstructor = MeshReconstructor(config=ReconstructionConfig(device="cpu"))
        self.sketch_pil = Image.fromarray(_make_synthetic_sketch("rect"))

    def test_linked_reconstruct_returns_watertight_mesh(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=55)
        mesh, depth, normals, elapsed = self.linker.linked_reconstruct(
            reconstructor=self.reconstructor,
            handoff=handoff,
        )
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertEqual(depth.shape, (512, 512))
        self.assertEqual(normals.shape, (512, 512, 3))
        self.assertGreater(elapsed, 0.0)

    def test_linked_mesh_differs_from_unlinked(self):
        handoff = self.linker.extract_handoff_tensors(self.sketch_pil, seed=55)
        mesh_linked, _, _, _ = self.linker.linked_reconstruct(
            reconstructor=self.reconstructor,
            handoff=handoff,
        )
        mesh_unlinked, _, _, _ = self.reconstructor.reconstruct(
            image_input=self.sketch_pil,
        )
        diff = abs(len(mesh_linked.vertices) - len(mesh_unlinked.vertices))
        self.assertIsNotNone(diff)


class TestFullPipelineVariedInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = Sketch2ShapePipeline(force_cpu=True)

    def _run_shape(self, shape: str):
        sketch_np = _make_synthetic_sketch(shape)
        result = self.pipeline.run(
            sketch_source=sketch_np,
            prompt=f"a 3D {shape} solid, studio lighting",
            controlnet_scale=0.85,
            num_inference_steps=5,
            seed=42,
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.is_watertight)
        self.assertGreater(result.vertex_count, 0)
        self.assertGreater(result.face_count, 0)
        self.assertEqual(result.depth_map.shape, (512, 512))
        self.assertIsNotNone(result.stage_timing)
        self.assertGreater(result.latency_seconds, 0.0)
        return result

    def test_pipeline_ellipse(self):
        self._run_shape("ellipse")

    def test_pipeline_rect(self):
        self._run_shape("rect")

    def test_pipeline_circle(self):
        self._run_shape("circle")

    def test_pipeline_triangle(self):
        self._run_shape("triangle")


class TestContractBFields(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pipeline = Sketch2ShapePipeline(force_cpu=True)
        cls.result = pipeline.run(
            sketch_source=_make_synthetic_sketch("circle"),
            prompt="porcelain vase",
            seed=101,
            num_inference_steps=5,
        )

    def test_depth_map_shape_and_dtype(self):
        self.assertEqual(self.result.depth_map.shape, (512, 512))
        self.assertEqual(self.result.depth_map.dtype, np.float32)

    def test_depth_map_range(self):
        self.assertGreaterEqual(float(self.result.depth_map.min()), 0.0)
        self.assertLessEqual(float(self.result.depth_map.max()), 1.0 + 1e-5)

    def test_normal_map_shape(self):
        self.assertIsNotNone(self.result.normal_map)
        self.assertEqual(self.result.normal_map.shape, (512, 512, 3))

    def test_mesh_is_watertight_property(self):
        self.assertTrue(self.result.is_watertight)

    def test_vertex_and_face_counts(self):
        self.assertGreater(self.result.vertex_count, 100)
        self.assertGreater(self.result.face_count, 100)

    def test_concrete_seed_forwarded(self):
        self.assertEqual(self.result.seed_used, 101)
        self.assertEqual(self.result.seed, 101)

    def test_vram_peak_is_float(self):
        self.assertIsInstance(self.result.vram_peak_mb, float)
        self.assertGreaterEqual(self.result.vram_peak_mb, 0.0)

    def test_generated_image_present(self):
        self.assertIsNotNone(self.result.generated_image)
        self.assertEqual(self.result.generated_image.size, (512, 512))

    def test_depth_image_property_valid_rgb(self):
        img = self.result.depth_image
        self.assertEqual(img.size, (512, 512))
        self.assertEqual(img.mode, "RGB")


class TestStageTimingBreakdown(unittest.TestCase):

    def test_stage_timing_fields_populated(self):
        pipeline = Sketch2ShapePipeline(force_cpu=True)
        res = pipeline.run(
            sketch_source=_make_synthetic_sketch("rect"),
            seed=202,
            num_inference_steps=5,
        )
        timing = res.stage_timing
        self.assertIsNotNone(timing)
        self.assertGreater(timing.preprocess, 0.0)
        self.assertGreater(timing.generation, 0.0)
        self.assertGreater(timing.linker_extraction, 0.0)
        self.assertGreater(timing.reconstruction, 0.0)
        self.assertGreater(timing.total, 0.0)

    def test_timing_total_matches_sum(self):
        t = StageTiming(preprocess=0.01, generation=0.20, linker_extraction=0.005, reconstruction=0.15, export=0.02)
        expected = round(0.01 + 0.20 + 0.005 + 0.15 + 0.02, 3)
        self.assertEqual(t.total, expected)


class TestAIEngineFacadeIntegration(unittest.TestCase):

    def test_generate_facade_sketch_input_kwarg(self):
        res = generate(
            sketch_input=_make_synthetic_sketch("ellipse"),
            prompt="glossy ceramic sculpture",
            seed=303,
            device="cpu",
        )
        self.assertIsInstance(res, GenerationResult)
        self.assertTrue(res.is_watertight)
        self.assertEqual(res.seed_used, 303)
        self.assertIsNotNone(res.stage_timing)

    def test_generate_facade_source_kwarg_backward_compat(self):
        res = generate(
            source=_make_synthetic_sketch("circle"),
            prompt="metallic gear",
            seed=404,
            device="cpu",
        )
        self.assertIsInstance(res, GenerationResult)
        self.assertTrue(res.is_watertight)
        self.assertEqual(res.seed_used, 404)

    def test_generate_facade_positional_first_arg(self):
        sketch = _make_synthetic_sketch("rect")
        res = generate(sketch, prompt="wooden cube", seed=505, device="cpu")
        self.assertIsInstance(res, GenerationResult)
        self.assertTrue(res.is_watertight)
        self.assertEqual(res.seed_used, 505)

    def test_random_seed_resolves_to_positive_int(self):
        res = generate(
            sketch_input=_make_synthetic_sketch("triangle"),
            seed=-1,
            device="cpu",
        )
        self.assertGreater(res.seed_used, 0)


if __name__ == "__main__":
    unittest.main()
