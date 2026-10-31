"""
tests/test_ai_inference.py
==========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Verification test suite for:
- Phase 1: GPU/VRAM validation, Checkpoint Architecture, Smoke Test
- Phase 2: Sketch-guided Generator, 3D Reconstruction, VRAM Pipeline Orchestrator,
           and Contract B AI Engine integration.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from check_gpu import inspect_system
from engine.ai_engine import GenerationResult, generate
from engine.checkpoints.download_checkpoints import (
    CHECKPOINT_REGISTRY,
    verify_checkpoints,
)
from engine.generator import DiffusionConfig, SketchDiffusionGenerator
from engine.mesh_utils import TrimeshFallback, repair_mesh
from engine.pipeline import Sketch2ShapePipeline
from engine.reconstruction import MeshReconstructor, ReconstructionConfig
from engine.test_pipeline import (
    check_dependencies,
    check_hardware,
    check_huggingface_auth,
    create_synthetic_sketch_image,
    run_smoke_test,
)


class TestGPUAndHardwareValidation(unittest.TestCase):
    """Verifies GPU and system hardware inspection capabilities."""

    def test_inspect_system_returns_valid_report(self):
        report = inspect_system(benchmark=False)
        self.assertIsNotNone(report.os_name)
        self.assertIsNotNone(report.python_version)
        self.assertIsNotNone(report.pytorch_version)
        self.assertIn(report.system_status, ["CPU_ONLY", "CUDA_ACCELERATED"])

    def test_check_hardware_returns_device_info(self):
        hw = check_hardware()
        self.assertIn("cuda_available", hw)
        self.assertIn("selected_device", hw)
        self.assertIn("sdpa_supported", hw)


class TestCheckpointArchitecture(unittest.TestCase):
    """Verifies checkpoint registry structure and directory specifications."""

    def test_checkpoint_registry_contains_all_core_models(self):
        expected_keys = ["sd15", "scribble", "canny", "instantmesh", "zero123plus", "unique3d"]
        for key in expected_keys:
            self.assertIn(key, CHECKPOINT_REGISTRY)
            self.assertIn("repo_id", CHECKPOINT_REGISTRY[key])
            self.assertIn("destination", CHECKPOINT_REGISTRY[key])
            self.assertIn("size_gb", CHECKPOINT_REGISTRY[key])

    def test_verify_checkpoints_runs_without_exceptions(self):
        status = verify_checkpoints()
        self.assertIsInstance(status, dict)
        self.assertEqual(len(status), len(CHECKPOINT_REGISTRY))

    def test_checkpoint_directories_exist(self):
        root = Path(__file__).resolve().parent.parent / "engine" / "checkpoints"
        self.assertTrue((root / "README.md").exists())
        self.assertTrue((root / "stable_diffusion").exists())
        self.assertTrue((root / "controlnet" / "scribble").exists())
        self.assertTrue((root / "controlnet" / "canny").exists())
        self.assertTrue((root / "reconstruction_3d" / "instantmesh").exists())
        self.assertTrue((root / "reconstruction_3d" / "unique3d").exists())


class TestPipelineSmokeTest(unittest.TestCase):
    """Verifies pipeline contracts, dependencies, and synthetic sketch inputs."""

    def test_dependencies_check_structure(self):
        res = check_dependencies()
        self.assertIn("status", res)
        self.assertIn("packages", res)
        self.assertIn("missing", res)
        self.assertIn("torch", res["packages"])

    def test_huggingface_auth_status(self):
        auth = check_huggingface_auth()
        self.assertTrue(auth["public_access_supported"])
        self.assertIn(auth["status"], ["AUTHENTICATED", "ANONYMOUS_PUBLIC_ALLOWED"])

    def test_synthetic_sketch_generation_contract(self):
        sketch = create_synthetic_sketch_image()
        self.assertEqual(sketch.size, (512, 512))
        self.assertEqual(sketch.mode, "RGB")

    def test_run_smoke_test_dry_run_execution(self):
        result = run_smoke_test(device="cpu", dry_run=True)
        self.assertIsInstance(result, dict)
        self.assertIn("stages", result)
        self.assertIn("dependencies", result["stages"])
        self.assertIn("auth", result["stages"])
        self.assertIn("hardware", result["stages"])
        self.assertIn("sketch_contract", result["stages"])
        self.assertTrue(result["stages"]["sketch_contract"]["verified"])


class TestPhase2GenerativeAndReconstruction(unittest.TestCase):
    """Verifies Phase 2 Generative Diffusion, 3D Reconstruction, and Pipeline VRAM Handover."""

    def setUp(self):
        # Create standard test sketch
        self.test_sketch = Image.new("RGB", (512, 512), (255, 255, 255))
        draw = ImageDraw.Draw(self.test_sketch)
        draw.rectangle([180, 160, 330, 360], outline=(0, 0, 0), width=4)
        draw.arc([310, 200, 390, 320], -90, 90, fill=(0, 0, 0), width=4)

    def test_sketch_diffusion_generator_prompt_conditioning(self):
        gen = SketchDiffusionGenerator(DiffusionConfig(device="cpu"))
        pos, neg = gen.build_conditioned_prompt("ceramic coffee mug")
        self.assertIn("ceramic coffee mug", pos)
        self.assertIn("3d asset", pos)
        self.assertIn("blurry", neg)

    def test_sketch_diffusion_generator_execution(self):
        gen = SketchDiffusionGenerator(DiffusionConfig(device="cpu", num_inference_steps=5))
        out_img, seed_used, elapsed = gen.generate(
            sketch_input=self.test_sketch,
            prompt="glossy ceramic mug",
            controlnet_scale=0.85,
            seed=42,
        )
        self.assertEqual(out_img.size, (512, 512))
        self.assertEqual(seed_used, 42)
        self.assertGreater(elapsed, 0.0)

    def test_mesh_reconstructor_predict_depth_and_normals(self):
        recon = MeshReconstructor(ReconstructionConfig(device="cpu"))
        img_np = np.array(self.test_sketch)
        depth, normals = recon.predict_depth_and_normals(img_np)
        self.assertEqual(depth.shape, (512, 512))
        self.assertEqual(normals.shape, (512, 512, 3))
        self.assertGreaterEqual(depth.min(), 0.0)
        self.assertLessEqual(depth.max(), 1.0)

    def test_mesh_reconstructor_watertight_mesh_reconstruction(self):
        recon = MeshReconstructor(ReconstructionConfig(device="cpu"))
        mesh, depth, normal, elapsed = recon.reconstruct(self.test_sketch)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)

    def test_mesh_export_obj_and_glb(self):
        recon = MeshReconstructor(ReconstructionConfig(device="cpu"))
        mesh, _, _, _ = recon.reconstruct(self.test_sketch)
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = Path(tmpdir) / "test_mesh.obj"
            glb_path = Path(tmpdir) / "test_mesh.glb"
            mesh.export(str(obj_path), file_type="obj")
            mesh.export(str(glb_path), file_type="glb")
            self.assertTrue(obj_path.exists())
            self.assertTrue(glb_path.exists())
            self.assertGreater(obj_path.stat().st_size, 0)
            self.assertGreater(glb_path.stat().st_size, 0)

    def test_end_to_end_pipeline_and_contract_b(self):
        pipeline = Sketch2ShapePipeline(force_cpu=True)
        result = pipeline.run(
            sketch_source=self.test_sketch,
            prompt="modern ceramic vase, studio lighting",
            controlnet_scale=0.85,
            num_inference_steps=5,
            seed=777,
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.depth_map.shape, (512, 512))
        self.assertEqual(result.seed_used, 777)
        self.assertTrue(result.mesh.is_watertight)
        self.assertGreater(result.latency_seconds, 0.0)

    def test_ai_engine_generate_facade(self):
        result = generate(
            source=self.test_sketch,
            prompt="ergonomic chair",
            seed=999,
            device="cpu",
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertTrue(result.mesh.is_watertight)
        self.assertEqual(result.seed_used, 999)


if __name__ == "__main__":
    unittest.main()
