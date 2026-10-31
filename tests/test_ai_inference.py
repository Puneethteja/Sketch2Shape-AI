"""
tests/test_ai_inference.py
==========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Verification test suite for:
- GPU & VRAM inspection (check_gpu.py)
- Model checkpoint registry & directories (engine/checkpoints/)
- Pipeline smoke test contracts (engine/test_pipeline.py)
"""

import unittest
from pathlib import Path

from check_gpu import inspect_system
from engine.checkpoints.download_checkpoints import (
    CHECKPOINT_REGISTRY,
    verify_checkpoints,
)
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


if __name__ == "__main__":
    unittest.main()
