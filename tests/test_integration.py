from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from engine.ai_engine import generate
from engine.pipeline import GenerationResult, Sketch2ShapePipeline
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
