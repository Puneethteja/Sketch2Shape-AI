from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_gradio_importable():
    import gradio as gr
    assert gr.__version__

def test_build_ui_returns_blocks():
    import gradio as gr
    from ui.app import build_ui

    demo = build_ui()
    assert isinstance(demo, gr.Blocks)

def test_ui_contains_expected_components():
    import gradio as gr
    from ui.app import build_ui

    demo = build_ui()
    component_types = {type(c).__name__ for c in demo.blocks.values()}
    required = {"Sketchpad", "File", "Textbox", "Image", "Button", "Slider"}
    missing = required - component_types
    assert not missing, f"Missing UI components: {missing}"

def test_preprocess_no_input_returns_warning():
    from ui.app import _run_preprocessing

    preview, status = _run_preprocessing(None, None)
    assert preview is None
    assert "No input" in status or "⚠" in status

def test_preprocess_with_synthetic_canvas():
    from ui.app import _run_preprocessing, _PREPROCESSING_AVAILABLE

    canvas = np.full((512, 512, 4), 255, dtype=np.uint8)
    preview, status = _run_preprocessing(canvas, None)

    if _PREPROCESSING_AVAILABLE:
        from PIL import Image
        assert isinstance(preview, Image.Image)
        assert "✔" in status or "complete" in status.lower()
    else:
        assert preview is None
        assert "unavailable" in status.lower() or "⚠" in status

def test_ai_generation_stub_returns_gracefully():
    from ui.app import _run_ai_generation

    result = _run_ai_generation(
        prompt="a ceramic vase",
        controlnet_scale=1.0,
        num_inference_steps=30,
        seed=42,
    )
    assert isinstance(result, tuple)
    assert len(result) == 4
    depth_img, glb_file, status, glb_url = result
    assert depth_img is None
    assert isinstance(status, str) and len(status) > 0

@pytest.mark.parametrize("fmt", ["OBJ", "GLB", "STL", "PLY"])
def test_mesh_export_stub_returns_gracefully(fmt):
    from ui.app import _run_mesh_export

    file_path, status = _run_mesh_export(fmt)
    assert isinstance(status, str) and len(status) > 0

def test_clear_session_returns_seven_tuple():
    from ui.app import _clear_session

    result = _clear_session()
    assert isinstance(result, tuple)
    assert len(result) == 7

    canvas, file_up, prompt, pre_view, depth_view, export_file, status = result
    assert canvas is None
    assert file_up is None
    assert prompt == ""
    assert pre_view is None
    assert depth_view is None
    assert export_file is None
    assert status == ""

def test_webgl_stub_html_is_valid():
    from ui.app import _WEBGL_STUB_HTML

    assert isinstance(_WEBGL_STUB_HTML, str)
    assert len(_WEBGL_STUB_HTML) > 100
    assert "<div" in _WEBGL_STUB_HTML
    assert "WebGL" in _WEBGL_STUB_HTML

def test_supported_upload_extensions():
    from ui.app import SUPPORTED_UPLOAD_EXTENSIONS

    for ext in [".png", ".jpg", ".jpeg"]:
        assert ext in SUPPORTED_UPLOAD_EXTENSIONS

def test_viewer_js_capabilities():
    viewer_path = ROOT / "ui" / "viewer.js"
    assert viewer_path.exists()

    content = viewer_path.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "Sketch2ShapeViewer" in content
    assert "OrbitControls" in content
    assert "downloadBinary" in content
    assert "spinner" in content.lower()
    assert "loadGLB" in content

def test_html5_canvas_markup_and_controls():
    from ui.app import _HTML5_CANVAS_HTML

    assert "s2s-brush-slider" in _HTML5_CANVAS_HTML
    assert "s2s-canvas-color" in _HTML5_CANVAS_HTML
    assert "s2s-tool-eraser" in _HTML5_CANVAS_HTML
    assert "s2s-btn-clear" in _HTML5_CANVAS_HTML
    assert "s2s-btn-undo" in _HTML5_CANVAS_HTML
    assert "s2s-btn-redo" in _HTML5_CANVAS_HTML
    assert "<canvas" in _HTML5_CANVAS_HTML

def test_preprocess_with_base64_dataurl():
    from ui.app import _run_preprocessing, _PREPROCESSING_AVAILABLE

    minimal_png = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    preview, status = _run_preprocessing(
        canvas_image=None,
        uploaded_file=None,
        canvas_data_url=minimal_png,
    )

    if _PREPROCESSING_AVAILABLE:
        from PIL import Image
        assert isinstance(preview, Image.Image)
        assert "✔" in status or "complete" in status.lower()
    else:
        assert preview is None
        assert "unavailable" in status.lower() or "⚠" in status
