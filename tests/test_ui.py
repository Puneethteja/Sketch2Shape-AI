"""
tests/test_ui.py
================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Laxmi (Frontend, WebGL & Technical Documentation)

Smoke-test suite for ui/app.py & ui/viewer.js.

What is tested
--------------
1.  Gradio import sanity          – gradio is installed and importable.
2.  build_ui() returns a Blocks   – the factory runs without errors.
3.  Component inventory           – expected component types are present.
4.  _run_preprocessing (no input) – returns a warning string, no crash.
5.  _run_preprocessing (ndarray)  – routes a synthetic NumPy image correctly.
6.  _run_ai_generation (stub)     – returns stub status, no crash.
7.  _run_mesh_export (stub)       – returns stub status, no crash.
8.  _clear_session                – returns a 7-tuple of reset values.
9.  WebGL stub HTML               – placeholder div is non-empty.
10. Supported upload extensions   – includes .png, .jpg, .jpeg.
11. WebGL viewer script           – ui/viewer.js exists, non-empty, and has orbit/download/spinner.
12. HTML5 canvas component        – contains brush size, color picker, eraser, clear, undo/redo.
13. Base64 data-URL preprocess    – routes canvas data-URL correctly.

Run with:
    pytest tests/test_ui.py -v
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Make sure project root is importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 · Gradio import sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_gradio_importable():
    """gradio must be installed and importable."""
    import gradio as gr  # noqa: F401
    assert gr.__version__, "gradio version string should be non-empty"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 · build_ui() returns a Blocks instance
# ─────────────────────────────────────────────────────────────────────────────

def test_build_ui_returns_blocks():
    """build_ui() should return a gr.Blocks without raising."""
    import gradio as gr
    from ui.app import build_ui

    demo = build_ui()
    assert isinstance(demo, gr.Blocks), (
        f"Expected gr.Blocks, got {type(demo)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 · Component inventory — key components present in the Blocks
# ─────────────────────────────────────────────────────────────────────────────

def test_ui_contains_expected_components():
    """The Blocks graph must include Sketchpad, File, Textbox, Image, Button, Slider."""
    import gradio as gr
    from ui.app import build_ui

    demo = build_ui()

    # Walk all registered components
    component_types = {type(c).__name__ for c in demo.blocks.values()}

    required = {"Sketchpad", "File", "Textbox", "Image", "Button", "Slider"}
    missing  = required - component_types
    assert not missing, f"Missing UI components: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 · _run_preprocessing — no input → graceful warning
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_no_input_returns_warning():
    """Calling _run_preprocessing with all inputs None must return a warning."""
    from ui.app import _run_preprocessing

    preview, status = _run_preprocessing(None, None)

    assert preview is None, "Preview should be None when no input provided"
    assert "No input" in status or "⚠" in status, (
        f"Expected a warning message, got: {status!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 · _run_preprocessing — synthetic NumPy canvas input
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_with_synthetic_canvas():
    """
    Pass a synthetic RGBA NumPy array (as the Gradio Sketchpad delivers).
    If the preprocessing engine is available the result should be a PIL Image;
    if the engine is absent the function must still return gracefully.
    """
    from ui.app import _run_preprocessing, _PREPROCESSING_AVAILABLE

    # Simulate a blank white canvas (512×512 RGBA) — same shape as gr.Sketchpad output
    canvas = np.full((512, 512, 4), 255, dtype=np.uint8)

    preview, status = _run_preprocessing(canvas, None)

    if _PREPROCESSING_AVAILABLE:
        from PIL import Image
        assert isinstance(preview, Image.Image), (
            "Expected PIL.Image when preprocessing engine is available"
        )
        assert "✔" in status or "complete" in status.lower(), (
            f"Unexpected status: {status!r}"
        )
    else:
        # Engine not installed — must fail gracefully, not raise
        assert preview is None
        assert "unavailable" in status.lower() or "⚠" in status


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 · _run_ai_generation — stub returns gracefully
# ─────────────────────────────────────────────────────────────────────────────

def test_ai_generation_stub_returns_gracefully():
    """The stub _run_ai_generation must return (None, str) without raising."""
    from ui.app import _run_ai_generation

    result_img, status = _run_ai_generation(
        prompt="a ceramic vase",
        controlnet_scale=1.0,
        num_inference_steps=30,
        seed=42,
    )

    assert result_img is None, "Stub should return None for image"
    assert isinstance(status, str) and len(status) > 0, (
        "Stub should return a non-empty status string"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 · _run_mesh_export — stub returns gracefully
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fmt", ["OBJ", "GLB", "STL", "PLY"])
def test_mesh_export_stub_returns_gracefully(fmt):
    """_run_mesh_export stub must handle every supported format without raising."""
    from ui.app import _run_mesh_export

    file_path, status = _run_mesh_export(fmt)

    # Stub: no mesh in session yet, so file_path may be None
    assert isinstance(status, str) and len(status) > 0, (
        f"Expected a non-empty status string for format {fmt}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 · _clear_session — returns the correct reset tuple
# ─────────────────────────────────────────────────────────────────────────────

def test_clear_session_returns_seven_tuple():
    """_clear_session must return exactly 7 values (one per output component)."""
    from ui.app import _clear_session

    result = _clear_session()

    assert isinstance(result, tuple), "Expected a tuple from _clear_session"
    assert len(result) == 7, (
        f"Expected 7-element reset tuple, got {len(result)}"
    )

    canvas, file_up, prompt, pre_view, depth_view, export_file, status = result

    assert canvas      is None, "Canvas should reset to None"
    assert file_up     is None, "File uploader should reset to None"
    assert prompt      == "",   "Prompt should reset to empty string"
    assert pre_view    is None, "Preprocessed view should reset to None"
    assert depth_view  is None, "Depth view should reset to None"
    assert export_file is None, "Export file should reset to None"
    assert status      == "",   "Status bar should reset to empty string"


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 · WebGL stub HTML — placeholder is non-empty and contains key tokens
# ─────────────────────────────────────────────────────────────────────────────

def test_webgl_stub_html_is_valid():
    """The WebGL placeholder HTML must be a non-empty string with a div tag."""
    from ui.app import _WEBGL_STUB_HTML

    assert isinstance(_WEBGL_STUB_HTML, str), "Stub HTML must be a string"
    assert len(_WEBGL_STUB_HTML) > 100,       "Stub HTML seems too short"
    assert "<div"  in _WEBGL_STUB_HTML,        "Stub HTML must contain a <div>"
    assert "WebGL" in _WEBGL_STUB_HTML,        "Stub HTML must mention WebGL"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 · Supported upload extensions
# ─────────────────────────────────────────────────────────────────────────────

def test_supported_upload_extensions():
    """Upload zone must accept .png, .jpg, and .jpeg files."""
    from ui.app import SUPPORTED_UPLOAD_EXTENSIONS

    for ext in [".png", ".jpg", ".jpeg"]:
        assert ext in SUPPORTED_UPLOAD_EXTENSIONS, (
            f"Expected {ext} to be supported in file uploads"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 · WebGL viewer script exists and contains required capabilities
# ─────────────────────────────────────────────────────────────────────────────

def test_viewer_js_capabilities():
    """ui/viewer.js must exist and contain orbit, pan, zoom, GLB download, and spinner logic."""
    viewer_path = ROOT / "ui" / "viewer.js"
    assert viewer_path.exists(), "ui/viewer.js should exist on disk"

    content = viewer_path.read_text(encoding="utf-8")
    assert len(content) > 1000, "ui/viewer.js seems too short"
    assert "Sketch2ShapeViewer" in content, "Missing Sketch2ShapeViewer class"
    assert "OrbitControls" in content, "Missing 360-degree orbit rotation logic"
    assert "downloadBinary" in content, "Missing binary GLB download handler"
    assert "spinner" in content.lower(), "Missing loading spinner logic"
    assert "loadGLB" in content, "Missing GLB loading function"


# ─────────────────────────────────────────────────────────────────────────────
# Test 12 · HTML5 canvas component contains required controls
# ─────────────────────────────────────────────────────────────────────────────

def test_html5_canvas_markup_and_controls():
    """_HTML5_CANVAS_HTML must contain brush slider, color picker, eraser, clear, and undo/redo."""
    from ui.app import _HTML5_CANVAS_HTML

    assert "s2s-brush-slider" in _HTML5_CANVAS_HTML, "Missing brush size slider"
    assert "s2s-canvas-color" in _HTML5_CANVAS_HTML, "Missing color picker"
    assert "s2s-tool-eraser" in _HTML5_CANVAS_HTML, "Missing eraser tool button"
    assert "s2s-btn-clear" in _HTML5_CANVAS_HTML, "Missing clear canvas button"
    assert "s2s-btn-undo" in _HTML5_CANVAS_HTML, "Missing undo buffer button"
    assert "s2s-btn-redo" in _HTML5_CANVAS_HTML, "Missing redo buffer button"
    assert "<canvas" in _HTML5_CANVAS_HTML, "Missing HTML5 <canvas> element"


# ─────────────────────────────────────────────────────────────────────────────
# Test 13 · Preprocessing with base64 data-URL input
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_with_base64_dataurl():
    """_run_preprocessing must accept canvas_data_url without crashing."""
    from ui.app import _run_preprocessing, _PREPROCESSING_AVAILABLE

    # Minimal 1x1 transparent PNG as base64 data-URL
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
