"""
ui/app.py
=========
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Laxmi (Frontend, WebGL & Technical Documentation)

Professional, utility-driven Gradio interface providing a structured
dual-viewport layout for sketch input and 3-D asset preview.

Layout
------
┌─────────────────────────────────┬─────────────────────────────────┐
│          INPUT PANEL            │         OUTPUT PANEL            │
│  ─ Drawing canvas               │  ─ WebGL 3-D previewer stub     │
│  ─ File uploader                │  ─ Preprocessed view            │
│  ─ Text prompt                  │  ─ Export controls              │
│  ─ Pipeline controls            │                                 │
└─────────────────────────────────┴─────────────────────────────────┘

Integration Contracts
---------------------
* Input canvas / file upload  → engine.preprocessing.preprocess()
* Preprocessed tensor         → engine.ai_engine.generate()   [Mona]
* Generated mesh              → trimesh serialiser / WebGL viewer [Laxmi]
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Lazy engine import (engine may not be installed in every demo env) ─────────
try:
    from engine.preprocessing import preprocess, TARGET_SIZE
    _PREPROCESSING_AVAILABLE = True
except ImportError:
    _PREPROCESSING_AVAILABLE = False

# -- Lazy AI engine import (Mona's generative + 3D pipeline) --
try:
    import engine.ai_engine as _ai_engine
    _AI_ENGINE_AVAILABLE = True
except ImportError:
    _AI_ENGINE_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.ui")

# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE   = "Sketch2Shape AI"
APP_VERSION = "0.1.0-alpha"

# Supported sketch export formats (populated by Mona's engine once ready)
SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = ("OBJ", "GLB", "STL", "PLY")

# Session state store for active pipeline assets
_SESSION: Dict[str, Any] = {
    "current_mesh": None,
    "current_source": None,
    "last_result": None,
}

# Placeholder colour for the WebGL viewport stub
_WEBGL_STUB_COLOR = "#1a1a2e"

# ─────────────────────────────────────────────────────────────────────────────
# Handler stubs
# These functions are integration points; internal logic will be filled in
# once Mona's AI engine module is available.
# ─────────────────────────────────────────────────────────────────────────────

def _run_preprocessing(
    canvas_image: Optional[np.ndarray],
    uploaded_file: Optional[str],
) -> Tuple[Optional[Image.Image], str]:
    """
    Stage 1 handler: ingest a canvas drawing or uploaded file and run
    Puneeth's preprocessing pipeline.

    Parameters
    ----------
    canvas_image : np.ndarray or None
        RGBA array from the Gradio sketchpad (``gr.Sketchpad``).
    uploaded_file : str or None
        Temporary file path string from the file-uploader component.

    Returns
    -------
    preview_image : PIL.Image or None
        Preprocessed / lineart image to display in the output panel.
    status_message : str
        Human-readable pipeline status for the status textbox.
    """
    source = None

    if canvas_image is not None:
        logger.info("UI: Using canvas drawing as source")
        source = canvas_image
    elif uploaded_file is not None:
        logger.info("UI: Using uploaded file as source: %s", uploaded_file)
        source = uploaded_file
    else:
        return None, "⚠  No input provided. Draw a sketch or upload an image."

    if not _PREPROCESSING_AVAILABLE:
        return None, (
            "⚠  Preprocessing engine unavailable. "
            "Ensure engine/preprocessing.py is importable."
        )

    try:
        result_np: np.ndarray = preprocess(source, return_tensor=False)
        preview = Image.fromarray(result_np)
        logger.info("UI: Preprocessing complete — shape %s", result_np.shape)
        return preview, "✔  Preprocessing complete. Ready for AI generation."
    except Exception as exc:  # noqa: BLE001
        logger.error("UI: Preprocessing error: %s", exc, exc_info=True)
        return None, f"✘  Preprocessing failed: {exc}"


def _run_ai_generation(
    prompt: str,
    controlnet_scale: float,
    num_inference_steps: int,
    seed: int,
) -> Tuple[Optional[Image.Image], str]:
    """
    Stage 2 handler: invoke Mona's AI engine to produce a depth/normal map.

    Parameters
    ----------
    prompt : str
        Free-form text prompt to condition generation.
    controlnet_scale : float
        ControlNet conditioning scale (0.0 – 2.0).
    num_inference_steps : int
        Diffusion denoising steps.
    seed : int
        RNG seed for reproducibility; -1 = random.

    Returns
    -------
    depth_preview : PIL.Image or None
        Depth / normal map preview image.
    status_message : str
    """
    logger.info(
        "UI: AI generation requested — prompt=%r  scale=%.2f  steps=%d  seed=%d",
        prompt, controlnet_scale, num_inference_steps, seed,
    )
    if not _AI_ENGINE_AVAILABLE:
        return None, "[!] AI engine unavailable. Ensure engine/ai_engine.py is importable."
    try:
        sketch = np.zeros((512, 512, 3), dtype=np.uint8)
        result = _ai_engine.generate(
            sketch_input=sketch,
            prompt=prompt,
            controlnet_scale=controlnet_scale,
            num_inference_steps=int(num_inference_steps),
            seed=int(seed) if seed != -1 else None,
        )
        _SESSION["current_mesh"] = result.mesh
        _SESSION["last_result"] = result
        depth_img = result.depth_image
        status = (
            f"[OK] Generation complete — seed={result.seed} "
            f"latency={result.elapsed_total:.2f}s "
            f"watertight={result.is_watertight}"
        )
        logger.info("UI: %s", status)
        return depth_img, status
    except Exception as exc:  # noqa: BLE001
        logger.error("UI: AI generation error: %s", exc, exc_info=True)
        return None, f"[ERR] AI generation failed: {exc}"


def _run_mesh_export(export_format: str) -> Tuple[Optional[str], str]:
    """
    Stage 3 handler: serialise the current in-memory mesh to the chosen format.

    Parameters
    ----------
    export_format : str
        One of SUPPORTED_EXPORT_FORMATS.

    Returns
    -------
    file_path : str or None
        Absolute path of the exported file (passed to gr.File for download).
    status_message : str
    """
    logger.info("UI: Mesh export requested — format=%s", export_format)
    mesh = _SESSION.get("current_mesh")
    if mesh is None:
        return None, "[!] No mesh in session. Run AI generation first."
    try:
        ext_map = {"OBJ": ".obj", "GLB": ".glb", "STL": ".stl", "PLY": ".ply"}
        ext = ext_map.get(export_format.upper(), ".obj")
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, prefix="sketch2shape_",
        )
        tmp.close()
        mesh.export(tmp.name, file_type=export_format.lower())
        status = f"[OK] Exported {export_format} mesh to {tmp.name}"
        logger.info("UI: %s", status)
        return tmp.name, status
    except Exception as exc:  # noqa: BLE001
        logger.error("UI: Mesh export error: %s", exc, exc_info=True)
        return None, f"[ERR] Mesh export failed: {exc}"


def _clear_session() -> Tuple:
    """Reset all interactive components to their initial states."""
    logger.info("UI: Session cleared by user")
    return (
        None,   # canvas
        None,   # file uploader
        "",     # prompt
        None,   # preprocessed preview
        None,   # depth preview
        None,   # export file
        "",     # status
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebGL Viewport Stub
# Rendered as an HTML block until Three.js / Babylon.js integration is wired.
# ─────────────────────────────────────────────────────────────────────────────

_WEBGL_STUB_HTML = """
<div style="
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #8b949e;
    user-select: none;
">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
         stroke="#444c56" stroke-width="1.5" stroke-linecap="round"
         stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2
                 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2
                 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
    </svg>
    <span style="font-size: 13px; letter-spacing: 0.04em;">
        WebGL 3D Previewer
    </span>
    <span style="font-size: 11px; color: #484f58;">
        Three.js · Awaiting mesh generation
    </span>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Construct and return the Gradio Blocks application.

    The layout follows a dual-viewport grid:
      - Left panel  : input controls (canvas, uploader, prompt, pipeline params)
      - Right panel : WebGL 3-D previewer stub + preprocessed image + export
    """
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ),
        css="""
        /* ── Global resets ────────────────────────────────────────── */
        .gradio-container { max-width: 1440px !important; margin: 0 auto; }

        /* ── Section labels ───────────────────────────────────────── */
        .panel-header {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--body-text-color-subdued);
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border-color-primary);
            margin-bottom: 8px;
        }

        /* ── Compact accordion ────────────────────────────────────── */
        .compact-accordion .label-wrap { padding: 6px 10px !important; }

        /* ── Status bar ───────────────────────────────────────────── */
        .status-bar textarea {
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            color: var(--body-text-color-subdued) !important;
            background: transparent !important;
            border: none !important;
            resize: none !important;
        }

        /* ── Export button row ────────────────────────────────────── */
        .export-row button { font-size: 12px !important; }
        """,
    ) as app:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Row(elem_classes=["header-row"]):
            gr.Markdown(
                f"""
                ## {APP_TITLE}
                Sketch → Diffusion → 3D Mesh &nbsp;·&nbsp;
                TechZephyr 4.0 · IIT Bhubaneswar &nbsp;·&nbsp;
                `v{APP_VERSION}`
                """,
                elem_id="app-title",
            )

        gr.Markdown("---")

        # ── Main dual-viewport grid ───────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT PANEL · Input Controls ───────────────────────────────────
            with gr.Column(scale=1, min_width=380):
                gr.HTML('<p class="panel-header">Input</p>')

                # Drawing canvas
                canvas = gr.Sketchpad(
                    label="Drawing Canvas",
                    type="numpy",
                    image_mode="RGBA",
                    elem_id="sketch-canvas",
                    height=380,
                )

                gr.HTML('<div style="margin: 4px 0 2px; font-size: 11px; '
                        'color: var(--body-text-color-subdued);">— or —</div>')

                # File uploader
                file_upload = gr.File(
                    label="Upload Sketch / Photo",
                    file_types=["image"],
                    file_count="single",
                    elem_id="file-uploader",
                )

                # Text prompt
                prompt_input = gr.Textbox(
                    label="Generation Prompt",
                    placeholder=(
                        'e.g. "a low-poly ceramic vase with a smooth rim" '
                        '· optional, leave blank for shape-only mode'
                    ),
                    lines=2,
                    max_lines=4,
                    elem_id="prompt-input",
                )

                # Pipeline parameters (collapsed by default to reduce clutter)
                with gr.Accordion(
                    label="Pipeline Parameters",
                    open=False,
                    elem_classes=["compact-accordion"],
                ):
                    controlnet_scale = gr.Slider(
                        label="ControlNet Conditioning Scale",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        info="Strength of sketch conditioning. 0.8–1.2 recommended.",
                    )
                    inference_steps = gr.Slider(
                        label="Denoising Steps",
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        info="More steps → higher quality, slower runtime.",
                    )
                    seed_input = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="-1 = random seed.",
                    )

                # Action buttons
                with gr.Row():
                    btn_preprocess = gr.Button(
                        "① Preprocess",
                        variant="secondary",
                        size="sm",
                    )
                    btn_generate = gr.Button(
                        "② Generate 3D",
                        variant="primary",
                        size="sm",
                    )
                    btn_clear = gr.Button(
                        "Clear",
                        variant="stop",
                        size="sm",
                    )

            # ── RIGHT PANEL · Output & Viewer ─────────────────────────────────
            with gr.Column(scale=1, min_width=380):
                gr.HTML('<p class="panel-header">Output</p>')

                # WebGL 3-D previewer stub
                gr.HTML(_WEBGL_STUB_HTML, elem_id="webgl-viewport")

                # Preprocessed image result
                preprocessed_view = gr.Image(
                    label="Preprocessed / Lineart Preview",
                    type="pil",
                    interactive=False,
                    show_download_button=True,
                    height=200,
                )

                # Depth / normal map from AI engine (Mona)
                depth_view = gr.Image(
                    label="Depth / Normal Map  [AI Engine Output]",
                    type="pil",
                    interactive=False,
                    show_download_button=True,
                    height=200,
                    visible=True,
                )

                # ── Export controls ────────────────────────────────────────────
                gr.HTML('<p class="panel-header" style="margin-top: 8px;">Export</p>')

                with gr.Row(elem_classes=["export-row"]):
                    export_format = gr.Dropdown(
                        label="Format",
                        choices=list(SUPPORTED_EXPORT_FORMATS),
                        value="OBJ",
                        scale=1,
                    )
                    btn_export = gr.Button(
                        "⬇  Export Mesh",
                        variant="secondary",
                        size="sm",
                        scale=2,
                    )

                export_file = gr.File(
                    label="Download",
                    interactive=False,
                    visible=True,
                )

        # ── Status bar (full width, below grid) ───────────────────────────────
        status_bar = gr.Textbox(
            label="Pipeline Status",
            value="Ready.",
            interactive=False,
            elem_classes=["status-bar"],
            lines=1,
            max_lines=2,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Event wiring
        # ─────────────────────────────────────────────────────────────────────

        btn_preprocess.click(
            fn=_run_preprocessing,
            inputs=[canvas, file_upload],
            outputs=[preprocessed_view, status_bar],
        )

        btn_generate.click(
            fn=_run_ai_generation,
            inputs=[prompt_input, controlnet_scale, inference_steps, seed_input],
            outputs=[depth_view, status_bar],
        )

        btn_export.click(
            fn=_run_mesh_export,
            inputs=[export_format],
            outputs=[export_file, status_bar],
        )

        btn_clear.click(
            fn=_clear_session,
            inputs=[],
            outputs=[
                canvas,
                file_upload,
                prompt_input,
                preprocessed_view,
                depth_view,
                export_file,
                status_bar,
            ],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_TITLE} UI")
    parser.add_argument("--host",   default="127.0.0.1", help="Bind host")
    parser.add_argument("--port",   type=int, default=7860, help="Bind port")
    parser.add_argument("--share",  action="store_true",   help="Create public Gradio share link")
    parser.add_argument("--debug",  action="store_true",   help="Enable Gradio debug mode")
    args = parser.parse_args()

    logger.info("Starting %s v%s", APP_TITLE, APP_VERSION)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
