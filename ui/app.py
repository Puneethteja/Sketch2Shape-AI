"""
ui/app.py
=========
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Laxmi (Frontend, WebGL & Technical Documentation)

Professional, utility-driven Gradio interface providing a structured
dual-viewport layout for sketch input and WebGL 3D asset preview.

Layout
------
┌─────────────────────────────────┬─────────────────────────────────┐
│          INPUT PANEL            │         OUTPUT PANEL            │
│  ─ HTML5 Interactive Canvas     │  ─ WebGL 3-D Previewer (Three)  │
│  ─ File uploader (.png/jpg/jpeg)│  ─ Preprocessed lineart view    │
│  ─ Material & style prompt      │  ─ AI Engine depth / normal map │
│  ─ Pipeline parameters          │  ─ Mesh export & binary GLB dl  │
└─────────────────────────────────┴─────────────────────────────────┘

Integration Contracts
---------------------
* Input canvas / file upload  → engine.preprocessing.preprocess() [Puneeth]
* Preprocessed tensor         → engine.ai_engine.generate()      [Mona]
* Generated mesh              → trimesh serialiser / WebGL viewer [Laxmi]
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict, Optional, Tuple
=======
from typing import Optional, Tuple, Union
>>>>>>> 2542941326ba4098c948ce9e0f89301eb7076c7d

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
APP_VERSION = "0.2.0-phase2"

# Supported sketch export formats
SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = ("OBJ", "GLB", "STL", "PLY")

<<<<<<< HEAD
# Session state store for active pipeline assets
_SESSION: Dict[str, Any] = {
    "current_mesh": None,
    "current_source": None,
    "last_result": None,
}

# Placeholder colour for the WebGL viewport stub
_WEBGL_STUB_COLOR = "#1a1a2e"
=======
# Supported raster sketch upload extensions
SUPPORTED_UPLOAD_EXTENSIONS = [".png", ".jpg", ".jpeg"]

# Placeholder colour for the WebGL viewport
_WEBGL_STUB_COLOR = "#090d16"
>>>>>>> 2542941326ba4098c948ce9e0f89301eb7076c7d

# ─────────────────────────────────────────────────────────────────────────────
# WebGL Viewport Stub & HTML Definition
# Preserved for backward compatibility and test validation
# ─────────────────────────────────────────────────────────────────────────────

_WEBGL_STUB_HTML = """
<div style="
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #090d16;
    border: 1px solid #334155;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    font-family: 'SF Mono', 'JetBrains Mono', monospace;
    color: #94a3b8;
    user-select: none;
">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
         stroke="#475569" stroke-width="1.5" stroke-linecap="round"
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
    <span style="font-size: 11px; color: #64748b;">
        Three.js · 360° Orbit · Pan · Zoom · GLB
    </span>
</div>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Embedded WebGL 3D Previewer Component Markup & Controller Hook
# ─────────────────────────────────────────────────────────────────────────────

_WEBGL_VIEWER_HTML = """
<div id="s2s-webgl-viewer-root" style="width: 100%; min-height: 420px; border-radius: 6px; overflow: hidden; border: 1px solid #334155;">
    <!-- Host container populated by ui/viewer.js -->
</div>

<script>
(function() {
    function mountViewer() {
        var container = document.getElementById("s2s-webgl-viewer-root");
        if (!container) return;

        // Load viewer.js if not already present
        if (typeof window.Sketch2ShapeViewer === "undefined") {
            var script = document.createElement("script");
            script.src = "/file=ui/viewer.js";
            script.onload = function() {
                if (window.Sketch2ShapeViewer) {
                    window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container, {
                        showGrid: true,
                        autoRotate: false
                    });
                }
            };
            script.onerror = function() {
                // Inline fallback if static file server is routed differently
                loadViewerInline(container);
            };
            document.head.appendChild(script);
        } else if (!window.s2sViewerInstance) {
            window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container, {
                showGrid: true,
                autoRotate: false
            });
        }
    }

    function loadViewerInline(container) {
        var script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
        script.onload = function() {
            var controls = document.createElement("script");
            controls.src = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js";
            controls.onload = function() {
                var localScript = document.createElement("script");
                localScript.src = "ui/viewer.js";
                localScript.onload = function() {
                    if (window.Sketch2ShapeViewer) {
                        window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container);
                    }
                };
                document.head.appendChild(localScript);
            };
            document.head.appendChild(controls);
        };
        document.head.appendChild(script);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountViewer);
    } else {
        setTimeout(mountViewer, 50);
    }
})();
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Interactive HTML5 Drawing Canvas Markup & Client-side Script
# ─────────────────────────────────────────────────────────────────────────────

_HTML5_CANVAS_HTML = """
<div id="s2s-canvas-component" style="
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px;
    color: #f8fafc;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    user-select: none;
">
    <!-- ── Toolbar Row 1: Brush Size & Tool Modes ── -->
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px;">
        <!-- Tool Selection: Brush vs Eraser -->
        <div style="display: flex; align-items: center; gap: 4px;">
            <button id="s2s-tool-brush" type="button" class="s2s-tool-btn active" title="Brush Tool" style="
                background: #1e293b;
                border: 1px solid #0284c7;
                color: #ffffff;
                padding: 4px 10px;
                font-size: 12px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 5px;
            ">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                    <path d="M2 2l7.586 7.586"/>
                    <circle cx="11" cy="11" r="2"/>
                </svg>
                Brush
            </button>
            <button id="s2s-tool-eraser" type="button" class="s2s-tool-btn" title="Eraser Tool" style="
                background: #1e293b;
                border: 1px solid #334155;
                color: #94a3b8;
                padding: 4px 10px;
                font-size: 12px;
                border-radius: 4px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 5px;
            ">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 20H7L3 16C2 15 2 13 3 12L13 2L22 11L20 20Z"/>
                    <path d="M17 17L7 7"/>
                </svg>
                Eraser
            </button>
        </div>

        <!-- Color Picker & Quick Swatches -->
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Color</span>
            <input id="s2s-canvas-color" type="color" value="#0f172a" title="Pick ink color" style="
                width: 24px;
                height: 24px;
                padding: 0;
                border: 1px solid #475569;
                border-radius: 4px;
                background: transparent;
                cursor: pointer;
            ">
            <!-- Palette swatches -->
            <div style="display: flex; gap: 3px;">
                <span class="s2s-swatch" data-color="#000000" style="width:14px;height:14px;border-radius:2px;background:#000000;cursor:pointer;border:1px solid #475569;" title="Black"></span>
                <span class="s2s-swatch" data-color="#334155" style="width:14px;height:14px;border-radius:2px;background:#334155;cursor:pointer;border:1px solid #475569;" title="Charcoal"></span>
                <span class="s2s-swatch" data-color="#64748b" style="width:14px;height:14px;border-radius:2px;background:#64748b;cursor:pointer;border:1px solid #475569;" title="Graphite"></span>
                <span class="s2s-swatch" data-color="#0284c7" style="width:14px;height:14px;border-radius:2px;background:#0284c7;cursor:pointer;border:1px solid #475569;" title="Cyan Blue"></span>
            </div>
        </div>

        <!-- Brush Size Adjustment -->
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Size</span>
            <input id="s2s-brush-slider" type="range" min="1" max="40" value="4" style="
                width: 90px;
                accent-color: #0284c7;
                cursor: pointer;
            ">
            <span id="s2s-brush-badge" style="font-size: 10px; font-family: monospace; color: #cbd5e1; min-width: 28px;">4px</span>
        </div>
    </div>

    <!-- ── Toolbar Row 2: Undo, Redo, Clear Canvas ── -->
    <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid #1e293b; padding-top: 6px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <button id="s2s-btn-undo" type="button" disabled title="Undo last stroke" style="
                background: #1e293b;
                border: 1px solid #334155;
                color: #64748b;
                padding: 3px 8px;
                font-size: 11px;
                border-radius: 4px;
                cursor: not-allowed;
                display: flex;
                align-items: center;
                gap: 4px;
            ">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 7v6h6"/>
                    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>
                </svg>
                Undo <span id="s2s-undo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
            <button id="s2s-btn-redo" type="button" disabled title="Redo stroke" style="
                background: #1e293b;
                border: 1px solid #334155;
                color: #64748b;
                padding: 3px 8px;
                font-size: 11px;
                border-radius: 4px;
                cursor: not-allowed;
                display: flex;
                align-items: center;
                gap: 4px;
            ">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 7v6h-6"/>
                    <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3l3 2.7"/>
                </svg>
                Redo <span id="s2s-redo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
        </div>

        <button id="s2s-btn-clear" type="button" title="Clear canvas contents" style="
            background: #1e293b;
            border: 1px solid #ef4444;
            color: #f87171;
            padding: 3px 8px;
            font-size: 11px;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
        ">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Clear Canvas
        </button>
    </div>

    <!-- ── Canvas Drawing Surface ── -->
    <div style="
        position: relative;
        width: 100%;
        aspect-ratio: 1 / 1;
        background: #ffffff;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #334155;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        cursor: crosshair;
        touch-action: none;
    ">
        <canvas id="s2s-html5-canvas" width="512" height="512" style="
            width: 100%;
            height: 100%;
            display: block;
        "></canvas>
    </div>
</div>

<script>
(function() {
    function initCanvas() {
        var canvas = document.getElementById("s2s-html5-canvas");
        if (!canvas) return;
        var ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;

        // Fill white background for sketch contrast
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // State variables
        var isDrawing = false;
        var mode = "brush"; // "brush" | "eraser"
        var brushColor = "#0f172a";
        var brushSize = 4;
        var undoStack = [];
        var redoStack = [];
        var maxHistory = 30;
        var lastPoint = null;

        // UI references
        var toolBrush = document.getElementById("s2s-tool-brush");
        var toolEraser = document.getElementById("s2s-tool-eraser");
        var colorPicker = document.getElementById("s2s-canvas-color");
        var brushSlider = document.getElementById("s2s-brush-slider");
        var brushBadge = document.getElementById("s2s-brush-badge");
        var btnUndo = document.getElementById("s2s-btn-undo");
        var btnRedo = document.getElementById("s2s-btn-redo");
        var btnClear = document.getElementById("s2s-btn-clear");
        var undoCount = document.getElementById("s2s-undo-count");
        var redoCount = document.getElementById("s2s-redo-count");

        function updateUIState() {
            if (btnUndo) {
                btnUndo.disabled = undoStack.length === 0;
                btnUndo.style.cursor = undoStack.length > 0 ? "pointer" : "not-allowed";
                btnUndo.style.color = undoStack.length > 0 ? "#cbd5e1" : "#64748b";
                if (undoCount) undoCount.textContent = undoStack.length > 0 ? "(" + undoStack.length + ")" : "";
            }
            if (btnRedo) {
                btnRedo.disabled = redoStack.length === 0;
                btnRedo.style.cursor = redoStack.length > 0 ? "pointer" : "not-allowed";
                btnRedo.style.color = redoStack.length > 0 ? "#cbd5e1" : "#64748b";
                if (redoCount) redoCount.textContent = redoStack.length > 0 ? "(" + redoStack.length + ")" : "";
            }
        }

        function saveState() {
            undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            if (undoStack.length > maxHistory) undoStack.shift();
            redoStack = [];
            updateUIState();
            syncToGradio();
        }

        function syncToGradio() {
            var dataUrl = canvas.toDataURL("image/png");
            var holder = document.querySelector("#canvas-data-holder textarea");
            if (holder) {
                holder.value = dataUrl;
                holder.dispatchEvent(new Event("input", { bubbles: true }));
                holder.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        // Pointer event coordinates mapped to canvas internal resolution
        function getPos(e) {
            var rect = canvas.getBoundingClientRect();
            var scaleX = canvas.width / rect.width;
            var scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.x) * scaleX,
                y: (e.clientY - rect.y) * scaleY
            };
        }

        canvas.addEventListener("pointerdown", function(e) {
            e.preventDefault();
            isDrawing = true;
            canvas.setPointerCapture(e.pointerId);
            saveState();

            var pt = getPos(e);
            lastPoint = pt;

            ctx.beginPath();
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = brushSize;
            ctx.strokeStyle = mode === "eraser" ? "#ffffff" : brushColor;

            ctx.moveTo(pt.x, pt.y);
            ctx.lineTo(pt.x, pt.y);
            ctx.stroke();
        });

        canvas.addEventListener("pointermove", function(e) {
            if (!isDrawing) return;
            e.preventDefault();
            var pt = getPos(e);

            ctx.beginPath();
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = brushSize;
            ctx.strokeStyle = mode === "eraser" ? "#ffffff" : brushColor;

            // Quadratic curve interpolation for smooth strokes
            var midX = (lastPoint.x + pt.x) / 2;
            var midY = (lastPoint.y + pt.y) / 2;
            ctx.moveTo(lastPoint.x, lastPoint.y);
            ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, midX, midY);
            ctx.stroke();

            lastPoint = pt;
        });

        function endStroke(e) {
            if (!isDrawing) return;
            isDrawing = false;
            try { canvas.releasePointerCapture(e.pointerId); } catch(err) {}
            syncToGradio();
        }

        canvas.addEventListener("pointerup", endStroke);
        canvas.addEventListener("pointercancel", endStroke);

        // Tool Mode selection
        if (toolBrush) {
            toolBrush.addEventListener("click", function() {
                mode = "brush";
                toolBrush.style.borderColor = "#0284c7";
                toolBrush.style.color = "#ffffff";
                if (toolEraser) {
                    toolEraser.style.borderColor = "#334155";
                    toolEraser.style.color = "#94a3b8";
                }
            });
        }
        if (toolEraser) {
            toolEraser.addEventListener("click", function() {
                mode = "eraser";
                toolEraser.style.borderColor = "#0284c7";
                toolEraser.style.color = "#ffffff";
                if (toolBrush) {
                    toolBrush.style.borderColor = "#334155";
                    toolBrush.style.color = "#94a3b8";
                }
            });
        }

        // Color Picker & Swatches
        if (colorPicker) {
            colorPicker.addEventListener("input", function(e) {
                brushColor = e.target.value;
                if (toolBrush) toolBrush.click();
            });
        }
        document.querySelectorAll(".s2s-swatch").forEach(function(swatch) {
            swatch.addEventListener("click", function() {
                brushColor = swatch.getAttribute("data-color");
                if (colorPicker) colorPicker.value = brushColor;
                if (toolBrush) toolBrush.click();
            });
        });

        // Brush Size Slider
        if (brushSlider) {
            brushSlider.addEventListener("input", function(e) {
                brushSize = parseInt(e.target.value, 10);
                if (brushBadge) brushBadge.textContent = brushSize + "px";
            });
        }

        // Undo & Redo
        if (btnUndo) {
            btnUndo.addEventListener("click", function() {
                if (undoStack.length === 0) return;
                redoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                var prev = undoStack.pop();
                ctx.putImageData(prev, 0, 0);
                updateUIState();
                syncToGradio();
            });
        }
        if (btnRedo) {
            btnRedo.addEventListener("click", function() {
                if (redoStack.length === 0) return;
                undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                var next = redoStack.pop();
                ctx.putImageData(next, 0, 0);
                updateUIState();
                syncToGradio();
            });
        }

        // Clear Canvas
        function clearCanvas() {
            saveState();
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            syncToGradio();
        }
        if (btnClear) {
            btnClear.addEventListener("click", clearCanvas);
        }

        // Global hook so Gradio Clear button can trigger canvas reset
        window.s2sClearCanvas = function() {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            undoStack = [];
            redoStack = [];
            updateUIState();
            syncToGradio();
        };

        // Initial sync
        setTimeout(syncToGradio, 200);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initCanvas);
    } else {
        setTimeout(initCanvas, 50);
    }
})();
</script>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Handler functions
# ─────────────────────────────────────────────────────────────────────────────

def _run_preprocessing(
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
) -> Tuple[Optional[Image.Image], str]:
    """
    Stage 1 handler: ingest an HTML5 canvas drawing, Gradio sketchpad,
    or uploaded file (.png, .jpg, .jpeg) and run Puneeth's preprocessing pipeline.

    Parameters
    ----------
    canvas_image : np.ndarray or None
        RGBA array from the Gradio sketchpad (``gr.Sketchpad``).
    uploaded_file : str or None
        File path from the file-uploader component.
    canvas_data_url : str or None
        Base64 data-URL string from the interactive HTML5 drawing canvas.

    Returns
    -------
    preview_image : PIL.Image or None
        Preprocessed / lineart image to display in the output panel.
    status_message : str
        Human-readable pipeline status for the status textbox.
    """
    source: Optional[Union[np.ndarray, str]] = None

    # Priority: 1. Gradio canvas ndarray, 2. Uploaded file, 3. HTML5 canvas data-url
    if canvas_image is not None:
        logger.info("UI: Using canvas drawing (ndarray) as source")
        source = canvas_image
    elif uploaded_file is not None:
        logger.info("UI: Using uploaded file as source: %s", uploaded_file)
        source = uploaded_file
    elif canvas_data_url is not None and len(canvas_data_url.strip()) > 0:
        logger.info("UI: Using HTML5 drawing canvas (base64 data-URL) as source")
        source = canvas_data_url
    else:
        return None, "⚠  No input provided. Draw on the canvas or upload a sketch (.png, .jpg, .jpeg)."

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
    """Reset all interactive components to their initial states (exact 7-tuple)."""
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
# Gradio UI Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Construct and return the Gradio Blocks application.

    The layout follows a dual-viewport grid:
      - Left panel  : HTML5 drawing canvas, file uploader (.png/jpg/jpeg), prompt, pipeline params
      - Right panel : WebGL 3D previewer + preprocessed image + export controls
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

        # Hidden holder for HTML5 canvas data URL synchronization
        canvas_data_holder = gr.Textbox(
            visible=False,
            elem_id="canvas-data-holder",
            label="Internal Canvas Buffer",
        )

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
                gr.HTML('<p class="panel-header">Sketch Input</p>')

                # Functional HTML5 Drawing Canvas Component with Toolbar
                gr.HTML(_HTML5_CANVAS_HTML, elem_id="html5-canvas-root")

                # Alternative / Legacy Sketchpad in a compact accordion (retains gr.Sketchpad in Blocks)
                with gr.Accordion(
                    label="Alternative / Fallback Sketchpad",
                    open=False,
                    elem_classes=["compact-accordion"],
                ):
                    canvas = gr.Sketchpad(
                        label="Standard Drawing Canvas",
                        type="numpy",
                        image_mode="RGBA",
                        elem_id="sketch-canvas",
                        height=260,
                    )

                gr.HTML('<div style="margin: 6px 0 4px; font-size: 11px; '
                        'color: var(--body-text-color-subdued); text-align: center;">— or upload sketch / photo —</div>')

                # Drag-and-drop file uploader zone accepting .png, .jpg, .jpeg
                file_upload = gr.File(
                    label="Drag & Drop Sketch (.png, .jpg, .jpeg)",
                    file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                    file_count="single",
                    elem_id="file-uploader",
                )

                # Text input field for optional material/style prompts
                prompt_input = gr.Textbox(
                    label="Material / Style Prompt (Optional)",
                    placeholder=(
                        'e.g. "matte porcelain finish", "brushed aluminum alloy", '
                        '"weathered terracotta" · leave blank for shape-only mode'
                    ),
                    lines=2,
                    max_lines=4,
                    elem_id="prompt-input",
                )

                # Pipeline parameters
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

            # ── RIGHT PANEL · Output & WebGL Viewer ───────────────────────────
            with gr.Column(scale=1, min_width=380):
                gr.HTML('<p class="panel-header">3D WebGL Asset Preview</p>')

                # WebGL 3D Previewer powered by Three.js / ui/viewer.js
                gr.HTML(_WEBGL_VIEWER_HTML, elem_id="webgl-viewport")

                # Preprocessed lineart image result
                preprocessed_view = gr.Image(
                    label="Preprocessed / Lineart Preview",
                    type="pil",
                    interactive=False,
                    show_download_button=True,
                    height=180,
                )

                # Depth / normal map from AI engine (Mona)
                depth_view = gr.Image(
                    label="Depth / Normal Map  [AI Engine Output]",
                    type="pil",
                    interactive=False,
                    show_download_button=True,
                    height=180,
                    visible=True,
                )

                # ── Export controls ────────────────────────────────────────────
                gr.HTML('<p class="panel-header" style="margin-top: 8px;">Mesh Export & Downloads</p>')

                with gr.Row(elem_classes=["export-row"]):
                    export_format = gr.Dropdown(
                        label="Format",
                        choices=list(SUPPORTED_EXPORT_FORMATS),
                        value="GLB",
                        scale=1,
                    )
                    btn_export = gr.Button(
                        "⬇  Export Mesh",
                        variant="secondary",
                        size="sm",
                        scale=2,
                    )

                export_file = gr.File(
                    label="Download Asset",
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
            inputs=[canvas, file_upload, canvas_data_holder],
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
            js="() => { if (window.s2sClearCanvas) window.s2sClearCanvas(); }",
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
        allowed_paths=["ui"],
    )
