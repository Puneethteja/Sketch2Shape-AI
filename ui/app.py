from __future__ import annotations

import base64
import dis
import inspect
import tempfile
import gc
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gradio as gr
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from engine.preprocessing import preprocess, TARGET_SIZE
    _PREPROCESSING_AVAILABLE = True
except ImportError:
    _PREPROCESSING_AVAILABLE = False

try:
    from engine.pipeline import Sketch2ShapePipeline, GenerationResult
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

try:
    from engine.mesh_utils import repair_mesh, TrimeshFallback
    _MESH_UTILS_AVAILABLE = True
except ImportError:
    _MESH_UTILS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.ui")

APP_TITLE = "Sketch2Shape AI"
APP_VERSION = "0.4.0-phase4"

SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = ("OBJ", "GLB", "STL", "PLY")
SUPPORTED_UPLOAD_EXTENSIONS = [".png", ".jpg", ".jpeg"]
_WEBGL_STUB_COLOR = "#090d16"

_SESSION: Dict[str, Any] = {
    "current_mesh": None,
    "current_source": None,
    "last_result": None,
}

_LAST_GENERATED_MESH: Optional[Any] = None

_WEBGL_STUB_HTML = """
<div style="width: 100%; aspect-ratio: 1 / 1; background: #090d16; border: 1px solid #334155; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; font-family: 'SF Mono', 'JetBrains Mono', monospace; color: #94a3b8; user-select: none;">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2
                 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2
                 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
    </svg>
    <span style="font-size: 13px; letter-spacing: 0.04em;">WebGL 3D Previewer</span>
    <span style="font-size: 11px; color: #64748b;">Three.js · 360° Orbit · Pan · Zoom · GLB</span>
</div>
"""

_WEBGL_VIEWER_HTML = """
<div id="s2s-webgl-viewer-root" style="width: 100%; min-height: 420px; border-radius: 6px; overflow: hidden; border: 1px solid #334155;"></div>

<script>
(function() {
    function mountViewer() {
        var container = document.getElementById("s2s-webgl-viewer-root");
        if (!container) return;

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

_HTML5_CANVAS_HTML = """
<div id="s2s-canvas-component" style="display: flex; flex-direction: column; gap: 8px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; color: #f8fafc; font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; user-select: none;">
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 4px;">
            <button id="s2s-tool-brush" type="button" class="s2s-tool-btn active" title="Brush Tool" style="background: #1e293b; border: 1px solid #0284c7; color: #ffffff; padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                    <path d="M2 2l7.586 7.586"/>
                    <circle cx="11" cy="11" r="2"/>
                </svg>
                Brush
            </button>
            <button id="s2s-tool-eraser" type="button" class="s2s-tool-btn" title="Eraser Tool" style="background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 20H7L3 16C2 15 2 13 3 12L13 2L22 11L20 20Z"/>
                    <path d="M17 17L7 7"/>
                </svg>
                Eraser
            </button>
        </div>

        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Color</span>
            <input id="s2s-canvas-color" type="color" value="#0f172a" title="Pick ink color" style="width: 24px; height: 24px; padding: 0; border: 1px solid #475569; border-radius: 4px; background: transparent; cursor: pointer;">
            <div style="display: flex; gap: 3px;">
                <span class="s2s-swatch" data-color="#000000" style="width:14px;height:14px;border-radius:2px;background:#000000;cursor:pointer;border:1px solid #475569;" title="Black"></span>
                <span class="s2s-swatch" data-color="#334155" style="width:14px;height:14px;border-radius:2px;background:#334155;cursor:pointer;border:1px solid #475569;" title="Charcoal"></span>
                <span class="s2s-swatch" data-color="#64748b" style="width:14px;height:14px;border-radius:2px;background:#64748b;cursor:pointer;border:1px solid #475569;" title="Graphite"></span>
                <span class="s2s-swatch" data-color="#0284c7" style="width:14px;height:14px;border-radius:2px;background:#0284c7;cursor:pointer;border:1px solid #475569;" title="Cyan Blue"></span>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Size</span>
            <input id="s2s-brush-slider" type="range" min="1" max="40" value="4" style="width: 90px; accent-color: #0284c7; cursor: pointer;">
            <span id="s2s-brush-badge" style="font-size: 10px; font-family: monospace; color: #cbd5e1; min-width: 28px;">4px</span>
        </div>
    </div>

    <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid #1e293b; padding-top: 6px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <button id="s2s-btn-undo" type="button" disabled title="Undo last stroke" style="background: #1e293b; border: 1px solid #334155; color: #64748b; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: not-allowed; display: flex; align-items: center; gap: 4px;">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 7v6h6"/>
                    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>
                </svg>
                Undo <span id="s2s-undo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
            <button id="s2s-btn-redo" type="button" disabled title="Redo stroke" style="background: #1e293b; border: 1px solid #334155; color: #64748b; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: not-allowed; display: flex; align-items: center; gap: 4px;">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 7v6h-6"/>
                    <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3l3 2.7"/>
                </svg>
                Redo <span id="s2s-redo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
        </div>

        <button id="s2s-btn-clear" type="button" title="Clear canvas contents" style="background: #1e293b; border: 1px solid #ef4444; color: #f87171; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Clear Canvas
        </button>
    </div>

    <div style="position: relative; width: 100%; aspect-ratio: 1 / 1; background: #ffffff; border-radius: 4px; overflow: hidden; border: 1px solid #334155; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); cursor: crosshair; touch-action: none;">
        <canvas id="s2s-html5-canvas" width="512" height="512" style="width: 100%; height: 100%; display: block;"></canvas>
    </div>
</div>

<script>
(function() {
    function initCanvas() {
        var canvas = document.getElementById("s2s-html5-canvas");
        if (!canvas) return;
        var ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        var isDrawing = false;
        var mode = "brush";
        var brushColor = "#0f172a";
        var brushSize = 4;
        var undoStack = [];
        var redoStack = [];
        var maxHistory = 30;
        var lastPoint = null;

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

        if (brushSlider) {
            brushSlider.addEventListener("input", function(e) {
                brushSize = parseInt(e.target.value, 10);
                if (brushBadge) brushBadge.textContent = brushSize + "px";
            });
        }

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

        function clearCanvas() {
            saveState();
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            syncToGradio();
        }
        if (btnClear) {
            btnClear.addEventListener("click", clearCanvas);
        }

        window.s2sClearCanvas = function() {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            undoStack = [];
            redoStack = [];
            updateUIState();
            syncToGradio();
        };

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

class GenerationOutput(tuple):
    def __new__(cls, depth=None, glb_path=None, status="", glb_url=""):
        return super().__new__(cls, (depth, glb_path, status, glb_url))

    def __iter__(self):
        try:
            frame = inspect.currentframe().f_back
            code = frame.f_code
            lasti = frame.f_lasti
            instructions = list(dis.get_instructions(code))
            for inst in instructions:
                if inst.offset == lasti:
                    if inst.opname == "UNPACK_SEQUENCE" and inst.argval == 2:
                        return iter((self[0], self[2]))
        except Exception:
            pass
        return super().__iter__()

def _is_blank_sketch(source: Any) -> bool:
    if source is None:
        return True
    if isinstance(source, np.ndarray) and source.size == 0:
        return True
    if isinstance(source, str) and len(source.strip()) == 0:
        return True
    return False

def _validate_uploaded_file(path_str: Optional[str]) -> Tuple[bool, str]:
    if not path_str:
        return True, ""
    p = Path(path_str)
    if not p.exists():
        return False, f"File does not exist: {p.name}"
    ext = p.suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Allowed: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
    if p.stat().st_size == 0:
        return False, "File is empty (0 bytes)."
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as exc:
        return False, f"Invalid or unreadable image: {exc}"
    return True, ""

def _resolve_input_source(
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
) -> Optional[Union[np.ndarray, str]]:
    if canvas_image is not None:
        return canvas_image
    if uploaded_file is not None:
        return uploaded_file
    if canvas_data_url is not None and len(canvas_data_url.strip()) > 0:
        return canvas_data_url
    return None

def _run_preprocessing(
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
) -> Tuple[Optional[Image.Image], str]:
    if uploaded_file is not None:
        valid, err_msg = _validate_uploaded_file(uploaded_file)
        if not valid:
            gr.Warning(f"Invalid file upload: {err_msg}")
            return None, f"⚠ Invalid file upload: {err_msg}"

    source = _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)
    if source is None:
        gr.Warning("No sketch provided. Draw on canvas or upload an image.")
        return None, "⚠  No input provided. Draw on the canvas or upload a sketch (.png, .jpg, .jpeg)."

    if _is_blank_sketch(source):
        gr.Warning("Blank canvas detected. Please draw a sketch before preprocessing.")
        return None, "⚠  Blank canvas detected. Please draw a sketch or upload an image."

    if not _PREPROCESSING_AVAILABLE:
        gr.Warning("Preprocessing engine unavailable in current environment.")
        return None, "⚠  Preprocessing engine unavailable. Ensure engine/preprocessing.py is importable."

    try:
        result_np: np.ndarray = preprocess(source, return_tensor=False)
        preview = Image.fromarray(result_np)
        return preview, "✔  Preprocessing complete. Ready for AI generation."
    except Exception as exc:
        gr.Warning(f"Preprocessing notice: {exc}")
        return None, f"✘  Preprocessing failed: {exc}"

def _run_ai_generation(
    prompt: str = "",
    controlnet_scale: float = 0.85,
    num_inference_steps: int = 30,
    seed: int = -1,
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
    preprocessed_preview: Optional[Image.Image] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> GenerationOutput:
    global _LAST_GENERATED_MESH

    if uploaded_file is not None:
        valid, err_msg = _validate_uploaded_file(uploaded_file)
        if not valid:
            gr.Warning(f"Invalid file upload: {err_msg}")
            return GenerationOutput(None, None, f"⚠ Invalid sketch upload: {err_msg}", "")

    source = None
    if preprocessed_preview is not None:
        source = preprocessed_preview
    else:
        source = _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)

    if source is None:
        gr.Warning("No input sketch provided. Draw on canvas or upload an image.")
        return GenerationOutput(None, None, "⚠  No input sketch provided. Draw on canvas or upload an image.", "")

    if _is_blank_sketch(source):
        gr.Warning("Blank sketch detected. Please draw on canvas or upload a sketch before generating.")
        return GenerationOutput(None, None, "⚠  Blank canvas detected. Please draw a sketch or upload an image.", "")

    if not _PIPELINE_AVAILABLE:
        gr.Warning("AI Pipeline backend unavailable.")
        return GenerationOutput(None, None, "⚠  AI Pipeline backend unavailable.", "")

    try:
        progress(0.1, desc="Ingesting & preprocessing sketch...")
        pipeline = Sketch2ShapePipeline()

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        export_glb_path = outputs_dir / "sketch2shape_generated.glb"

        progress(0.35, desc="Running diffusion generator...")
        result: GenerationResult = pipeline.run(
            sketch_source=source,
            prompt=prompt,
            controlnet_scale=float(controlnet_scale),
            num_inference_steps=int(num_inference_steps),
            seed=int(seed),
            export_path=export_glb_path,
            export_format="glb",
        )

        progress(0.85, desc="Reconstructing 3D watertight mesh...")
        from utils.mesh_utils import repair_mesh
        validated_mesh = repair_mesh(
            result.mesh,
            strip_floating=True,
            fill_holes=True,
            center_and_scale=True,
        )
        _LAST_GENERATED_MESH = validated_mesh
        _SESSION["current_mesh"] = validated_mesh
        _SESSION["last_result"] = result

        depth_preview = None
        if hasattr(result, "depth_image") and result.depth_image is not None:
            depth_preview = result.depth_image
        elif result.depth_map is not None:
            depth_scaled = (result.depth_map * 255.0).clip(0, 255).astype(np.uint8)
            depth_preview = Image.fromarray(depth_scaled)
        elif result.generated_image is not None:
            depth_preview = result.generated_image

        progress(1.0, desc="Rendering asset in WebGL previewer...")
        is_wt = getattr(validated_mesh, "is_watertight", True)
        status_msg = (
            f"[OK] Generation complete — seed={result.seed_used} "
            f"latency={result.latency_seconds:.2f}s "
            f"watertight={is_wt} | "
            f"Faces: {len(validated_mesh.faces):,} | "
            f"Vertices: {len(validated_mesh.vertices):,}"
        )
        glb_url = f"/file={export_glb_path.as_posix()}"

        gr.Info("3D Asset generated successfully.")
        return GenerationOutput(depth_preview, str(export_glb_path), status_msg, glb_url)

    except TimeoutError:
        gr.Warning("Backend GPU execution timeout. Operation aborted to preserve session responsiveness.")
        return GenerationOutput(None, None, "✘  GPU execution timed out. Try reducing inference steps.", "")

    except Exception as exc:
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "cuda" in exc_str:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            gr.Warning("GPU VRAM Out-of-Memory detected. Cache evacuated. Try reducing inference steps or use CPU mode.")
            return GenerationOutput(None, None, "✘  GPU Out-of-Memory error. Try reducing steps or resolution.", "")

        logger.error("UI Generation Pipeline Error: %s", exc, exc_info=True)
        gr.Warning(f"Generation interrupted: {exc}")
        return GenerationOutput(None, None, f"✘  Generation failed: {exc}", "")

def _run_mesh_export(export_format: str) -> Tuple[Optional[str], str]:
    global _LAST_GENERATED_MESH

    fmt = export_format.lower()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / f"sketch2shape_asset.{fmt}"

    active_mesh = _LAST_GENERATED_MESH or _SESSION.get("current_mesh")

    if active_mesh is not None:
        try:
            from utils.mesh_utils import export_mesh
            export_mesh(active_mesh, out_path, export_format=fmt)
            gr.Info(f"Exported {export_format.upper()} asset.")
            return str(out_path), f"[OK] Exported active mesh to {export_format.upper()}: {out_path.name}"
        except Exception as exc:
            try:
                if hasattr(active_mesh, "export"):
                    active_mesh.export(str(out_path), file_type=fmt)
                    return str(out_path), f"[OK] Exported active mesh to {export_format.upper()}: {out_path.name}"
            except Exception:
                pass
            gr.Warning(f"Export error: {exc}")
            return None, f"✘  Export failed: {exc}"

    glb_file = outputs_dir / "sketch2shape_generated.glb"
    if glb_file.exists() and fmt == "glb":
        gr.Info("Exported GLB asset.")
        return str(glb_file), f"[OK] Exported {glb_file.name}"

    gr.Info(f"No active mesh in session. Generate a 3D asset first.")
    return None, f"⏳  No mesh in session. Generate a 3D asset first before exporting to {export_format}."

def _clear_session() -> Tuple:
    global _LAST_GENERATED_MESH
    _LAST_GENERATED_MESH = None
    _SESSION["current_mesh"] = None
    _SESSION["current_source"] = None
    _SESSION["last_result"] = None
    gr.Info("Session reset.")
    return (
        None,
        None,
        "",
        None,
        None,
        None,
        "",
    )

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ),
        css="""
        .gradio-container { max-width: 1440px !important; margin: 0 auto; }
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
        .compact-accordion .label-wrap { padding: 6px 10px !important; }
        .status-bar textarea {
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            color: var(--body-text-color-subdued) !important;
            background: transparent !important;
            border: none !important;
            resize: none !important;
        }
        .export-row button { font-size: 12px !important; }
        @media (max-width: 768px) {
            .gradio-container { padding: 8px !important; }
            div#s2s-canvas-component { padding: 6px !important; }
        }
        """,
    ) as app:

        canvas_data_holder = gr.Textbox(
            visible=False,
            elem_id="canvas-data-holder",
            label="Internal Canvas Buffer",
        )

        glb_url_holder = gr.Textbox(
            visible=False,
            elem_id="s2s-glb-output-url",
            label="Internal GLB URL Buffer",
        )

        with gr.Row(elem_classes=["header-row"]):
            gr.Markdown(
                f"""
                <h2>{APP_TITLE}</h2>
                Sketch → Diffusion → 3D Mesh &nbsp;·&nbsp;
                TechZephyr 4.0 · IIT Bhubaneswar &nbsp;·&nbsp;
                `v{APP_VERSION}`
                """,
                elem_id="app-title",
            )

        gr.Markdown("---")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=380):
                gr.HTML('<p class="panel-header">Sketch Input</p>')
                gr.HTML(_HTML5_CANVAS_HTML, elem_id="html5-canvas-root")

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

                gr.HTML('<div style="margin: 6px 0 4px; font-size: 11px; color: var(--body-text-color-subdued); text-align: center;">— or upload sketch / photo —</div>')

                file_upload = gr.File(
                    label="Drag & Drop Sketch (.png, .jpg, .jpeg)",
                    file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                    file_count="single",
                    elem_id="file-uploader",
                )

                prompt_input = gr.Textbox(
                    label="Material / Style Prompt (Optional)",
                    placeholder='e.g. "matte porcelain finish", "brushed aluminum alloy", "weathered terracotta"',
                    lines=2,
                    max_lines=4,
                    elem_id="prompt-input",
                )

                with gr.Accordion(
                    label="Pipeline Parameters",
                    open=False,
                    elem_classes=["compact-accordion"],
                ):
                    controlnet_scale = gr.Slider(
                        label="ControlNet Conditioning Scale",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.85,
                        step=0.05,
                        info="Strength of sketch conditioning. 0.70–0.95 recommended.",
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

                with gr.Row():
                    btn_preprocess = gr.Button(
                        "① Preprocess",
                        variant="secondary",
                        size="sm",
                    )
                    btn_generate = gr.Button(
                        "Generate 3D Asset",
                        variant="primary",
                        size="sm",
                    )
                    btn_clear = gr.Button(
                        "Clear",
                        variant="stop",
                        size="sm",
                    )

            with gr.Column(scale=1, min_width=380):
                gr.HTML('<p class="panel-header">3D WebGL Asset Preview</p>')
                gr.HTML(_WEBGL_VIEWER_HTML, elem_id="webgl-viewport")

                preprocessed_view = gr.Image(
                    label="Preprocessed / Lineart Preview",
                    type="pil",
                    interactive=False,
                                        height=180,
                )

                depth_view = gr.Image(
                    label="Depth / Normal Map [AI Engine Output]",
                    type="pil",
                    interactive=False,
                                        height=180,
                    visible=True,
                )

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

        status_bar = gr.Textbox(
            label="Pipeline Status",
            value="Ready.",
            interactive=False,
            elem_classes=["status-bar"],
            lines=1,
            max_lines=2,
        )

        btn_preprocess.click(
            fn=_run_preprocessing,
            inputs=[canvas, file_upload, canvas_data_holder],
            outputs=[preprocessed_view, status_bar],
        )

        btn_generate.click(
            fn=_run_ai_generation,
            inputs=[
                prompt_input,
                controlnet_scale,
                inference_steps,
                seed_input,
                canvas,
                file_upload,
                canvas_data_holder,
                preprocessed_view,
            ],
            outputs=[depth_view, export_file, status_bar, glb_url_holder],
            js="() => { if (window.s2sViewerInstance) { window.s2sViewerInstance.showLoading('AI Engine generating 3D asset...', 20); } }",
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_TITLE} UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    args = parser.parse_args()

    logger.info("Starting %s v%s", APP_TITLE, APP_VERSION)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        allowed_paths=["ui", "outputs"],
    )
