"""
server.py
=========
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Unified FastAPI Production Server: Serves the Web SPA Frontend and
exposes high-performance REST APIs for diffusion-guided 3D synthesis.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.ai_engine import generate, get_pipeline, GenerationResult
from engine.preprocessing import preprocess, ingest
from engine.trellis_engine import (
    get_trellis_engine,
    TRELLIS_MATERIALS,
    MODEL_METRICS_SPECS,
    TrellisGenerationResult,
)
from engine.usp import (
    compute_sketch_fidelity,
    generate_multiview_projections,
    generate_design_variations,
    ComponentEditor,
    CAMERA_PRESETS,
    MATERIAL_PALETTES,
    COMPONENT_PRESETS,
    DESIGN_STYLE_PRESETS,
)
from check_gpu import inspect_system
from utils.mesh_utils import export_mesh, repair_mesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.server")

# Fast global reference to last generated result for instant multi-format downloads
_LAST_GENERATED_RESULT: Optional[GenerationResult] = None


# ── Request / Response Schemas ───────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    sketch: str = Field(..., description="Base64 data URL or file path of the 2D sketch input")
    return_lineart: bool = Field(default=True, description="Whether to compute binary lineart mask")


class GenerateRequest(BaseModel):
    sketch: Optional[str] = Field(default=None, description="Base64 data URL or path of the 2D sketch input")
    image: Optional[str] = Field(default=None, description="Alias for sketch (image data URL or path)")
    prompt: str = Field(default="", description="Optional aesthetic guidance or text prompt conditioning")
    material: str = Field(default="ceramic", description="Target material finish: 'ceramic', 'slate', or 'bronze'")
    hf_token: Optional[str] = Field(default=None, description="Optional Hugging Face token to bypass ZeroGPU anonymous quota limits")


class ExportRequest(BaseModel):
    format: str = Field(default="glb", description="Desired 3D exchange format (glb, obj, stl, ply)")


class FidelityRequest(BaseModel):
    sketch: str = Field(..., description="Base64 data URL or path of the input sketch")
    view: str = Field(default="front", description="Camera projection plane ('front', 'side', 'top')")


class ComponentEditRequest(BaseModel):
    component: str = Field(..., description="Component zone to edit (e.g. Upper Section / Top Rim)")
    material: str = Field(..., description="Target material palette name")
    custom_color: Optional[str] = Field(default=None, description="Optional hex color string (#RRGGBB)")


class VariationsRequest(BaseModel):
    sketch: str = Field(..., description="Base64 data URL of the input sketch")
    prompt: str = Field(default="", description="Base user prompt to blend with style presets")
    styles: Optional[list[str]] = Field(default=None, description="List of style preset names to run")


# ── Application Factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Sketch2Shape AI — Generative 2D-to-3D Synthesis Server",
        description="Dual-stage diffusion-guided geometric 3D reconstruction engine with watertight topology guarantees.",
        version="0.4.0",
    )

    # Enable Cross-Origin Resource Sharing (CORS) for maximum dev flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure required directories exist
    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    web_dir = ROOT_DIR / "web"
    if not web_dir.exists():
        raise RuntimeError(f"Web SPA directory not found at {web_dir}")

    # Helper: Convert NumPy array to PNG base64 data URL
    def _array_to_data_url(arr: np.ndarray) -> str:
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # Helper: Convert PIL Image to PNG base64 data URL
    def _pil_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # ── API Endpoints ────────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health_check() -> Dict[str, Any]:
        """Health check and service status."""
        return {
            "status": "ok",
            "app": "Sketch2Shape AI",
            "version": "0.4.0",
            "engine": "FastAPI + PyTorch FP16",
            "timestamp": time.time(),
        }

    @app.get("/api/system/diagnostics")
    async def get_system_diagnostics() -> Dict[str, Any]:
        """Hardware, CUDA, VRAM, and runtime diagnostic inspection."""
        try:
            report = inspect_system(benchmark=False)
            return asdict(report)
        except Exception as exc:
            logger.error(f"Error inspecting system: {exc}")
            return {"error": str(exc), "cuda_available": False, "system_status": "ERROR"}

    @app.get("/api/samples")
    async def get_samples_manifest() -> Dict[str, Any]:
        """Curated demo asset manifest with pre-computed watertight models."""
        manifest_paths = [
            web_dir / "assets" / "samples" / "manifest.json",
            outputs_dir / "samples" / "manifest.json",
        ]
        for p in manifest_paths:
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(f"Failed to read manifest at {p}: {exc}")
        return {"samples": []}

    @app.post("/api/preprocess")
    async def api_preprocess(payload: PreprocessRequest) -> Dict[str, Any]:
        """
        Stage 1: Bilateral denoising, aspect-ratio letterboxing, and Gaussian
        adaptive binary lineart stroke extraction.
        """
        try:
            sketch_data = payload.sketch.strip()
            if not sketch_data:
                raise HTTPException(status_code=400, detail="Empty sketch payload provided.")

            img_ingested = ingest(sketch_data)
            processed_color = preprocess(img_ingested, return_lineart=False, return_tensor=False)
            lineart_mask = preprocess(img_ingested, return_lineart=True, return_tensor=False)

            return {
                "status": "success",
                "preprocessed_url": _array_to_data_url(processed_color),
                "lineart_url": _array_to_data_url(lineart_mask),
            }
        except Exception as exc:
            logger.exception(f"Preprocessing failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(exc)}")

    @app.get("/api/model/metrics")
    async def get_model_metrics() -> Dict[str, Any]:
        """Microsoft TRELLIS.2-4B architecture specifications, parameter breakdown, and metrics."""
        return {
            "status": "success",
            "specs": MODEL_METRICS_SPECS,
            "materials": TRELLIS_MATERIALS,
        }

    @app.post("/api/generate-3d")
    @app.post("/api/generate")
    async def api_generate(payload: GenerateRequest) -> Dict[str, Any]:
        """
        Microsoft TRELLIS (TRELLIS.2-4B) 3D Synthesis Pipeline:
        Converts 2D visual contours to high-density, watertight 3D meshes using
        Structured 3D Latents (SLat) with automated multi-format export (STL, OBJ, GLB, PLY).
        """
        global _LAST_GENERATED_RESULT
        try:
            sketch_data = (payload.sketch or payload.image or "").strip()
            if not sketch_data:
                raise HTTPException(status_code=400, detail="Please sketch an outline or upload an image first.")

            logger.info("Executing Microsoft TRELLIS.2-4B generation request...")
            trellis_engine = get_trellis_engine()
            res = trellis_engine.generate(
                image_input=sketch_data,
                prompt=payload.prompt,
                material=payload.material,
                hf_token=payload.hf_token,
                output_dir=outputs_dir,
            )
            _LAST_GENERATED_RESULT = res

            logger.info(
                f"Microsoft TRELLIS generation succeeded: {res.vertices} vertices, {res.faces} faces, "
                f"watertight={res.is_watertight}, latency={res.latency_seconds:.2f}s"
            )

            return {
                "status": "success",
                "model_name": "Microsoft TRELLIS (TRELLIS.2-4B)",
                "glb_url": res.glb_url,
                "obj_url": res.obj_url,
                "stl_url": res.stl_url,
                "ply_url": res.ply_url,
                "preview_url": res.preview_image_url,
                "vertices": res.vertices,
                "faces": res.faces,
                "is_watertight": res.is_watertight,
                "latency": res.latency_seconds,
                "material_used": res.material_used,
                "used_real_model": getattr(res, "used_real_model", False),
                "inference_mode": res.metrics.get("inference_mode", "unknown"),
                "formats": ["STL", "OBJ", "GLB", "PLY"],
                "metrics": res.metrics,
            }
        except Exception as exc:
            logger.exception(f"TRELLIS pipeline failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    @app.post("/api/export")
    async def api_export(payload: ExportRequest) -> Dict[str, Any]:
        """
        Export the active generated mesh to standard exchange formats:
        GLB (Binary 3D), OBJ (Wavefront), STL (3D Print), or PLY (Stanford Polygon).
        """
        global _LAST_GENERATED_RESULT
        fmt = payload.format.lower().strip().replace(".", "")
        if fmt not in ("glb", "obj", "stl", "ply"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{fmt}'. Choose from 'glb', 'obj', 'stl', or 'ply'.",
            )

        target_mesh = None
        if _LAST_GENERATED_RESULT is not None and _LAST_GENERATED_RESULT.mesh is not None:
            target_mesh = _LAST_GENERATED_RESULT.mesh
        else:
            # Fallback to default sample vase model
            fallback_glb = web_dir / "assets" / "samples" / "sample_01_vase" / "model.glb"
            if fallback_glb.exists():
                try:
                    import trimesh
                    target_mesh = trimesh.load(str(fallback_glb))
                except Exception as exc:
                    logger.warning(f"Could not load fallback sample mesh: {exc}")

        if target_mesh is None:
            raise HTTPException(status_code=404, detail="No active 3D model available to export.")

        export_filename = f"sketch2shape_exported.{fmt}"
        export_path = outputs_dir / export_filename

        try:
            export_mesh(target_mesh, export_path, export_format=fmt)
            return {
                "status": "success",
                "download_url": f"/outputs/{export_filename}",
                "filename": export_filename,
                "format": fmt.upper(),
            }
        except Exception as exc:
            logger.exception(f"Export failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Export failed: {str(exc)}")

    # ── USP Suite Endpoints ──────────────────────────────────────────────────

    def _get_active_mesh():
        """Helper to get current active mesh from memory or sample fallback."""
        global _LAST_GENERATED_RESULT
        if _LAST_GENERATED_RESULT is not None and getattr(_LAST_GENERATED_RESULT, "mesh", None) is not None:
            return _LAST_GENERATED_RESULT.mesh
        # Fallback to default sample vase model
        fallback_glb = web_dir / "assets" / "samples" / "sample_01_vase" / "model.glb"
        if fallback_glb.exists():
            try:
                import trimesh
                return trimesh.load(str(fallback_glb))
            except Exception as exc:
                logger.warning(f"Could not load fallback sample mesh: {exc}")
        return None

    @app.post("/api/usp/fidelity")
    async def api_fidelity_score(payload: FidelityRequest) -> Dict[str, Any]:
        """USP 1: Compute geometric IoU, Dice coefficient, and visual contour overlap."""
        try:
            mesh = _get_active_mesh()
            if mesh is None:
                raise HTTPException(status_code=404, detail="No active 3D model available for fidelity comparison.")

            report = compute_sketch_fidelity(
                sketch_input=payload.sketch,
                mesh=mesh,
                view=payload.view,
                canvas_size=512,
            )

            return {
                "status": "success",
                "percentage": report.percentage,
                "rating": report.rating,
                "explanation": report.explanation,
                "iou_score": report.iou_score,
                "dice_score": report.dice_score,
                "metrics": report.metrics,
                "comparison_image_url": _pil_to_data_url(report.comparison_image),
            }
        except Exception as exc:
            logger.exception(f"Fidelity calculation failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Fidelity error: {str(exc)}")

    @app.post("/api/usp/multiview")
    async def api_multiview() -> Dict[str, Any]:
        """USP 2: Render 4-perspective orthographic projections (Front, Side, Back, Top)."""
        try:
            mesh = _get_active_mesh()
            if mesh is None:
                raise HTTPException(status_code=404, detail="No active 3D model available for multiview projections.")

            composite_img = generate_multiview_projections(mesh, tile_size=256)
            return {
                "status": "success",
                "multiview_url": _pil_to_data_url(composite_img),
                "camera_presets": CAMERA_PRESETS,
            }
        except Exception as exc:
            logger.exception(f"Multiview generation failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Multiview error: {str(exc)}")

    @app.get("/api/usp/components")
    async def api_get_components() -> Dict[str, Any]:
        """USP 3: List component zones and material palettes for component editing."""
        return {
            "status": "success",
            "components": ComponentEditor.get_components(),
            "materials": ComponentEditor.get_materials(),
            "material_palettes": {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in MATERIAL_PALETTES.items()},
        }

    @app.post("/api/usp/component-edit")
    async def api_component_edit(payload: ComponentEditRequest) -> Dict[str, Any]:
        """USP 3: Apply component-level material and recoloring to 3D geometry."""
        global _LAST_GENERATED_RESULT
        try:
            mesh = _get_active_mesh()
            if mesh is None:
                raise HTTPException(status_code=404, detail="No active 3D model available to edit.")

            updated_mesh = ComponentEditor.apply_edit(
                mesh=mesh,
                component_name=payload.component,
                material_name=payload.material,
                custom_color=payload.custom_color,
            )

            # Update cached result
            if _LAST_GENERATED_RESULT is not None:
                _LAST_GENERATED_RESULT.mesh = updated_mesh

            # Export updated mesh as GLB for immediate viewer reload
            edit_filename = f"edited_mesh_{int(time.time())}.glb"
            edit_path = outputs_dir / edit_filename
            export_mesh(updated_mesh, edit_path, export_format="glb")

            return {
                "status": "success",
                "glb_url": f"/outputs/{edit_filename}",
                "component": payload.component,
                "material": payload.material,
                "vertices": len(updated_mesh.vertices),
                "faces": len(updated_mesh.faces),
            }
        except Exception as exc:
            logger.exception(f"Component edit failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Component edit error: {str(exc)}")

    @app.post("/api/usp/variations")
    async def api_design_variations(payload: VariationsRequest) -> Dict[str, Any]:
        """USP 4: Generate stylized AI design variations."""
        try:
            variations = generate_design_variations(
                sketch_input=payload.sketch,
                user_prompt=payload.prompt,
                selected_styles=payload.styles,
            )

            results = []
            for style_key, var_img, blended_prompt in variations:
                results.append({
                    "style": style_key,
                    "prompt": blended_prompt,
                    "image_url": _pil_to_data_url(var_img),
                })

            return {
                "status": "success",
                "variations": results,
                "available_styles": list(DESIGN_STYLE_PRESETS.keys()),
            }
        except Exception as exc:
            logger.exception(f"Design variations failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Variations error: {str(exc)}")

    # ── Static File Mounts ───────────────────────────────────────────────────

    # Mount /outputs for dynamic 3D GLB/OBJ models and generated maps
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    # Mount the full SPA (index.html, js, css, assets) as the root application
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="spa_web")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch2Shape AI Web SPA Application Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🚀 Sketch2Shape AI — Generative 2D-to-3D Synthesis Studio")
    print(f" 🌐 Web Interface (SPA) : http://{args.host}:{args.port}")
    print(f" 📚 Interactive API Docs : http://{args.host}:{args.port}/docs")
    print("=" * 70 + "\n")

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
