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
    sketch: str = Field(..., description="Base64 data URL of the sketch canvas")
    prompt: str = Field(default="", description="Text conditioning prompt")
    controlnet_scale: float = Field(default=0.85, ge=0.0, le=2.0, description="ControlNet conditioning strength")
    inference_steps: int = Field(default=20, ge=1, le=100, description="Diffusion denoising iterations")
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: int = Field(default=-1, description="Deterministic PRNG seed (-1 for random)")
    auto_repair: bool = Field(default=True, description="Enforce watertight non-manifold topological repair")


class ExportRequest(BaseModel):
    format: str = Field(default="glb", description="Desired 3D exchange format (glb, obj, stl, ply)")


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

    @app.post("/api/generate")
    async def api_generate(payload: GenerateRequest) -> Dict[str, Any]:
        """
        Full 4-Stage 2D-to-3D Synthesis Pipeline:
        1. Preprocessing & lineart conditioning
        2. ControlNet Diffusion geometry generation
        3. 3D Volumetric Mesh Reconstruction
        4. Non-manifold topological repair and GLB serialization
        """
        global _LAST_GENERATED_RESULT
        try:
            sketch_data = payload.sketch.strip()
            if not sketch_data:
                raise HTTPException(status_code=400, detail="Please sketch an outline or upload an image first.")

            logger.info("Executing API generation request...")
            result: GenerationResult = generate(
                sketch_input=sketch_data,
                prompt=payload.prompt,
                controlnet_scale=payload.controlnet_scale,
                num_inference_steps=payload.inference_steps,
                seed=payload.seed,
            )
            _LAST_GENERATED_RESULT = result

            # Depth map visualization
            depth_url: Optional[str] = None
            if result.depth_image is not None:
                depth_url = _pil_to_data_url(result.depth_image)

            # Lineart mask visualization
            lineart_url: Optional[str] = None
            try:
                lineart_mask = preprocess(sketch_data, return_lineart=True, return_tensor=False)
                lineart_url = _array_to_data_url(lineart_mask)
            except Exception:
                pass

            glb_filename = Path(result.glb_path).name if result.glb_path else f"generated_mesh_{result.seed_used}.glb"
            glb_url = f"/outputs/{glb_filename}"

            logger.info(
                f"Generation succeeded: {result.vertex_count} vertices, {result.face_count} faces, "
                f"watertight={result.is_watertight}, latency={result.latency_seconds:.2f}s"
            )

            return {
                "status": "success",
                "glb_url": glb_url,
                "seed": result.seed_used,
                "vertices": result.vertex_count,
                "faces": result.face_count,
                "is_watertight": result.is_watertight,
                "latency": round(result.latency_seconds, 2),
                "lineart_url": lineart_url,
                "depth_url": depth_url,
                "prompt": payload.prompt,
                "vram_peak_mb": result.vram_peak_mb,
            }
        except Exception as exc:
            logger.exception(f"Generation pipeline failed: {exc}")
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
