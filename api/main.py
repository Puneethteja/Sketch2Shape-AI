"""
Sketch2Shape-AI — FastAPI Backend
==================================
Two-stage 3D reconstruction endpoint:
  Stage 1 → ControlNet depth/normal guidance (geometry extraction)
  Stage 2 → InstantMesh-style mesh generation (UV-textured .GLB/.OBJ/.STL)

Features
--------
- FP16 precision on CUDA, FP32 fallback on CPU
- Singleton pipeline with lazy initialisation
- CORS enabled for the frontend canvas editor
- Structured JSON error responses
- Static file serving for generated outputs
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# FastAPI imports (fastapi>=0.100.0 required)
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sketch2shape.api")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sketch2Shape-AI API",
    description="2D sketch / image → high-fidelity 3D mesh (GLB/OBJ/STL)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for static file serving
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ---------------------------------------------------------------------------
# Singleton pipeline (lazy init)
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_device: str = "cpu"
_pipeline_dtype: str = "fp32"


def _get_pipeline():
    """Return (and lazily initialise) the Sketch2ShapePipeline singleton."""
    global _pipeline, _pipeline_device, _pipeline_dtype

    if _pipeline is not None:
        return _pipeline

    try:
        import torch
        _use_cuda = torch.cuda.is_available()
    except ImportError:
        _use_cuda = False

    device = "cuda" if _use_cuda else "cpu"
    use_fp16 = _use_cuda  # FP16 only on GPU

    logger.info("Initialising Sketch2ShapePipeline on device=%s fp16=%s", device, use_fp16)

    try:
        # Adjust sys.path so `engine` is importable regardless of cwd
        repo_root = str(BASE_DIR)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from engine.pipeline import Sketch2ShapePipeline  # type: ignore

        _pipeline = Sketch2ShapePipeline(
            device=device,
            use_fp16=use_fp16,
        )
        _pipeline_device = device
        _pipeline_dtype = "fp16" if use_fp16 else "fp32"
        logger.info("Pipeline ready (%s / %s)", _pipeline_device, _pipeline_dtype)

    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        logger.debug(traceback.format_exc())
        # Keep _pipeline as None so the next request retries
        raise RuntimeError(f"Pipeline initialisation failed: {exc}") from exc

    return _pipeline


# ---------------------------------------------------------------------------
# Helper: save PIL image to outputs dir and return URL path
# ---------------------------------------------------------------------------
def _save_atlas(atlas_img, stem: str) -> Optional[str]:
    """Persist texture atlas PNG; return its /outputs URL or None."""
    if atlas_img is None:
        return None
    try:
        filename = f"{stem}_atlas.png"
        dest = OUTPUTS_DIR / filename
        atlas_img.save(str(dest))
        return f"/outputs/{filename}"
    except Exception as exc:
        logger.warning("Could not save texture atlas: %s", exc)
        return None


def _atlas_to_b64(atlas_img) -> Optional[str]:
    """Return base64-encoded PNG string for embedding in JSON."""
    if atlas_img is None:
        return None
    try:
        buf = io.BytesIO()
        atlas_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe — returns device and version info."""
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    except ImportError:
        torch_ver = "not installed"
        cuda_avail = False
        cuda_device = "N/A"

    return {
        "status": "ok",
        "version": "2.0.0",
        "pipeline_initialized": _pipeline is not None,
        "device": _pipeline_device,
        "dtype": _pipeline_dtype,
        "torch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_device": cuda_device,
    }


@app.post("/generate")
async def generate(
    file: UploadFile = File(..., description="Input image or sketch (PNG/JPEG/WebP)"),
    prompt: str = Form(default="", description="Text prompt for material/style guidance"),
    controlnet_scale: float = Form(default=0.8, ge=0.0, le=2.0, description="ControlNet conditioning scale"),
    num_inference_steps: int = Form(default=30, ge=1, le=150, description="Diffusion inference steps"),
    seed: int = Form(default=-1, description="RNG seed (-1 = random)"),
    export_format: str = Form(default="glb", description="Output mesh format: glb | obj | stl"),
):
    """
    Convert a 2D image/sketch into a UV-textured 3D mesh.

    **Stage 1** — ControlNet depth + normal guidance extracts geometry.  
    **Stage 2** — InstantMesh-style reconstruction builds the 3D mesh with
    colour-projected UV texture atlas (no grey wash-out).

    Returns JSON with mesh URL, metadata, and optional base64 texture atlas.
    """
    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    fmt = export_format.lower().strip()
    if fmt not in {"glb", "obj", "stl"}:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported export_format '{export_format}'. Choose glb, obj, or stl.",
        )

    allowed_mime = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type and file.content_type not in allowed_mime:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload PNG, JPEG, or WebP.",
        )

    # -----------------------------------------------------------------------
    # Load image
    # -----------------------------------------------------------------------
    try:
        image_bytes = await file.read()
        from PIL import Image  # type: ignore
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}")

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**31)

    try:
        result = pipeline.run(
            image=input_image,
            prompt=prompt or None,
            seed=actual_seed,
            num_inference_steps=num_inference_steps,
            controlnet_scale=controlnet_scale,
            export_format=fmt,
        )
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    total_time = time.perf_counter() - t0

    # -----------------------------------------------------------------------
    # Persist mesh output
    # -----------------------------------------------------------------------
    run_id = str(uuid.uuid4())[:8]
    mesh_filename = f"mesh_{run_id}.{fmt}"
    mesh_path = OUTPUTS_DIR / mesh_filename
    mesh_url: Optional[str] = None

    try:
        if result.mesh is not None:
            result.mesh.export(str(mesh_path))
            mesh_url = f"/outputs/{mesh_filename}"
            logger.info("Mesh saved → %s (%s)", mesh_path, fmt.upper())
    except Exception as exc:
        logger.warning("Could not save mesh: %s", exc)

    # Also handle gltf_path set by pipeline
    gltf_url: Optional[str] = None
    if getattr(result, "gltf_path", None):
        gp = Path(result.gltf_path)
        if gp.exists():
            dest_gltf = OUTPUTS_DIR / gp.name
            try:
                import shutil
                shutil.copy2(str(gp), str(dest_gltf))
                gltf_url = f"/outputs/{gp.name}"
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Texture atlas
    # -----------------------------------------------------------------------
    atlas_url = _save_atlas(getattr(result, "texture_atlas", None), f"atlas_{run_id}")
    atlas_b64 = _atlas_to_b64(getattr(result, "texture_atlas", None))

    # -----------------------------------------------------------------------
    # Mesh statistics
    # -----------------------------------------------------------------------
    vertices = 0
    faces = 0
    is_watertight = False
    try:
        if result.mesh is not None:
            vertices = len(result.mesh.vertices)
            faces = len(result.mesh.faces)
            is_watertight = bool(getattr(result.mesh, "is_watertight", False))
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Response
    # -----------------------------------------------------------------------
    return JSONResponse(
        content={
            "success": True,
            "run_id": run_id,
            # Mesh
            "mesh_url": mesh_url,
            "gltf_url": gltf_url,
            "export_format": fmt,
            "vertices": vertices,
            "faces": faces,
            "is_watertight": is_watertight,
            # Texture
            "texture_atlas_url": atlas_url,
            "texture_atlas_b64": atlas_b64,
            "has_uv_texture": bool(getattr(result, "has_uv_texture", False)),
            "color_fidelity_score": float(getattr(result, "color_fidelity_score", 1.0)),
            # Generation metadata
            "seed_used": actual_seed,
            "prompt": prompt,
            "controlnet_scale": controlnet_scale,
            "num_inference_steps": num_inference_steps,
            # Timing
            "timing": {
                "total_seconds": round(total_time, 3),
                "stage_timing": getattr(result, "stage_timing", {}),
            },
            # Device info
            "device": _pipeline_device,
            "dtype": _pipeline_dtype,
        }
    )


@app.post("/preview")
async def preview(
    file: UploadFile = File(..., description="Input image or sketch"),
    mode: str = Form(default="depth", description="Preview mode: depth | normal | multiview"),
):
    """
    Return a base64-encoded preview image (depth map, normal map, or multi-view grid).
    Useful for the frontend canvas to show intermediate ControlNet outputs.
    """
    try:
        image_bytes = await file.read()
        from PIL import Image, ImageFilter  # type: ignore
        import numpy as np

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        mode_lc = mode.lower().strip()

        if mode_lc == "depth":
            # Grayscale luminance as proxy depth map
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            # Invert so foreground (dark) is "close"
            depth = 255.0 - gray
            depth_img = Image.fromarray(depth.astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(radius=1)
            )
            out_img = depth_img.convert("RGB")

        elif mode_lc == "normal":
            # Synthetic normal map: encode surface gradient as RGB
            from scipy.ndimage import sobel  # type: ignore
            gray = np.mean(arr, axis=2)
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
            gz = np.ones_like(gray) * 128.0
            length = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-8
            nx = (gx / length * 127.5 + 127.5).clip(0, 255)
            ny = (gy / length * 127.5 + 127.5).clip(0, 255)
            nz = (gz / length * 127.5 + 127.5).clip(0, 255)
            normal_arr = np.stack([nx, ny, nz], axis=2).astype(np.uint8)
            out_img = Image.fromarray(normal_arr)

        else:  # multiview: tile 2×3 rotated previews
            w, h = img.size
            views = []
            for angle in [0, 60, 120, 180, 240, 300]:
                rotated = img.rotate(angle, expand=False)
                views.append(rotated.resize((w // 2, h // 2)))
            grid = Image.new("RGB", (w // 2 * 3, h // 2 * 2))
            for i, v in enumerate(views):
                grid.paste(v, ((i % 3) * (w // 2), (i // 3) * (h // 2)))
            out_img = grid

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return JSONResponse(
            content={
                "mode": mode_lc,
                "width": out_img.width,
                "height": out_img.height,
                "image_b64": b64,
            }
        )

    except Exception as exc:
        logger.error("Preview error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {exc}")


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Download a previously generated mesh file by name."""
    # Prevent path traversal
    safe_name = Path(filename).name
    file_path = OUTPUTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found.")
    return FileResponse(
        path=str(file_path),
        filename=safe_name,
        media_type="application/octet-stream",
    )
