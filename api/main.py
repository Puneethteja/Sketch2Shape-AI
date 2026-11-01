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
import asyncio
import dataclasses

import base64
import io
import logging
import os
import sys
import time
import traceback
import uuid
import json
import urllib.error
import urllib.request
import struct
import hashlib
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


def _load_local_environment() -> None:
    """Load ignored project .env values without overriding process settings."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] == value[-1:] and value.startswith(("'", '"')):
                value = value[1:-1]
            if key and key.replace("_", "").isalnum():
                os.environ.setdefault(key, value)
    except OSError as exc:
        logger.warning("Could not read local .env file: %s", exc)


_load_local_environment()

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
            force_cpu=(device == 'cpu'),
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


def _analyze_mesh(mesh, reference_image):
    """Return diagnostics computed from actual mesh topology and source pixels."""
    if mesh is None:
        return {"available": False, "reason": "No mesh was returned."}
    try:
        import numpy as np
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        bounds = np.asarray(mesh.bounds, dtype=float)
        degenerate_count = int(np.count_nonzero(mesh.area_faces <= 1e-12))
        try:
            edge_counts = np.bincount(np.asarray(mesh.edges_unique_inverse), minlength=len(mesh.edges_unique))
            boundary_edges = int(np.count_nonzero(edge_counts == 1))
            nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
        except Exception:
            boundary_edges = None
            nonmanifold_edges = None
        try:
            components = mesh.split(only_watertight=False)
        except Exception:
            components = []
        watertight = bool(getattr(mesh, "is_watertight", False))
        integrity = max(0, min(100, int(round(100 * (0.65 * float(watertight) + 0.35 * (1 - degenerate_count / max(1, len(faces))))))))
        result = {
            "available": True, "vertices": int(len(vertices)), "faces": int(len(faces)),
            "dimensions": [round(float(x), 5) for x in (bounds[1] - bounds[0])],
            "watertight": watertight, "degenerate_faces": degenerate_count,
            "boundary_edges": boundary_edges, "nonmanifold_edges": nonmanifold_edges,
            "connected_components": int(len(components)) if components else None,
            "mesh_integrity": integrity,
            "validity_score": None,
            "shape_similarity": None, "structural_consistency": None,
            "issues": ([f"{boundary_edges} boundary edges detected."] if boundary_edges else []) + ([f"{nonmanifold_edges} non-manifold edges detected."] if nonmanifold_edges else []) + ([f"{degenerate_count} degenerate faces detected."] if degenerate_count else []),
            "parts": []
        }
        if components:
            for index, component in enumerate(sorted(components, key=lambda value: len(value.faces), reverse=True), 1):
                cb = np.asarray(component.bounds, dtype=float)
                result["parts"].append({"name": f"Connected mesh part {index}", "vertices": int(len(component.vertices)),
                    "faces": int(len(component.faces)), "dimensions": [round(float(x), 5) for x in cb[1]-cb[0]],
                    "center": [round(float(x), 5) for x in component.centroid]})
        try:
            import numpy as np
            source_gray = np.asarray(reference_image.convert("L"))
            foreground_pixels = int(np.count_nonzero(source_gray < 220))
            if foreground_pixels >= 30:
                from engine.usp import compute_sketch_fidelity
                fidelity = compute_sketch_fidelity(reference_image, mesh)
                result["shape_similarity"] = round(float(fidelity.percentage), 1)
        except Exception as exc:
            logger.info("Silhouette comparison unavailable: %s", exc)
        available_metrics = [integrity]
        if result["shape_similarity"] is not None:
            available_metrics.append(result["shape_similarity"])
        result["validity_score"] = round(sum(available_metrics) / len(available_metrics))
        return result
    except Exception as exc:
        logger.warning("Mesh analysis unavailable: %s", exc)
        return {"available": False, "reason": "Mesh inspection failed."}


def _validate_glb(payload: bytes, provider: str = "3D service"):
    """Validate GLB container, drawable primitives, and PBR texture payloads."""
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise ValueError("The generation service did not return a GLB container.")
    version, declared_length = struct.unpack_from("<II", payload, 4)
    if version != 2 or declared_length != len(payload):
        raise ValueError("The returned GLB header or file length is invalid.")
    json_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if chunk_type != 0x4E4F534A or 20 + json_length > len(payload):
        raise ValueError("The returned GLB has no valid glTF JSON chunk.")
    gltf = json.loads(payload[20:20 + json_length].decode("utf-8").rstrip(" \t\r\n\0"))
    primitives = [p for mesh in gltf.get("meshes", []) for p in mesh.get("primitives", [])]
    if not primitives or not any("POSITION" in p.get("attributes", {}) for p in primitives):
        raise ValueError("The returned GLB contains no renderable mesh geometry.")
    material_count = len(gltf.get("materials", []))
    texture_count = len(gltf.get("textures", []))
    image_count = len(gltf.get("images", []))
    if material_count == 0 or texture_count == 0 or image_count == 0:
        raise ValueError(f"{provider} returned a GLB without PBR materials and embedded texture images.")
    return {"mesh_primitives": len(primitives), "materials": material_count,
            "textures": texture_count, "embedded_images": image_count, "pbr_textured": True}


def _meshy_json_request(method: str, url: str, api_key: str, payload=None):
    """Call Meshy from the backend so the API credential never reaches the browser."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            details = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            details = {}
        message = details.get("message") or details.get("error") or "Check the Meshy API key, model access, and available credits."
        raise HTTPException(status_code=502, detail=f"Meshy API returned HTTP {exc.code}: {str(message)[:400]}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to Meshy: {exc.reason}") from exc


def _download_remote_file(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


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

    reported_device = _pipeline_device if _pipeline is not None else ("cuda" if cuda_avail else "cpu")
    reported_dtype = _pipeline_dtype if _pipeline is not None else ("fp16" if cuda_avail else "fp32")
    provider_setting = os.getenv("SKETCH2SHAPE_PROVIDER", "tokenhub").strip().lower()
    tokenhub_configured = bool(os.getenv("TENCENT_TOKENHUB_API_KEY") or os.getenv("HUNYUAN3D_API_KEY"))
    meshy_configured = bool(os.getenv("MESHY_API_KEY"))
    if provider_setting == "local":
        provider = "Local Diffusers/ControlNet"
        model = "Stable Diffusion 1.5 + ControlNet"
    elif provider_setting == "meshy":
        provider = "Meshy 7.1 Image-to-3D" if meshy_configured else "Meshy Image-to-3D (API key missing)"
        model = os.getenv("MESHY_AI_MODEL", "meshy-7.1")
    elif tokenhub_configured:
        provider = "Tencent TokenHub Hi3D"
        model = os.getenv("HUNYUAN3D_MODEL", "hi3d-3.0")
    else:
        provider = "Tencent TokenHub Hi3D (API key missing)"
        model = os.getenv("HUNYUAN3D_MODEL", "hi3d-3.0")

    return {
        "status": "ok",
        "version": "2.0.0",
        "pipeline_initialized": _pipeline is not None,
        "device": reported_device,
        "dtype": reported_dtype,
        "torch_version": torch_ver,
        "cuda_available": cuda_avail,
        "cuda_device": cuda_device,
        "generation_provider": provider,
        "tokenhub_configured": tokenhub_configured,
        "meshy_configured": meshy_configured,
        "model": model,
        "provider_setting": provider_setting,
    }


@app.post("/prepare-crop")
async def prepare_crop(file: UploadFile = File(...)):
    """Return a centered clean crop preview and the detector's confidence note."""
    try:
        from PIL import Image
        from utils.sketch_crop import prepare_sketch
        source = Image.open(io.BytesIO(await file.read())).convert("RGB")
        preview, metadata = prepare_sketch(source)
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        return JSONResponse(content={"image_b64": base64.b64encode(buf.getvalue()).decode("ascii"), **metadata})
    except Exception as exc:
        logger.exception("Sketch crop preview failed")
        raise HTTPException(status_code=400, detail=f"Could not prepare crop preview: {exc}") from exc


@app.post("/generate")
async def generate(
    file: UploadFile = File(..., description="Input image or sketch (PNG/JPEG/WebP)"),
    prompt: str = Form(default="", description="Text prompt for material/style guidance"),
    controlnet_scale: float = Form(default=0.8, ge=0.0, le=2.0, description="ControlNet conditioning scale"),
    num_inference_steps: int = Form(default=30, ge=1, le=150, description="Diffusion inference steps"),
    seed: int = Form(default=-1, description="RNG seed (-1 = random)"),
    export_format: str = Form(default="glb", description="Output mesh format: glb | obj | stl"),
    category: str = Form(default="Auto-Detect from Sketch"),
    input_mode: str = Form(default="image"),
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
    # Tencent TokenHub Hi3D image-conditioned generation.
    tokenhub_key = os.getenv("TENCENT_TOKENHUB_API_KEY") or os.getenv("HUNYUAN3D_API_KEY")
    provider_setting = os.getenv("SKETCH2SHAPE_PROVIDER", "tokenhub").strip().lower()
    if provider_setting not in {"tokenhub", "meshy", "local"}:
        raise HTTPException(status_code=500, detail="SKETCH2SHAPE_PROVIDER must be 'tokenhub', 'meshy', or 'local'.")
    meshy_key = os.getenv("MESHY_API_KEY")
    if provider_setting == "meshy" and not meshy_key:
        raise HTTPException(status_code=503, detail="Meshy is not configured. Run .\\scripts\\configure_meshy.ps1, then restart with .\\scripts\\start_local.ps1.")
    if provider_setting == "tokenhub" and not tokenhub_key:
        raise HTTPException(status_code=503, detail="Tencent TokenHub is not configured. Run .\\scripts\\configure_tencent.ps1, then restart the backend. The local Diffusers generator is available only when SKETCH2SHAPE_PROVIDER=local and its model weights are installed.")

    if provider_setting == "meshy" and meshy_key:
        if input_mode not in {"draw", "image"}:
            raise HTTPException(status_code=422, detail="input_mode must be 'draw' or 'image'.")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is too large for Meshy (maximum 20 MB).")
        from utils.sketch_crop import prepare_sketch
        normalized, crop_info = prepare_sketch(input_image)
        if crop_info["uncertain"]:
            raise HTTPException(status_code=422, detail=f"Automatic crop needs your review: {crop_info['reason']} Open Crop Preview and adjust the bounds before generating.")

        image_buffer = io.BytesIO()
        normalized.save(image_buffer, format="PNG")
        image_data_uri = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")
        meshy_model = os.getenv("MESHY_AI_MODEL", "meshy-7.1").strip()
        if meshy_model not in {"meshy-7.1", "meshy-6", "meshy-6-lite"}:
            raise HTTPException(status_code=422, detail="MESHY_AI_MODEL must be meshy-7.1, meshy-6, or meshy-6-lite.")

        texture_prompt = ". ".join(part for part in (
            "Create color and physically based materials appropriate to the depicted object",
            "Object category: " + category if not category.lower().startswith("auto") else "",
            prompt.strip(),
        ) if part)[:800]
        meshy_payload = {
            "image_url": image_data_uri,
            "model_type": "standard",
            "ai_model": meshy_model,
            "geometry_resolution": "standard",
            "should_texture": True,
            "enable_pbr": True,
            "target_formats": ["glb"],
        }
        # Image enhancement is accepted by meshy-7.1 / meshy-6 but not
        # meshy-6-lite. Keep the request valid if an operator selects Lite.
        if meshy_model != "meshy-6-lite":
            meshy_payload["image_enhancement"] = False
        if texture_prompt:
            meshy_payload["texture_prompt"] = texture_prompt

        started_at = time.perf_counter()
        create_result = await asyncio.to_thread(
            _meshy_json_request, "POST", "https://api.meshy.ai/openapi/v1/image-to-3d", meshy_key, meshy_payload
        )
        task_id = create_result.get("result")
        if not task_id:
            raise HTTPException(status_code=502, detail="Meshy accepted no task ID for this image-to-3D request.")

        task_url = f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}"
        deadline = time.monotonic() + 600
        task = {}
        while time.monotonic() < deadline:
            task = await asyncio.to_thread(_meshy_json_request, "GET", task_url, meshy_key)
            status = str(task.get("status", "")).upper()
            if status == "SUCCEEDED":
                break
            if status in {"FAILED", "CANCELED"}:
                task_error = task.get("task_error") or {}
                reason = task_error.get("message") or f"Meshy task status: {status}."
                raise HTTPException(status_code=502, detail=f"Meshy image-to-3D generation failed: {reason[:400]}")
            await asyncio.sleep(2)
        if str(task.get("status", "")).upper() != "SUCCEEDED":
            raise HTTPException(status_code=504, detail="Meshy generation is still processing. Check the task in your Meshy account before retrying.")

        glb_url = (task.get("model_urls") or {}).get("glb")
        if not glb_url:
            raise HTTPException(status_code=502, detail="Meshy completed without a GLB result.")
        model_bytes = await asyncio.to_thread(_download_remote_file, glb_url)
        try:
            output_validation = _validate_glb(model_bytes, "Meshy")
            import trimesh
            loaded_scene = trimesh.load(io.BytesIO(model_bytes), file_type="glb", force="scene")
            if not isinstance(loaded_scene, trimesh.Scene) or not loaded_scene.geometry:
                raise ValueError("The GLB loader found no geometry nodes.")
            mesh = loaded_scene.to_geometry()
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError("The GLB does not contain non-empty triangle geometry.")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Meshy returned an invalid or untextured GLB: {exc}") from exc

        run_id = str(uuid.uuid4())[:8]
        glb_filename = f"mesh_{run_id}.glb"
        (OUTPUTS_DIR / glb_filename).write_bytes(model_bytes)
        mesh_filename = glb_filename
        if fmt != "glb":
            mesh_filename = f"mesh_{run_id}.{fmt}"
            mesh.export(str(OUTPUTS_DIR / mesh_filename), file_type=fmt)
        return JSONResponse(content={
            "success": True, "run_id": run_id,
            "mesh_url": f"/outputs/{mesh_filename}", "gltf_url": f"/outputs/{glb_filename}",
            "export_format": fmt, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "is_watertight": bool(mesh.is_watertight), "category": category,
            "generation_provider": "Meshy Image-to-3D", "source_image_used": True,
            "analysis": _analyze_mesh(mesh, normalized), "output_validation": output_validation,
            "generation_settings": {
                "model": meshy_model, "model_type": meshy_payload["model_type"],
                "geometry_resolution": meshy_payload["geometry_resolution"],
                "should_texture": True, "enable_pbr": True,
            },
            "mesh_task_id": task_id, "credits_used": task.get("consumed_credits"),
            "input_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "conditioned_image_sha256": hashlib.sha256(image_buffer.getvalue()).hexdigest(),
            "crop_review": crop_info,
            "timing": {"total_seconds": round(time.perf_counter() - started_at, 3)},
            "device": "Meshy", "dtype": "remote",
        })

    if provider_setting == "tokenhub" and tokenhub_key:
        if input_mode not in {"draw", "image"}:
            raise HTTPException(status_code=422, detail="input_mode must be 'draw' or 'image'.")
        if len(image_bytes) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image is too large for TokenHub Hi3D (maximum 20 MB).")
        # Crop/clean/center only when the foreground detector is confident.
        from utils.sketch_crop import prepare_sketch
        normalized, crop_info = prepare_sketch(input_image)
        if crop_info["uncertain"]:
            raise HTTPException(status_code=422, detail=f"Automatic crop needs your review: {crop_info['reason']} Open Crop Preview and adjust the bounds before generating.")
        image_buffer = io.BytesIO()
        normalized.save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        if len(encoded_image) > 6 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="The normalized image exceeds TokenHub Hi3D's 20 MB request limit.")
        model_id = os.getenv("HUNYUAN3D_MODEL", "hi3d-3.0").strip().lower()
        # Earlier app versions wrote the Cloud API slug into TokenHub config.
        # Migrate that known 3.0 alias; do not map 3.1 because TokenHub has a
        # separate model list and its current API contract differs.
        if model_id == "hy-3d-3.0":
            model_id = "hi3d-3.0"
        if model_id not in {"hi3d-2.1", "hi3d-3.0"}:
            raise HTTPException(status_code=422, detail="TokenHub HUNYUAN3D_MODEL must be hi3d-2.1 or hi3d-3.0. Re-run .\\scripts\\configure_tencent.ps1 to update an older model setting.")
        try:
            face_count = int(os.getenv("HUNYUAN3D_FACE_COUNT", "100000"))
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="HUNYUAN3D_FACE_COUNT must be an integer supported by the selected Hi3D model.") from exc
        max_faces = 2_000_000 if model_id == "hi3d-2.1" else 5_000_000
        if not 100_000 <= face_count <= max_faces:
            raise HTTPException(status_code=422, detail=f"HUNYUAN3D_FACE_COUNT must be between 100000 and {max_faces} for {model_id}.")
        resolution = os.getenv("HUNYUAN3D_RESOLUTION", "2048quality" if model_id == "hi3d-3.0" else "1536pro").strip().lower()
        allowed_resolutions = {"2048quality", "2048master"} if model_id == "hi3d-3.0" else {"1536fast", "1536pro"}
        if resolution not in allowed_resolutions:
            raise HTTPException(status_code=422, detail=f"HUNYUAN3D_RESOLUTION must be one of: {', '.join(sorted(allowed_resolutions))}.")
        payload = {"model": model_id,
                   "image_base64": encoded_image,
                   "request_type": 3,
                   "pbr": 1,
                   "face": face_count,
                   "resolution": resolution,
                   "format": 2}
        request = urllib.request.Request("https://tokenhub.tencentmaas.com/v1/api/3d/submit",
            data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {tokenhub_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                job = json.loads(response.read())
            job_id = job["id"]
            submit_request_id = job.get("request_id")
            model_id = payload["model"]
            deadline = time.monotonic() + 240
            job_result = job
            while time.monotonic() < deadline:
                if job_result.get("status") == "completed": break
                if job_result.get("status") == "failed":
                    error = job_result.get("error") or {}
                    reason = error.get("message") or error.get("code") or "Generation failed."
                    request_id = job_result.get("request_id") or submit_request_id
                    suffix = f" (Request ID: {request_id})" if request_id else ""
                    raise HTTPException(status_code=502, detail=f"Tencent TokenHub Hi3D failed: {str(reason)[:300]}{suffix}")
                await asyncio.sleep(3)
                query = urllib.request.Request("https://tokenhub.tencentmaas.com/v1/api/3d/query",
                    data=json.dumps({"model": model_id, "id": job_id}).encode(), headers={"Authorization": f"Bearer {tokenhub_key}", "Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(query, timeout=30) as response:
                    job_result = json.loads(response.read())
            if job_result.get("status") != "completed":
                raise HTTPException(status_code=504, detail="HY-3D job is still processing; retry generation.")
            data = job_result.get("data")
            if isinstance(data, dict):
                model_url = data.get("url")
            elif isinstance(data, list):
                model_url = next((item.get("url") for item in data if isinstance(item, dict) and item.get("type", "").lower() == "glb"), None)
            else:
                model_url = None
            if not model_url:
                raise HTTPException(status_code=502, detail="TokenHub Hi3D completed without a GLB download URL.")
            with urllib.request.urlopen(model_url, timeout=120) as response:
                model_bytes = response.read()
            run_id = str(uuid.uuid4())[:8]
            try:
                output_validation = _validate_glb(model_bytes, "Tencent TokenHub Hi3D")
                import trimesh
                loaded_scene = trimesh.load(io.BytesIO(model_bytes), file_type="glb", force="scene")
                if not isinstance(loaded_scene, trimesh.Scene) or not loaded_scene.geometry:
                    raise ValueError("The GLB loader found no geometry nodes.")
                remote_mesh = loaded_scene.to_geometry()
                if not isinstance(remote_mesh, trimesh.Trimesh) or len(remote_mesh.vertices) == 0 or len(remote_mesh.faces) == 0:
                    raise ValueError("The GLB does not contain non-empty triangle geometry.")
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"HY-3D returned an invalid or untextured GLB: {exc}") from exc
            glb_filename = f"mesh_{run_id}.glb"
            (OUTPUTS_DIR / glb_filename).write_bytes(model_bytes)
            mesh_filename = glb_filename
            if fmt != "glb":
                mesh_filename = f"mesh_{run_id}.{fmt}"
                remote_mesh.export(str(OUTPUTS_DIR / mesh_filename), file_type=fmt)
            analysis = _analyze_mesh(remote_mesh, normalized)
            remote_vertices, remote_faces = len(remote_mesh.vertices), len(remote_mesh.faces)
            remote_watertight = bool(remote_mesh.is_watertight)
            remote_glb_url = f"/outputs/{glb_filename}"
            return JSONResponse(content={"success": True, "run_id": run_id, "mesh_url": f"/outputs/{mesh_filename}",
                "gltf_url": remote_glb_url, "export_format": fmt,
                "vertices": remote_vertices, "faces": remote_faces, "is_watertight": remote_watertight,
                "category": category, "generation_provider": "Tencent TokenHub Hi3D", "source_image_used": True,
                "analysis": analysis, "output_validation": output_validation,
                "generation_settings": {"model": model_id, "request_type": payload["request_type"], "pbr": payload["pbr"], "face_count": payload["face"], "resolution": payload["resolution"], "format": "glb", "sketch_mode_supported_by_provider": False, "category_prompt_supported_by_provider": False},
                "tokenhub_request_id": job_result.get("request_id") or submit_request_id,
                "input_image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "conditioned_image_sha256": hashlib.sha256(image_buffer.getvalue()).hexdigest(),
                "crop_review": crop_info,
                "timing": {"total_seconds": 0}, "device": "Tencent HY-3D", "dtype": "remote"})
        except HTTPException:
            raise
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            try:
                error_body = json.loads(raw_error)
            except Exception:
                error_body = {}
            response_body = error_body.get("Response") if isinstance(error_body.get("Response"), dict) else {}
            error = error_body.get("error") or response_body.get("Error") or {}
            if not isinstance(error, dict):
                error = {}
            message = error.get("message") or error.get("Message") or error_body.get("message") or error_body.get("detail")
            request_id = error_body.get("request_id") or error_body.get("RequestId") or response_body.get("RequestId")
            safe_message = str(message or "Check that the API key is valid, Hi3D is enabled for the account, and the model is available.")[:300]
            logger.error("Tencent TokenHub Hi3D returned HTTP %s; request_id=%s; error=%s", exc.code, request_id or "unavailable", safe_message)
            suffix = f" (Request ID: {request_id})" if request_id else ""
            raise HTTPException(status_code=502, detail=f"Tencent TokenHub Hi3D returned HTTP {exc.code}: {safe_message}{suffix}") from exc
        except Exception as exc:
            logger.exception("Tencent TokenHub Hi3D generation failed")
            raise HTTPException(status_code=502, detail=f"Tencent TokenHub Hi3D generation failed: {exc}") from exc

    # Local legacy pipeline (not Hunyuan3D). Explicitly identify its provider.
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    actual_seed = seed if seed >= 0 else int(time.time() * 1000) % (2**31)
    category_hint = "" if category.lower().startswith("auto") else f"Object hint: {category}. Preserve the object's identity and silhouette from the provided image."
    conditioned_prompt = ", ".join(part for part in (category_hint, prompt.strip()) if part) or None

    try:
        result = pipeline.run(
            sketch_source=input_image,
            prompt=conditioned_prompt,
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
    mesh_analysis = _analyze_mesh(result.mesh, input_image)

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
                "stage_timing": dataclasses.asdict(result.stage_timing) if dataclasses.is_dataclass(getattr(result, "stage_timing", None)) else {},
            },
            # Device info
            "device": _pipeline_device,
            "dtype": _pipeline_dtype,
            "generation_provider": "Legacy local Stable Diffusion + reconstruction",
            "source_image_used": True,
            "analysis": mesh_analysis,
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

        if mode_lc == "lineart":
            import cv2
            gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            lineart = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 7)
            out_img = Image.fromarray(lineart, mode="L").convert("RGB")

        elif mode_lc == "depth":
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

# Mount frontend Single Page Application at root
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
