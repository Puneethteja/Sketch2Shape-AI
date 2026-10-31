# Sketch2Shape AI — Technical Architecture Specification

> **Document series:** TechZephyr 4.0 · IIT Bhubaneswar  
> **Author:** Laxmi — Lead Frontend, WebGL & Technical Documentation  
> **Status:** Living document · Phase 1 baseline  
> **Last revised:** 2026-10-31

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Dataflow — Tensor Lifecycle](#2-system-dataflow--tensor-lifecycle)
3. [Algorithmic Specifications](#3-algorithmic-specifications)
   - 3.1 [Preprocessing Pipeline](#31-preprocessing-pipeline-puneeth)
   - 3.2 [ControlNet Conditioning](#32-controlnet-conditioning-mona)
   - 3.3 [FP16 Memory Management](#33-fp16-memory-management)
   - 3.4 [Trimesh Non-Manifold Recovery](#34-trimesh-non-manifold-recovery)
4. [Component Integration Contracts](#4-component-integration-contracts)
   - 4.1 [Contract A — Preprocessing → AI Engine](#41-contract-a--preprocessing--ai-engine)
   - 4.2 [Contract B — AI Engine → UI Layer](#42-contract-b--ai-engine--ui-layer)
   - 4.3 [Contract C — UI Layer → Mesh Exporter](#43-contract-c--ui-layer--mesh-exporter)
5. [Error Propagation Model](#5-error-propagation-model)
6. [Dependency & Runtime Matrix](#6-dependency--runtime-matrix)

---

## 1. System Overview

Sketch2Shape AI converts a raw hand-drawn sketch (or uploaded raster image) into
a watertight 3-D mesh through a three-stage diffusion-guided pipeline:

```
 [User Input]          [Preprocessing]      [AI Engine]        [3-D Serialisation]
  Canvas / File   →   Puneeth's pipeline  →  Mona's engine  →  trimesh + WebGL
  (PNG / JPEG /       (OpenCV + PIL)         (ControlNet +      (OBJ / GLB / STL)
   base64 blob)        §3.1                   Stable Diffusion)  §3.4
                                               §3.2 / §3.3
```

The Gradio UI (Laxmi, `ui/app.py`) orchestrates all three stages, exposes
interactive controls, and renders the WebGL 3-D viewport.

---

## 2. System Dataflow — Tensor Lifecycle

The following table traces every representational transformation from raw
user input to final serialised mesh file.

| Step | Stage | Operator | Input shape / type | Output shape / type | Notes |
|------|-------|----------|--------------------|---------------------|-------|
| 0 | **Ingestion** | `engine.preprocessing.ingest()` | `str \| Path \| bytes \| np.ndarray \| PIL.Image` | `np.ndarray (H, W, C) uint8` | Handles file path, base64 data-URL, raw bytes, NumPy, Pillow |
| 1 | **Channel normalisation** | `engine.preprocessing.to_rgb()` | `(H, W) \| (H, W, 1) \| (H, W, 3) \| (H, W, 4)` uint8 | `(H, W, 3)` uint8 RGB | Alpha composited over white `(255,255,255)` |
| 2 | **Letterbox resize** | `engine.preprocessing.pad_and_resize()` | `(H, W, 3)` uint8 | `(512, 512, 3)` uint8 | `INTER_AREA` on downscale; `INTER_CUBIC` on upscale |
| 3 | **Bilateral denoising** | `engine.preprocessing.bilateral_filter()` | `(512, 512, 3)` uint8 | `(512, 512, 3)` uint8 | `d=9, σ_color=75, σ_space=75` |
| 4a | **Colour tensor** | `engine.preprocessing._to_tensor()` | `(512, 512, 3)` uint8 | `torch.Tensor (3, 512, 512)` float32 `[0,1]` | HWC → CHW, `/255` |
| 4b | **Lineart mask** | `engine.preprocessing.to_lineart()` | `(512, 512, 3)` uint8 | `(512, 512)` uint8 `{0,255}` | Gaussian adaptive threshold + morphological opening → tensor `(1, 512, 512)` |
| 5 | **ControlNet conditioning** | `engine.ai_engine.generate()` | Lineart tensor `(1, 512, 512)` float32 | Latent `(4, 64, 64)` float16 | Conditioning scale applied to `down_block_res_samples` and `mid_block_res_sample` |
| 6 | **Diffusion decode** | `AutoencoderKL.decode()` | Latent `(4, 64, 64)` float16 | `(3, 512, 512)` float16 | Latent ÷ 0.18215 before decode |
| 7 | **Depth estimation** | Depth head (MiDaS / ZoeDepth) | `(3, 512, 512)` float32 | Depth map `(1, 512, 512)` float32 | Normalised to `[0, 1]`; fed into marching-cubes voxelisation |
| 8 | **Voxelisation** | SDF grid construction | Depth map `(1, 512, 512)` float32 | Occupancy grid `(N, N, N)` bool | Default N=128; configurable |
| 9 | **Mesh extraction** | `trimesh` marching cubes | Occupancy grid | `trimesh.Trimesh` raw (may be non-manifold) | §3.4 recovery applied post-extraction |
| 10 | **Mesh repair** | `trimesh` non-manifold recovery | `trimesh.Trimesh` | `trimesh.Trimesh` watertight | See §3.4 |
| 11 | **Serialisation** | `trimesh.exchange` | `trimesh.Trimesh` | `.obj` / `.glb` / `.stl` / `.ply` bytes | Streamed to `outputs/` and offered for download |

### Dataflow diagram

```
  [Raw Input]
       │ ingest()
       ▼
  (H,W,C) uint8
       │ to_rgb()
       ▼
  (H,W,3) uint8 RGB
       │ pad_and_resize()
       ▼
  (512,512,3) uint8
       │ bilateral_filter()
       ▼
  (512,512,3) uint8 denoised
       ├──────────────────────────────────────────┐
       │ _to_tensor()                             │ to_lineart() → _to_tensor()
       ▼                                          ▼
  Tensor(3,512,512) f32                    Tensor(1,512,512) f32
  [Colour cond. input]                     [ControlNet scribble cond.]
                                                  │
                                                  │ ControlNet forward (§3.2)
                                                  ▼
                                           Latent (4,64,64) f16
                                                  │ VAE decode (§3.3)
                                                  ▼
                                           Image (3,512,512) f32
                                                  │ Depth head
                                                  ▼
                                           Depth map (1,512,512) f32
                                                  │ Voxelisation
                                                  ▼
                                           Occupancy grid (128³) bool
                                                  │ Marching cubes
                                                  ▼
                                           trimesh.Trimesh [raw]
                                                  │ Non-manifold repair (§3.4)
                                                  ▼
                                           trimesh.Trimesh [watertight]
                                                  │ serialise
                                                  ▼
                                           .obj / .glb / .stl / .ply
```

---

## 3. Algorithmic Specifications

### 3.1 Preprocessing Pipeline (Puneeth)

Implemented in [`engine/preprocessing.py`](../engine/preprocessing.py).

#### Stage constants

| Parameter | Symbol | Default | Valid range | Effect |
|-----------|--------|---------|-------------|--------|
| Target resolution | `TARGET_SIZE` | `(512, 512)` | any square power of 2 | Model expects 512×512 |
| Padding colour | `PAD_COLOR` | `(255,255,255)` | any RGB | Letterbox fill |
| Bilateral diameter | `BILATERAL_D` | `9` | 5 – 15 (odd preferred) | Larger → more smoothing |
| Bilateral colour σ | `BILATERAL_SIGMA_COLOR` | `75.0` | 10 – 150 | Controls colour-blending radius |
| Bilateral spatial σ | `BILATERAL_SIGMA_SPACE` | `75.0` | 10 – 150 | Controls spatial influence radius |
| Adaptive threshold block | `LINEART_BLOCK_SIZE` | `11` | odd ≥ 3 | Local neighbourhood for binarisation |
| Threshold constant | `LINEART_C` | `2` | -10 – 20 | Positive → fewer foreground pixels |
| Morphological kernel | `LINEART_MORPH_KERNEL` | `2` | 0 (skip) – 5 | Noise removal granularity |

#### Accepted input types

```
ImageInput = str | Path | bytes | bytearray | np.ndarray | PIL.Image.Image
```

Data-URL format: `data:image/<fmt>;base64,<payload>`

#### Output contract

```python
preprocess(source, return_tensor=True)
# → torch.Tensor shape (3, 512, 512) float32, range [0.0, 1.0]   (colour)

preprocess(source, return_lineart=True, return_tensor=True)
# → torch.Tensor shape (1, 512, 512) float32, values {0.0, 1.0}  (lineart)
```

---

### 3.2 ControlNet Conditioning (Mona)

#### Model reference

- Base model: `runwayml/stable-diffusion-v1-5` (or `stabilityai/stable-diffusion-2-1`)
- ControlNet checkpoint: `lllyasviel/sd-controlnet-scribble`
- Scheduler: `UniPCMultistepScheduler` (default; swappable)

#### Conditioning scale parameter — `controlnet_conditioning_scale`

```
Range:        [0.0, 2.0]
Default:      1.0
Recommended:  0.8 – 1.2  (sketch-to-shape regime)
Hard bounds:  < 0.5  → conditioning too weak; sketch ignored
              > 1.5  → over-conditioning; aliasing artifacts likely
```

The scale multiplies the ControlNet residuals injected into the UNet at each
`down_block_res_sample_i` and `mid_block_res_sample` before the cross-attention
steps:

```
unet_down_block_res_samples[i] += controlnet_scale * ctrl_down_block[i]
unet_mid_block_res_sample      += controlnet_scale * ctrl_mid_block
```

#### Inference parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_inference_steps` | 30 | 20–50 for interactive; 75–100 for final export |
| `guidance_scale` | 7.5 | Classifier-free guidance scale |
| `seed` | -1 (random) | Set fixed seed for reproducible outputs |
| `height` / `width` | 512 / 512 | Must match preprocessing `TARGET_SIZE` |
| `negative_prompt` | `"blurry, low quality, distorted"` | Recommended baseline |

---

### 3.3 FP16 Memory Management

All model weights and activations run in `torch.float16` (`FP16`) on CUDA
devices. The following bounds must be observed to prevent OOM on consumer GPUs
(≥6 GB VRAM target).

#### Memory envelope per inference pass

| Component | Approx. VRAM | Notes |
|-----------|-------------|-------|
| SD v1.5 UNet (FP16) | ~2.0 GB | Weights only |
| ControlNet (FP16) | ~1.4 GB | Weights only |
| VAE encoder/decoder (FP16) | ~0.5 GB | Shared enc/dec |
| Attention intermediates | ~0.8 GB | Scales with `num_inference_steps` |
| Batch overhead (bs=1) | ~0.3 GB | Activations, gradients disabled |
| **Total (estimated)** | **~5.0 GB** | Within 6 GB budget |

#### Key implementation guards

```python
# 1. Force FP16 on CUDA; fallback to FP32 on CPU
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 2. Enable memory-efficient attention (xformers or sdpa)
pipe.enable_xformers_memory_efficient_attention()   # preferred
# or: pipe.enable_attention_slicing(slice_size="auto")   # fallback

# 3. VAE slicing for large batches or high-res targets
pipe.enable_vae_slicing()

# 4. CPU offload for extremely constrained environments (<5 GB)
pipe.enable_model_cpu_offload()   # moves modules to CPU when not in use

# 5. Clear CUDA cache between pipeline runs
torch.cuda.empty_cache()

# 6. Hard OOM guard — reject requests exceeding safe batch size
MAX_BATCH_SIZE = 1   # enforced in engine.ai_engine.generate()
```

#### FP16 numerical stability

- Latent scaling factor: divide latents by `0.18215` before `vae.decode()`.
  Skipping this step causes saturation in FP16 and produces washed-out images.
- Clip activations to `[-32768, 32767]` (FP16 range) when custom schedulers
  produce out-of-range noise estimates.
- Mixed precision: keep text encoder (`CLIPTextModel`) in FP32 to avoid
  embedding drift; set `text_encoder.to(torch.float32)` explicitly.

---

### 3.4 Trimesh Non-Manifold Recovery

Raw marching-cubes output frequently contains non-manifold edges (edges shared
by ≥ 3 faces), duplicate vertices, and small disconnected components. The
following repair sequence is applied in `engine.mesh_utils.repair_mesh()`
(Mona / Laxmi shared module — to be created in Phase 2).

#### Repair sequence

```python
import trimesh

def repair_mesh(mesh: trimesh.Trimesh, *, verbose: bool = False) -> trimesh.Trimesh:
    """
    Apply a deterministic, idempotent repair sequence to a raw trimesh.
    Returns a new Trimesh; input is not modified.
    """
    # Step 1 — Remove duplicate and degenerate faces
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()

    # Step 2 — Merge vertices within floating-point tolerance
    mesh.merge_vertices(merge_tex=True, merge_norm=True)

    # Step 3 — Fill holes (up to 1000-edge boundary loops)
    trimesh.repair.fill_holes(mesh)

    # Step 4 — Fix face winding for consistent outward normals
    trimesh.repair.fix_winding(mesh)

    # Step 5 — Fix inward-facing normals
    trimesh.repair.fix_inversion(mesh, multibody=True)

    # Step 6 — Remove degenerate components (< 50 faces)
    components = mesh.split(only_watertight=False)
    largest    = max(components, key=lambda c: len(c.faces))
    mesh       = largest

    # Step 7 — Final manifold assertion
    if not mesh.is_watertight:
        # Attempt convex hull as last resort for topology recovery
        mesh = mesh.convex_hull
        if verbose:
            print("repair_mesh: fallback to convex hull — topology unrecoverable")

    return mesh
```

#### Tolerance parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Vertex merge tolerance | `1e-8` (trimesh default) | Prevents false merging of close but distinct vertices |
| Hole fill max boundary edges | `1000` | Limits fill complexity; larger holes signal bad voxelisation |
| Min component face count | `50` | Removes marching-cubes noise islands |
| Convex hull fallback | When `not mesh.is_watertight` after all steps | Guarantees a valid manifold for export |

---

## 4. Component Integration Contracts

### 4.1 Contract A — Preprocessing → AI Engine

**Owner:** Puneeth (producer) · Mona (consumer)

```python
# ── Producer (Puneeth) ───────────────────────────────────────────────────────

from engine.preprocessing import preprocess

# Colour conditioning tensor
colour_tensor: torch.Tensor = preprocess(
    source,                         # ImageInput
    return_tensor=True,             # → torch.Tensor
    return_lineart=False,
)
# shape: (3, 512, 512), dtype: float32, range: [0.0, 1.0]

# ControlNet scribble / lineart tensor
lineart_tensor: torch.Tensor = preprocess(
    source,
    return_tensor=True,
    return_lineart=True,
)
# shape: (1, 512, 512), dtype: float32, values: {0.0, 1.0}


# ── Consumer (Mona) ──────────────────────────────────────────────────────────

# Mona's generate() must accept exactly these tensor shapes.
# Batching: unsqueeze(0) to add batch dim before feeding to pipeline.
colour_batch   = colour_tensor.unsqueeze(0)    # (1, 3, 512, 512)
lineart_batch  = lineart_tensor.unsqueeze(0)   # (1, 1, 512, 512)
```

**Invariants:**
- `colour_tensor.shape  == (3, 512, 512)`
- `colour_tensor.dtype  == torch.float32`
- `lineart_tensor.shape == (1, 512, 512)`
- `lineart_tensor.max() <= 1.0` and `lineart_tensor.min() >= 0.0`
- Both tensors on `device="cpu"` at contract boundary; Mona's engine moves to GPU.

---

### 4.2 Contract B — AI Engine → UI Layer

**Owner:** Mona (producer) · Laxmi (consumer)

```python
# ── Producer (Mona) — engine/ai_engine.py (to be created) ───────────────────

from dataclasses import dataclass
from typing import Optional
import numpy as np
import trimesh

@dataclass
class GenerationResult:
    """
    Return type for engine.ai_engine.generate().
    All fields are on CPU and ready for UI consumption.
    """
    depth_map:   np.ndarray          # shape (512, 512), float32, range [0, 1]
    normal_map:  Optional[np.ndarray]# shape (512, 512, 3), float32, range [0, 1]
    mesh:        trimesh.Trimesh     # watertight after §3.4 repair
    seed_used:   int                 # actual seed (important when seed=-1)
    vram_peak_mb: float              # peak VRAM during inference (monitoring)


# ── Consumer (Laxmi) — ui/app.py ────────────────────────────────────────────

def _run_ai_generation(...) -> Tuple[Optional[Image.Image], str]:
    result: GenerationResult = engine.ai_engine.generate(...)

    # Display depth map
    depth_pil = Image.fromarray(
        (result.depth_map * 255).astype(np.uint8)
    ).convert("L")

    # Hand mesh to session state for export
    _SESSION["current_mesh"] = result.mesh

    status = (
        f"✔  Generation complete — seed={result.seed_used}  "
        f"VRAM peak={result.vram_peak_mb:.0f} MB"
    )
    return depth_pil, status
```

**Invariants:**
- `result.depth_map.shape  == (512, 512)` float32
- `result.mesh.is_watertight == True` (post-repair guarantee)
- `result.seed_used` is always a concrete integer (never -1)
- `generate()` raises `RuntimeError` (not `OOM`) on VRAM exhaustion after
  attempting `enable_model_cpu_offload()` fallback

---

### 4.3 Contract C — UI Layer → Mesh Exporter

**Owner:** Laxmi (orchestrator)

```python
# ── Export handler — ui/app.py ────────────────────────────────────────────────

SUPPORTED_EXPORT_FORMATS = ("OBJ", "GLB", "STL", "PLY")

def _run_mesh_export(export_format: str) -> Tuple[Optional[str], str]:
    mesh: trimesh.Trimesh = _SESSION.get("current_mesh")
    if mesh is None:
        return None, "⚠  No mesh in session. Run generation first."

    out_path = Path("outputs") / f"sketch2shape_export.{export_format.lower()}"
    out_path.parent.mkdir(exist_ok=True)

    mesh.export(str(out_path))           # trimesh handles format dispatch
    return str(out_path), f"✔  Exported {export_format} → {out_path.name}"
```

**Format encoding details:**

| Format | Encoding | Notes |
|--------|----------|-------|
| OBJ | UTF-8 text + MTL sidecar | Widest tool support; no binary overhead |
| GLB | Binary glTF 2.0 | Self-contained; preferred for WebGL / Three.js |
| STL | Binary STL | Slicer-compatible; no vertex colour support |
| PLY | Binary PLY | Preserves vertex colours and normals |

---

## 5. Error Propagation Model

All stages return typed exceptions that propagate upward to the Gradio status
bar without crashing the server process.

```
Stage              Exception type          UI outcome
─────────────────────────────────────────────────────────────────────
ingest()           FileNotFoundError       "✘ File not found: …"
                   ValueError              "✘ Unsupported format: …"
to_lineart()       ValueError (block_size) "✘ Invalid parameter: …"
generate()         RuntimeError (OOM)      "✘ VRAM exhausted — reduce steps"
                   RuntimeError (model)    "✘ Model not loaded: …"
repair_mesh()      (none raised)           Convex hull fallback; logged
export()           OSError                 "✘ Export failed: …"
```

The UI layer wraps every engine call in `try / except Exception`:

```python
try:
    result = engine_call(...)
except Exception as exc:
    logger.error("Stage failed: %s", exc, exc_info=True)
    return None, f"✘  {type(exc).__name__}: {exc}"
```

---

## 6. Dependency & Runtime Matrix

| Package | Version pin | Runtime role |
|---------|------------|--------------|
| `gradio` | `≥4.20.0` | UI framework (Laxmi) |
| `torch` | `≥2.1.0` | Tensor ops, CUDA backend |
| `torchvision` | `≥0.16.0` | Image transforms |
| `diffusers` | `≥0.27.0` | SD pipeline + ControlNet (Mona) |
| `transformers` | `≥4.38.0` | CLIP encoder, depth head (Mona) |
| `accelerate` | `≥0.28.0` | Model offload, mixed precision |
| `opencv-python` | `≥4.8.0` | Bilateral filter, adaptive threshold (Puneeth) |
| `Pillow` | `≥10.0.0` | Image I/O throughout |
| `numpy` | `≥1.24,<2.0` | Array backbone |
| `trimesh` | `≥4.0.0` | Mesh manipulation, repair, export (Laxmi + Mona) |
| `scipy` | `≥1.11.0` | trimesh marching cubes backend |
| `rembg` | `≥2.0.50` | Background removal (optional preprocessing step) |
| `huggingface-hub` | `≥0.22.0` | Model weight retrieval |

**Minimum hardware:**
- GPU: NVIDIA with ≥6 GB VRAM (CUDA 11.8 or 12.1)
- RAM: ≥16 GB system RAM
- Disk: ≥20 GB for model weights cache

**CPU-only fallback:**
All pipeline stages support CPU execution; inference is ~20–40× slower.
FP16 is disabled automatically (`dtype = torch.float32`).

---

*Sketch2Shape AI — TechZephyr 4.0 Hackathon · IIT Bhubaneswar*  
*Document maintained by Laxmi (Lead Frontend, WebGL & Technical Documentation)*
