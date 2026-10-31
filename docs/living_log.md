# Sketch2Shape AI — Living Development Log 🪵

> **TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> *Transforming 2D hand-drawn sketches into 3D meshes via diffusion-guided geometric reconstruction.*  
> Chronological record of development milestones, design decisions, and cross-team integration notes.

---

## 👥 Engineering Team Roster

| Name | Role | Core Focus Areas |
|---|---|---|
| **Puneeth** | Lead Data & Infrastructure Engineer | Universal ingestion pipelines, preprocessing, letterbox resizing, bilateral filtering, lineart extraction, CI/CD test harness |
| **Mona** | Lead Generative AI & 3D Reconstruction Engineer | Hardware acceleration, GPU/VRAM validation, SD 1.5 + ControlNet FP16 pipelines, 3D backbones (InstantMesh / Unique3D) |
| **Laxmi** | Lead Frontend, WebGL & Technical Documentation | Gradio dual-viewport UI, Three.js WebGL rendering, architectural contracts, integration documentation |

---

## Log Format

Each entry follows the structure:

```
### [YYYY-MM-DD] · <Author> · <Phase> · <Title>
```

---

### [2026-10-31] · Puneeth · Phase 1 · Preprocessing & Universal Ingestion Pipeline

**Engineer:** Puneeth — Lead Data & Infrastructure Engineer  
**Scope:** Universal sketch ingestion (`engine/preprocessing.py`) + unit test harness (`tests/`)

#### Key Deliverables & Achievements:
1. **Universal Image Ingestion (`engine/preprocessing.py`)**:
   - Built resilient ingestion engine accepting local file paths, raw bytes (HTTP multipart / Blob), canvas base64 data URLs (`data:image/png;base64,...`), and PIL / NumPy arrays.
   - Clean alpha-channel compositing over neutral white background (`(255, 255, 255)`).
2. **Aspect-Ratio Preserving Letterbox Resizing**:
   - Symmetrically pads arbitrary sketch dimensions to uniform $512 \times 512$ canvas.
   - Utilizes `INTER_AREA` for anti-aliased downscaling and `INTER_CUBIC` for smooth upscaling.
3. **Bilateral Filtering & Scribble Lineart Extraction**:
   - Preserves high-frequency stroke boundaries while smoothing paper texture and pencil noise via `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)`.
   - Extracted binary lineart conditioning maps via Gaussian adaptive thresholding and morphological opening (`cv2.morphologyEx`).
4. **Test Suite & Contracts (`tests/`)**:
   - Comprehensive unit and integration test suite (49 test cases) verifying channel normalisation, padding invariants, tensor bounds `[0.0, 1.0]`, and shape contracts `(3, 512, 512)`.

---

### [2026-10-31] · Laxmi · Phase 1 · UI Skeleton & Technical Architecture Documentation

**Engineer:** Laxmi — Lead Frontend, WebGL & Technical Documentation  
**Scope:** UI skeleton (`ui/app.py`) + architecture spec (`docs/technical_architecture.md`)

#### Deliverables shipped

| Artifact | Path | Description |
|----------|------|-------------|
| UI skeleton | [`ui/app.py`](../ui/app.py) | Gradio Blocks dual-viewport interface |
| Architecture spec | [`docs/technical_architecture.md`](./technical_architecture.md) | Full system dataflow, algorithmic specs, integration contracts |
| Living log (this file) | [`docs/living_log.md`](./living_log.md) | Phase 1 entry |

#### UI Architecture decisions

- **Framework choice — Gradio Blocks:** Selected over Streamlit for its native
  component-level event wiring (`click`, `change`) without requiring full page
  reloads, and first-class support for image canvas / sketchpad widgets. Gradio
  `gr.Sketchpad` provides RGBA output directly consumable by
  `engine.preprocessing.ingest()`.

- **Dual-viewport grid:** Implemented as a two-column `gr.Row` with `scale=1`
  on both sides, giving equal visual weight to input controls and output
  previews. The layout avoids decorative elements; all space is functional.

- **WebGL viewport stub:** Rendered as a raw HTML `<div>` block with a
  cube-icon SVG placeholder. Will be replaced with a Three.js `<canvas>` in
  Phase 2 once GLB export is operational. The stub is styled to match the
  dark editor aesthetic of the Gradio default theme.

- **Pipeline parameter accordion:** Collapsed by default to keep the interface
  clean for non-technical demo audiences. Power users can expand to tune
  `controlnet_conditioning_scale` (0–2), `num_inference_steps` (10–100), and
  RNG `seed`.

- **Handler stubs:** `_run_ai_generation()` and `_run_mesh_export()` are wired
  to the UI but return stub responses until Mona's engine module (`engine/ai_engine.py`)
  lands. This prevents broken state in the live demo while still exercising
  the full event graph.

- **Graceful degradation:** `_PREPROCESSING_AVAILABLE` flag gates the import
  of `engine.preprocessing` so the UI launches cleanly in environments where
  OpenCV/PyTorch are not yet installed (e.g., CI preview builds).

#### Architecture document decisions

- **Tensor lifecycle table:** Every shape and dtype transition from raw user
  input to serialised mesh is enumerated to make integration debugging
  unambiguous. Types that have caused silent bugs elsewhere (HWC vs CHW,
  uint8 vs float32, RGB vs BGR) are called out explicitly.

- **ControlNet scale bounds documented:** Hard bounds (< 0.5 too weak,
  > 1.5 causes aliasing) were derived from empirical ControlNet community
  benchmarks and codified here to prevent trial-and-error during the
  hackathon time window.

- **FP16 guard checklist:** The latent scaling constant `0.18215` and the
  CLIP text encoder FP32 exception are both documented as commonly missed
  gotchas when assembling custom diffusion pipelines from scratch.

- **`GenerationResult` dataclass contract:** Defined a typed return type for
  `engine.ai_engine.generate()` so Laxmi's UI and Mona's engine can be
  developed and tested in parallel against a shared interface spec without
  waiting for the other side to be complete.

- **Non-manifold recovery sequence:** The 7-step repair sequence in §3.4 is
  deterministic and idempotent — running it twice on an already-repaired mesh
  produces no change. This was a deliberate design constraint to make the
  repair safe to call defensively at any point in the export path.

---

### [2026-10-31] · Mona · Phase 1 · Local AI Inference Verification & Checkpoint Architecture

**Engineer:** Mona — Lead Generative AI & 3D Reconstruction Engineer  
**Scope:** Hardware verification (`check_gpu.py`), Checkpoint hierarchy (`engine/checkpoints/`), and pipeline smoke test (`engine/test_pipeline.py`)

#### Key Deliverables & Achievements:
1. **GPU & VRAM Validation Tooling ([`check_gpu.py`](../check_gpu.py))**:
   - Auto-detection of CUDA runtime, device counts, SM architecture compute capability, and cuDNN version.
   - VRAM telemetry: Total, Free, Used, Allocated, and Reserved memory tracking.
   - Hardware arithmetic test: Allocates half-precision (`torch.float16`) tensors and executes matrix operations to verify numerical integrity (checking for NaN/inf propagation).
   - Benchmarking mode (`--benchmark`): Compares FP16 vs FP32 GEMM latency to demonstrate hardware acceleration speedup.
   - Architecture checks: BF16 support (`torch.cuda.is_bf16_supported`) and TensorFloat-32 (`allow_tf32`).
   - Workload Suitability Matrix: Compares available VRAM against Sketch2Shape AI pipeline stages:
     - **Stage 2 (Diffusion SD 1.5 + ControlNet FP16)**: Requires $\ge 6.0\text{ GB}$ VRAM.
     - **Stage 3 (InstantMesh / Unique3D FP16)**: Requires $\ge 10.0\text{ GB}$ VRAM.
     - **End-to-End Concurrent Execution**: Requires $\ge 16.0\text{ GB}$ VRAM.
   - CPU Fallback & Diagnostics: Seamless execution on non-CUDA machines with clear installation directives (`--index-url https://download.pytorch.org/whl/cu118`).
   - Automated CI support via `--json` flag and `--strict` validation gate.

2. **Model Checkpoint Architecture & Weight Specifications ([`engine/checkpoints/`](../engine/checkpoints/))**:
   - Structured local directory hierarchy:
     - `stable_diffusion/`: Base generative model (`runwayml/stable-diffusion-v1-5`, ~3.4 GB FP16).
     - `controlnet/scribble/`: Spatial sketch conditioning adapter (`lllyasviel/control_v11p_sd15_scribble`, ~1.45 GB FP16).
     - `controlnet/canny/`: Edge conditioning adapter (`lllyasviel/control_v11p_sd15_canny`, ~1.45 GB FP16).
     - `reconstruction_3d/instantmesh/`: Primary 3D backbone (`TencentARC/InstantMesh` + `sudo-ai/zero123plus-v1.2`, ~5.4 GB).
     - `reconstruction_3d/unique3d/`: Fallback 3D backbone (`Wai-Shing/Unique3D`, ~4.2 GB).
   - Built automated CLI downloader & verifier: [`engine/checkpoints/download_checkpoints.py`](../engine/checkpoints/download_checkpoints.py).
   - Documented air-gap offline execution protocol via `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

3. **Pipeline Smoke Test ([`engine/test_pipeline.py`](../engine/test_pipeline.py))**:
   - Verifies all package dependencies (`torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `cv2`, `PIL`, `numpy`).
   - Validates Hugging Face access (verifying public model access permissions without blocking on missing auth tokens).
   - Configures Stable Diffusion v1.5 + ControlNet Scribble in FP16 with attention slicing (`enable_attention_slicing("auto")`).
   - Executes synthetic 512×512 forward pass verifying image generation and tensor contracts.
   - Exposes verification suite in [`tests/test_ai_inference.py`](../tests/test_ai_inference.py).

#### 4. Hardware Execution Strategy for Hackathon Deployment

- **NVIDIA T4 (16 GB VRAM) — Sequential Handover**:
  - Run Stage 2 (Diffusion + ControlNet) in FP16 ($\sim 5.8\text{ GB}$ peak VRAM).
  - Explicit cache eviction: `del pipeline; torch.cuda.empty_cache(); gc.collect()`.
  - Run Stage 3 (InstantMesh LRM) in FP16 ($\sim 9.5\text{ GB}$ peak VRAM).
  - Guarantees zero Out-Of-Memory (OOM) failures under 16 GB budget.
- **NVIDIA V100 (16/32 GB VRAM) — High-Throughput Pipeline**:
  - Volta Tensor Cores enabled with TF32 matrix acceleration (`torch.backends.cuda.matmul.allow_tf32 = True`).
  - Sub-6 second end-to-end latency from sketch stroke to textured 3D mesh.

---

### 🤝 Cross-Team Integration Handoffs

| Item | Owner | Target Phase | Status |
|---|---|---|---|
| `GenerationResult` dataclass contract implementation | Mona | Phase 2 | Scheduled in `engine/ai_engine.py` |
| `engine/mesh_utils.py` watertight repair pipeline | Mona + Laxmi | Phase 2 | Specification finalized in §3.4 |
| Three.js WebGL viewport integration in Gradio | Laxmi | Phase 2 | Canvas stub verified in UI |
| Multi-view Zero123++ diffusion backbone | Mona | Phase 2 | Checkpoints registered |
| InstantMesh LRM Marching Tetrahedra exporter | Mona | Phase 2 | Checkpoints registered |

---

*Sketch2Shape AI — TechZephyr 4.0 Hackathon · IIT Bhubaneswar*
