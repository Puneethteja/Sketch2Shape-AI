# Sketch2Shape AI

> **Diffusion-Guided 3D Spatial Asset Generation from 2D Hand-Drawn Sketches**  
> **TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> **System Release:** `v1.0.0-final`

---

## 1. Executive Overview & Problem Statement

Transforming conceptual 2D hand-drawn sketches into production-ready 3D assets is a pervasive bottleneck across industrial design, rapid prototyping, computer graphics, and digital fabrication. Conventional 3D modeling pipelines demand steep geometric modeling expertise, proprietary CAD suites, and hours of manual topology retopology.

**Sketch2Shape AI** bridges the ideation-to-geometry divide through an accelerated, diffusion-guided generative pipeline. The system ingests arbitrary hand-drawn raster sketches or freehand stylus strokes directly in the browser and constructs clean, watertight, textured 3D meshes in under 22 seconds on standard commodity GPU hardware (such as an NVIDIA T4).

---

## 2. Key Architectural Highlights

```
┌──────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐     ┌──────────────────────┐
│  Stage 1: Ingest │ ──> │ Stage 2: 2.5D Diffuse │ ──> │  Stage 3: Tensor Link  │ ──> │ Stage 4: Mesh Repair │
│  & Denoise       │     │ SD 1.5 + ControlNet   │     │  & InstantMesh 3D LRM  │     │ 7-Step Non-Manifold  │
└──────────────────┘     └───────────────────────┘     └────────────────────────┘     └──────────────────────┘
         │                           │                              │                             │
         ▼                           ▼                              ▼                             ▼
Universal Normalization      UniPC (20 steps, FP16)         Depth Priors & Shading        Watertight Topology
Bilateral Filter (d=9)       CFG 7.0, Scale 0.80            Direct Inter-Stage Tensors    Binary GLB & OBJ Output
```

1. **Universal Preprocessing & Foreground Isolation (`engine/preprocessing.py`, `utils/segmentation.py`)**:
   - Ingests raw drawing canvas base64 payloads, file uploads (`.png`, `.jpg`, `.jpeg`), and raw image bytes.
   - Normalizes inputs to $512 \times 512$ letterboxed representations with bilateral edge-preserving smoothing and U²-Net/IS-Net salient foreground masking.

2. **Sketch-Conditioned 2.5D Diffusion Engine (`engine/generator.py`)**:
   - Leverages Stable Diffusion v1.5 conditioned via ControlNet Scribble adapters operating in FP16 half-precision.
   - Employs calibrated Classifier-Free Guidance (CFG: 7.0) and ControlNet conditioning scale (0.80) with a fast 20-step UniPC scheduler to maximize contour fidelity and eliminate halo artifacts.

3. **Direct Tensor-Level Linkage & 3D Reconstruction (`engine/linker.py`, `engine/reconstruction.py`)**:
   - Injects generator-derived depth priors, Lambertian shading decomposition maps, and foreground masks directly into the 3D Large Reconstruction Model (InstantMesh LRM) via `StageLinker`.
   - Bypasses raw pixel re-estimation, ensuring reconstructed surface geometry faithfully adheres to the original sketch silhouette.

4. **Strict Sequential VRAM Evacuation Handover (`engine/pipeline.py`)**:
   - Enforces automated model CPU offloading (`model.to("cpu")`), garbage collection (`gc.collect()`), and CUDA cache evacuation (`torch.cuda.empty_cache()`) between pipeline stages.
   - Caps peak concurrent VRAM allocation at **6.50 GB**, eliminating Out-Of-Memory (OOM) failures on 8GB and 16GB GPUs.

5. **Deterministic Watertight Mesh Repair (`utils/mesh_utils.py`)**:
   - Implements a rigorous 7-step non-manifold recovery pipeline: eliminates degenerate zero-area faces, unifies winding order, repairs inverted normals, seals boundary loop holes, prunes unattached floating fragments, and guarantees closed manifold topology.

6. **Utility-Focused WebGL 3D Previewer (`ui/viewer.js`, `ui/app.py`)**:
   - Client-side Three.js viewport featuring 360-degree orbital rotation, pan, zoom, dynamic lighting, real-time loading progress bar, asset reset controls, and instant binary `.glb` downloads.
   - Resilient HTML5 drawing canvas equipped with pen pressure curve smoothing, eraser, clear canvas, and undo/redo stacks.

7. **Offline Demo Resilience & Backup Suite (`utils/backup_runner.py`)**:
   - Bundles verified pre-computed sample assets (`outputs/samples/`) across multiple semantic categories with instant zero-latency loading for offline hackathon evaluations.

---

## 3. Repository Architecture

```
Sketch2Shape-AI/
├── README.md                         Primary evaluation overview and execution manual
├── requirements.txt                  pip package dependency specifications
├── environment.yml                   Conda environment specification (CUDA-enabled)
├── check_gpu.py                      Diagnostic tool for CUDA, VRAM, and FP16 validation
├── docs/                             Technical documentation suite
│   ├── technical_architecture.md     System dataflow, contracts, and algorithmic specs
│   ├── technical_report.md           Formal performance benchmarks and hardware analysis
│   └── living_log.md                 Chronological team engineering logs (Phases 1-5)
├── engine/                           Core generative inference pipeline
│   ├── ai_engine.py                  Public API facade for UI integration
│   ├── pipeline.py                   Unified orchestrator with VRAM handover
│   ├── generator.py                  Stable Diffusion + ControlNet scribble pipeline
│   ├── linker.py                     StageLinker inter-stage tensor handoff bundle
│   ├── reconstruction.py             InstantMesh 3D Large Reconstruction Model
│   ├── mesh_utils.py                  Watertight repair and multi-format exporters
│   ├── preprocessing.py              Universal canvas and file ingestion routines
│   └── checkpoints/                  Model weight scaffolding and HF downloaders
├── ui/                               Frontend presentation layer
│   ├── app.py                        Gradio dual-viewport interface and error bounds
│   └── viewer.js                     Embedded Three.js WebGL 3D preview component
├── utils/                            Shared infrastructure utilities
│   ├── mesh_utils.py                 Hardened mesh repair and topology sanitation
│   ├── segmentation.py               Rembg background isolation and centering
│   ├── image_utils.py                Color-safe image I/O and visualization helpers
│   └── backup_runner.py              Offline demo asset loader and integrity verifier
├── outputs/                          Generated artifacts and sample suite
│   ├── samples/                      Curated pre-computed demo assets and manifest
│   │   ├── manifest.json             Structured metadata and geometric metrics
│   │   ├── sample_01_vase/           Vase sketch, lineart, GLB, and OBJ bundle
│   │   ├── sample_02_mug/            Mug sketch, lineart, GLB, and OBJ bundle
│   │   ├── sample_03_chair/          Chair sketch, lineart, GLB, and OBJ bundle
│   │   ├── sample_04_bottle/         Bottle sketch, lineart, GLB, and OBJ bundle
│   │   └── sample_05_bowl/           Bowl sketch, lineart, GLB, and OBJ bundle
│   └── preview_*.glb                 Session exports for active WebGL inspection
└── tests/                            Automated pytest verification suite
    ├── test_mesh_stress.py           Stress tests for noisy sketches and malformed meshes
    ├── test_integration.py          End-to-end multi-stage pipeline integration tests
    ├── test_ai_inference.py          Diffusion and 3D reconstruction contract tests
    ├── test_mesh_utils.py            Watertight assertion and non-manifold repair tests
    ├── test_segmentation.py          Foreground isolation and bounding box tests
    ├── test_preprocessing.py         Canvas ingestion, letterboxing, and lineart tests
    ├── test_gpu_and_pipeline.py      Hardware diagnostics and VRAM checks
    └── test_ui.py                    UI component layout and viewer script verification
```

---

## 🚀 Running the Web Interface

### Launch the Single Unified SPA Web Studio & Generative Backend

\\ash
python app.py --port 8000
\
- **Web Studio (SPA)**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive REST API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (OpenAPI / Swagger).
- **Custom Port/Host**: python app.py --port 8080 --host 0.0.0.0
---

## 4. Environment Prerequisites & Installation

### Hardware Requirements

| Tier | GPU Specification | Target VRAM | Expected End-to-End Latency | Mode |
|---|---|:---:|:---:|---|
| **Optimal** | NVIDIA V100 / A100 | $\ge 24\text{ GB}$ | ~11.9s | FP16 Accelerated |
| **Recommended** | NVIDIA T4 / RTX 3060 / RTX 4060 | $\ge 12\text{ GB}$ | ~21.7s | FP16 Sequential Handover |
| **Constrained** | NVIDIA GTX 1660 Ti / RTX 3050 | $6\text{--}8\text{ GB}$ | ~35s | FP16 CPU Model Offload |
| **Fallback** | Multi-Core CPU (x86_64, 8+ cores) | N/A (16GB RAM) | Sub-5s (Mock) / ~128s (Full) | CPU Verification Mode |

### Step-by-Step Installation

#### Option A: Conda Environment (Recommended)

```bash
git clone https://github.com/Puneethteja/Sketch2Shape-AI.git
cd Sketch2Shape-AI
conda env create -f environment.yml
conda activate sketch2shape
```

#### Option B: Standard Python Virtual Environment

```bash
git clone https://github.com/Puneethteja/Sketch2Shape-AI.git
cd Sketch2Shape-AI
python -m venv venv
```

Activate the environment:

On Linux / macOS:
```bash
source venv/bin/activate
```

On Windows (Command Prompt / PowerShell):
```powershell
venv\Scripts\activate
```

Install PyTorch with matching CUDA runtime:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install core dependencies:
```bash
pip install -r requirements.txt
```

Verify hardware and environment configuration:
```bash
python check_gpu.py
```

---

## 5. Judge Evaluation & Quick Start Guide

### 1. Launch the Live Interactive Application

Start the Gradio web interface with the embedded WebGL 3D previewer:

```bash
python ui/app.py
```

To generate a public share link for remote evaluation:

```bash
python ui/app.py --share
```

Once launched, open your web browser at:
```
http://127.0.0.1:7860
```

#### How to Evaluate Live Workflows in the UI:
1. **Draw a Sketch**: Use the interactive HTML5 drawing canvas on the left panel to sketch an object (e.g. vase, cup, bottle, chair). Use the brush slider, color picker, eraser, and undo/redo buttons as needed.
2. **Or Upload an Image**: Drag and drop a `.png` or `.jpg` sketch into the upload zone.
3. **Optional Style Prompt**: Provide material or aesthetic guidance in the prompt box (e.g., *"smooth white ceramic porcelain"* or *"polished oak wood"*).
4. **Generate 3D Asset**: Click **Generate 3D Asset**. Observe the real-time status monitor, intermediate lineart extraction, depth estimation map, and interactive WebGL 3D viewer.
5. **Inspect & Download**: Orbit (left click + drag), pan (right click + drag), and zoom (scroll wheel) in the 3D viewport. Click **Download .GLB** or select export formats (`.obj`, `.stl`, `.ply`) to save binary geometry.

---

### 2. Zero-Latency Offline Evaluation Suite

If evaluating in an environment without discrete GPU acceleration or with limited network bandwidth, utilize the pre-computed artifact runner:

Verify sample artifact integrity:
```bash
python utils/backup_runner.py --verify
```

List available pre-computed samples:
```bash
python utils/backup_runner.py --list
```

Deploy a curated sample directly to the 3D preview viewport:
```bash
python utils/backup_runner.py --load vase
```
*(Accepts sample keywords: `vase`, `mug`, `chair`, `bottle`, `bowl`)*

---

### 3. Run Automated Verification Test Suite

Execute the comprehensive test harness:

```bash
pytest tests/ -v
```

Execute specific subsystem suites:
```bash
pytest tests/test_mesh_stress.py -v
pytest tests/test_integration.py -v
pytest tests/test_ai_inference.py -v
```

**Test Coverage Summary:** 152 automated tests passing with 100% pass rate across preprocessing, AI inference, mesh repair, UI components, and stress testing.

---

## 6. Performance Benchmarks

Empirical performance recorded across 50 consecutive forward passes on standard $512 \times 512$ canvas inputs:

| Metric | Baseline FP32 | Sketch2Shape AI (FP16 Optimized) | Performance Gain |
|---|---|---|:---:|
| **Diffusion Pass Latency (Stage 2)** | 28.40s | 13.15s | **2.16x Faster** |
| **3D Reconstruction Latency (Stage 3)** | 16.82s | 7.62s | **2.21x Faster** |
| **Mesh Repair & GLB Export (Stage 4 & 5)** | 0.82s | 0.45s | **1.82x Faster** |
| **Total End-to-End Latency** | 46.45s | **21.69s** | **2.14x Overall Speedup** |
| **Peak Concurrent VRAM** | 12.85 GB | **6.50 GB** | **49.4% Memory Reduction** |
| **Watertight Manifold Success Rate** | 82.4% | **100.0%** | **Guaranteed Manifold** |

For comprehensive hardware evaluations, latency profiling breakdowns, and failure mode analyses, refer to [`docs/technical_report.md`](docs/technical_report.md).

---

## 7. Project Engineering Team

| Team Member | Engineering Role & Responsibilities | Key Deliverables |
|---|---|---|
| **Puneeth** | Lead Data & Infrastructure Engineer | Universal preprocessing pipeline (`engine/preprocessing.py`), background segmentation (`utils/segmentation.py`), hardened mesh repair (`utils/mesh_utils.py`), stress testing suite (`tests/test_mesh_stress.py`), sample curation, and offline backup runner (`utils/backup_runner.py`). |
| **Mona** | Lead Generative AI & 3D Reconstruction Engineer | Generative diffusion pipeline (`engine/generator.py`), InstantMesh 3D LRM integration (`engine/reconstruction.py`), StageLinker tensor handoff (`engine/linker.py`), sequential VRAM evacuation (`engine/pipeline.py`), and formal evaluation report (`docs/technical_report.md`). |
| **Laxmi** | Lead Frontend, WebGL & Technical Documentation Engineer | Dual-viewport interface (`ui/app.py`), HTML5 drawing canvas with Bézier smoothing, Three.js WebGL 3D previewer (`ui/viewer.js`), error boundaries, technical architecture specification (`docs/technical_architecture.md`), and evaluation packaging (`README.md`). |

---

## 8. Technical Documentation References

- **Technical Architecture & Contract Specification**: [`docs/technical_architecture.md`](docs/technical_architecture.md) — Comprehensive technical blueprint detailing mathematical formulations, multi-stage tensor contracts, coordinate frames, and 7-step repair sequence.
- **Formal Evaluation & Hardware Report**: [`docs/technical_report.md`](docs/technical_report.md) — Research-grade report covering empirical latency profiles, VRAM trajectories, precision trade-offs, and future scaling vectors.
- **Chronological Engineering Living Log**: [`docs/living_log.md`](docs/living_log.md) — Verified milestone trajectory documenting development progress across all project phases (Phases 1 through 5).

---

*TechZephyr 4.0 Hackathon · IIT Bhubaneswar*
