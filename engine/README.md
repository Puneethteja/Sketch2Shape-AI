# Engine Module Specification ⚙️

> **Sketch2Shape AI · TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> **Engineering Leads**: Puneeth (Data & Infrastructure) & Mona (Generative AI & 3D Reconstruction)

The `engine` package forms the core computational pipeline for Sketch2Shape AI, transforming 2D hand-drawn input sketches into production-ready watertight 3D mesh assets.

---

## 🗂️ Module Architecture

```
engine/
├── __init__.py
├── preprocessing.py               # Phase 1: Ingestion, Letterboxing, Bilateral Filtering, Lineart (Puneeth)
├── test_pipeline.py               # Phase 1: SD 1.5 + ControlNet FP16 Smoke Test & Verification (Mona)
├── generator.py                   # Phase 2: Sketch-guided SD 1.5 + ControlNet Scribble Generator (Mona)
├── reconstruction.py              # Phase 2: Deep learning 3D reconstruction (InstantMesh / Unique3D) (Mona)
├── mesh_utils.py                  # Phase 2: 7-step watertight repair (§3.4) & multi-format serializers (Mona)
├── pipeline.py                    # Phase 2: Unified orchestrator with sequential VRAM handover (Mona)
├── ai_engine.py                   # Phase 2: Public API & Contract B facade for Gradio UI layer (Mona)
├── checkpoints/                   # Model weights & checkpoint management (Mona)
│   ├── README.md                  # Checkpoint architecture & weight specifications
│   ├── download_checkpoints.py    # Automated Hugging Face download utility
│   ├── stable_diffusion/          # Base diffusion pipeline (runwayml/stable-diffusion-v1-5)
│   ├── controlnet/                # ControlNet adapters (scribble & canny)
│   └── reconstruction_3d/         # 3D backbones (InstantMesh LRM & Unique3D)
```

---

## 🔄 Computational Pipeline Flow & Sequential VRAM Handover

```
+------------------------+
| 1. User Hand Sketch    |  (Canvas Base64 / File / Raw Bytes)
+------------------------+
           │
           ▼
+---------------------------------------------------------+
| engine/preprocessing.py                                 |
| - Ingestion & Alpha compositing                         |
| - 512x512 Letterbox padding                             |
| - Bilateral edge-preserving filtering                   |
| - Gaussian adaptive thresholding -> Lineart Mask        |
+---------------------------------------------------------+
           │
           ▼ (Contract A: 512x512 Lineart & Color Tensors)
+---------------------------------------------------------+
| engine/generator.py (SketchDiffusionGenerator)          |
| - Stable Diffusion v1.5 + ControlNet Scribble (FP16)    |
| - Prompt conditioning + 3D quality booster              |
| - Calibrated scale (0.85) & CFG (7.5)                   |
| - Peak VRAM: ~5.2 GB                                    |
+---------------------------------------------------------+
           │
           ▼ [SEQUENTIAL VRAM HANDOVER]
           │ Evacuate generator weights from VRAM
           │ torch.cuda.empty_cache(); gc.collect()
           │
+---------------------------------------------------------+
| engine/reconstruction.py (MeshReconstructor)            |
| - Depth & surface normal estimation                     |
| - Pretrained InstantMesh / Unique3D backbone (FP16)     |
| - Peak VRAM: ~6.5 GB                                    |
+---------------------------------------------------------+
           │
           ▼
+---------------------------------------------------------+
| engine/mesh_utils.py (Watertight Manifold Repair §3.4)  |
| - Duplicate/degenerate face removal                     |
| - Vertex merging within tolerance                       |
| - Boundary loop hole-filling                            |
| - Normal winding and inversion fixing                   |
| - Guaranteed is_watertight == True                      |
+---------------------------------------------------------+
           │
           ▼ (Contract B: GenerationResult to UI & Session)
+---------------------------------------------------------+
| Mesh Exporters (OBJ, GLB, STL, PLY)                     |
| -> outputs/sketch2shape_model.obj / .glb                |
+---------------------------------------------------------+
```

---

## 🚀 Execution Commands

```bash
# 1. Run full end-to-end inference pipeline from CLI
python engine/pipeline.py path/to/sketch.png --prompt "ceramic coffee mug" --output outputs/mug.glb --format glb

# 2. Run automated test suite verifying all 16 pipeline contracts
python -m unittest tests/test_ai_inference.py -v

# 3. Launch the Gradio UI connected to the live AI engine
python ui/app.py
```
