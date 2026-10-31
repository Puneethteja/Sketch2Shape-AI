# Engine Module Specification ⚙️

> **Sketch2Shape AI · TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> **Engineering Leads**: Puneeth (Data & Infrastructure) & Mona (Generative AI & 3D Reconstruction)

The `engine` package forms the core computational pipeline for Sketch2Shape AI, transforming 2D hand-drawn input sketches into production-ready 3D mesh assets.

---

## 🗂️ Module Architecture

```
engine/
├── __init__.py
├── preprocessing.py               # Phase 1: Ingestion, Letterboxing, Bilateral Filtering, Lineart (Puneeth)
├── test_pipeline.py               # Phase 1: SD 1.5 + ControlNet FP16 Smoke Test & Verification (Mona)
├── checkpoints/                   # Model weights & checkpoint management (Mona)
│   ├── README.md                  # Checkpoint architecture & weight specifications
│   ├── download_checkpoints.py    # Automated Hugging Face download utility
│   ├── stable_diffusion/          # Base diffusion pipeline (runwayml/stable-diffusion-v1-5)
│   ├── controlnet/                # ControlNet adapters (scribble & canny)
│   └── reconstruction_3d/         # 3D backbones (InstantMesh LRM & Unique3D)
```

---

## 🔄 Computational Pipeline Flow

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
           ▼ (Clean 512x512 RGB & Binary Lineart Conditioning)
+---------------------------------------------------------+
| engine/test_pipeline.py & Diffusion Pipeline            |
| - Stable Diffusion v1.5 + ControlNet Scribble           |
| - FP16 Half-Precision on NVIDIA GPU (T4 / V100)          |
| - Memory-efficient attention (SDPA / xFormers)          |
| - Multi-view 2.5D asset generation                      |
+---------------------------------------------------------+
           │
           ▼ (Multi-view Normal & Color Projections)
+---------------------------------------------------------+
| 3D Reconstruction Backbone (InstantMesh / Unique3D)     |
| - Large Reconstruction Model (LRM) Transformer          |
| - Instant triplane / NeRF representation                |
| - Marching Tetrahedra mesh extraction                   |
| -> outputs/asset.obj / outputs/asset.glb                |
+---------------------------------------------------------+
```

---

## 🚀 Execution Commands

```bash
# 1. Verify GPU, VRAM, and FP16 half-precision support
python check_gpu.py

# 2. Inspect checkpoint structure and verify local model weights
python engine/checkpoints/download_checkpoints.py --verify

# 3. Run Pipeline Smoke Test (verifies dependencies, auth, and pipeline loading)
python engine/test_pipeline.py
```
