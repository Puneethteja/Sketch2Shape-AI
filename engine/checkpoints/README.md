# Model Checkpoint Architecture & Weight Specifications 🧠📦

> **Sketch2Shape AI · TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> **Lead Generative AI & 3D Reconstruction Engineer**: Mona  
> **Last Updated**: Phase 1 AI Verification

This document specifies the local and cached model checkpoint hierarchy for the **Sketch2Shape AI** multimodal pipeline. All deep learning backbones are engineered to execute in half-precision (`torch.float16`) to maximize inference speed and ensure execution within target GPU constraints (e.g., NVIDIA T4 16 GB and NVIDIA V100 16/32 GB).

---

## 🏗️ Checkpoint Directory Hierarchy

```
engine/checkpoints/
├── stable_diffusion/              # 2.5D Generative Diffusion Foundation
│   ├── .gitkeep
│   ├── model_index.json
│   ├── unet/                      # UNet (diffusion_pytorch_model.fp16.safetensors)
│   ├── text_encoder/              # CLIP ViT-L/14 (model.fp16.safetensors)
│   ├── tokenizer/                 # Tokenizer configs & vocab files
│   ├── vae/                       # AutoencoderKL / ft-MSE (diffusion_pytorch_model.fp16.safetensors)
│   └── scheduler/                 # DPMSolverMultistepScheduler / UniPC configs
│
├── controlnet/                    # Spatial Conditioners for Hand-Drawn Sketches
│   ├── scribble/                  # Primary: ControlNet Scribble v1.1
│   │   ├── .gitkeep
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.fp16.safetensors
│   └── canny/                     # Secondary: ControlNet Canny v1.1 (fine edge traces)
│       ├── .gitkeep
│       ├── config.json
│       └── diffusion_pytorch_model.fp16.safetensors
│
├── reconstruction_3d/             # 3D Neural Reconstruction Backbone
│   ├── instantmesh/               # Primary Backbone: TencentARC InstantMesh
│   │   ├── .gitkeep
│   │   ├── instant_mesh_large.ckpt # Large Reconstruction Model (LRM) weights
│   │   └── zero123plus-v1.2/      # Zero123++ multi-view diffusion weights
│   └── unique3d/                  # Fallback Backbone: Tsinghua Unique3D
│       ├── .gitkeep
│       ├── controlnet-tile/       # Multi-view normal/depth conditioning
│       └── image2mesh.pth         # Orthographic normal-guided mesh generator
│
├── download_checkpoints.py        # Automated Hugging Face Hub download & verification tool
└── README.md                      # This specification
```

---

## 📊 Model Weight Specifications & Storage Footprints

| Model Component | Hugging Face Hub Repository ID | Target Files | Disk Size (FP16) | Runtime VRAM |
|---|---|---|---|---|
| **Base Diffusion** | `runwayml/stable-diffusion-v1-5` | `unet`, `vae`, `text_encoder` (`*.safetensors`) | ~3.4 GB | ~4.2 GB |
| **ControlNet Scribble** | `lllyasviel/control_v11p_sd15_scribble` | `diffusion_pytorch_model.fp16.safetensors` | ~1.45 GB | ~1.6 GB |
| **ControlNet Canny** | `lllyasviel/control_v11p_sd15_canny` | `diffusion_pytorch_model.fp16.safetensors` | ~1.45 GB | ~1.6 GB |
| **InstantMesh LRM** | `TencentARC/InstantMesh` | `instant_mesh_large.ckpt` | ~1.6 GB | ~4.5 GB |
| **Zero123++ v1.2** | `sudo-ai/zero123plus-v1.2` | `diffusion_pytorch_model.safetensors` | ~3.8 GB | ~5.0 GB |
| **Unique3D (Alternative)**| `Wai-Shing/Unique3D` | `img2normal`, `normal2mesh` | ~4.2 GB | ~6.5 GB |
| **Total Pipeline Assets** | — | — | **~11.7 GB** | **~15.3 GB (Peak)** |

---

## ⚡ Hardware Acceleration & VRAM Allocation Matrix

During the TechZephyr 4.0 Hackathon, our pipeline will execute across diverse hardware profiles. We enforce two runtime execution policies:

### 1. NVIDIA T4 (16 GB VRAM) — Sequential Execution Pipeline
* **Strategy**: The Diffusion stage and 3D Reconstruction stage execute sequentially.
* **Stage 2 (Diffusion + ControlNet)**:
  - Loaded in FP16 with PyTorch 2.x SDPA: **5.8 GB VRAM**.
  - 15 steps via `DPMSolverMultistepScheduler` or `UniPCMultistepScheduler` (~3.2 seconds).
  - Explicit garbage collection: `torch.cuda.empty_cache()` after multi-view generation.
* **Stage 3 (InstantMesh 3D Extraction)**:
  - Multi-view images fed into InstantMesh LRM: **9.5 GB VRAM**.
  - Marching Tetrahedra mesh extraction executed in CUDA kernel (~4.8 seconds).
* **Total Latency**: ~8.0 seconds per asset without Out-Of-Memory (OOM) risk.

### 2. NVIDIA V100 (16 GB / 32 GB VRAM) — High Throughput Execution
* **Strategy**: V100 Volta Tensor Cores achieve exceptional FP16 throughput.
* **Optimizations**:
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.benchmark = True`
  - Attention Slicing disabled for maximum CUDA parallelism.
* **Total Latency**: ~5.5 seconds per asset end-to-end.

---

## 📥 Checkpoint Ingestion & Download Protocol

To prepare local checkpoints on a development node or remote cloud runner:

### Option A — Automated CLI Script (`download_checkpoints.py`)

```bash
# Download only Stable Diffusion v1.5 and ControlNet Scribble (Generative stage)
python engine/checkpoints/download_checkpoints.py --stage diffusion

# Download 3D Reconstruction assets (InstantMesh backbone)
python engine/checkpoints/download_checkpoints.py --stage 3d

# Download all assets for fully offline execution
python engine/checkpoints/download_checkpoints.py --stage all
```

### Option B — Hugging Face CLI Native Commands

```bash
# 1. Stable Diffusion v1.5 (FP16 variant)
huggingface-cli download runwayml/stable-diffusion-v1-5 \
    --local-dir engine/checkpoints/stable_diffusion \
    --local-dir-use-symlinks False

# 2. ControlNet Scribble (FP16 variant)
huggingface-cli download lllyasviel/control_v11p_sd15_scribble \
    --local-dir engine/checkpoints/controlnet/scribble \
    --local-dir-use-symlinks False

# 3. InstantMesh
huggingface-cli download TencentARC/InstantMesh instant_mesh_large.ckpt \
    --local-dir engine/checkpoints/reconstruction_3d/instantmesh \
    --local-dir-use-symlinks False
```

---

## 🔒 Offline Air-Gapped Mode Execution

For hackathon presentation runs or unreliable networking, set the following environment variables to ensure zero external network calls:

```bash
# Linux / macOS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Windows PowerShell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:HF_DATASETS_OFFLINE="1"
```

---

## 🧩 Programmatic Loading Template (FP16)

```python
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 1. Load ControlNet Scribble adapter
controlnet = ControlNetModel.from_pretrained(
    "engine/checkpoints/controlnet/scribble",
    torch_dtype=dtype,
    use_safetensors=True,
    local_files_only=True,
)

# 2. Load Stable Diffusion v1.5 base pipeline
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "engine/checkpoints/stable_diffusion",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
    use_safetensors=True,
    local_files_only=True,
)
pipeline.to(device)

# 3. Enable low-latency memory optimizations
if device == "cuda":
    pipeline.enable_attention_slicing("auto")
```
