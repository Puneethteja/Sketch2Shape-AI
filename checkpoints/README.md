# Model Checkpoints Architecture & Weights Directory

```
checkpoints/
├── stable-diffusion-v1-5/    # Stable Diffusion v1.5 base checkpoint (runwayml/stable-diffusion-v1-5)
├── controlnet-scribble/      # ControlNet scribble / lineart adapter (lllyasviel/control_v11p_sd15_scribble)
├── reconstruction_3d/        # 3D reconstruction backbone (InstantMesh / Unique3D)
└── rembg/                    # U²-Net alpha matting checkpoint (u2net.onnx)
```

## Model Specifications

| Component | Hugging Face / Source ID | Precision | Min VRAM |
|---|---|---|---|
| **Base 2D Diffusion** | `runwayml/stable-diffusion-v1-5` | FP16 (`torch.float16`) | ~4.0 GB |
| **ControlNet Adapter** | `lllyasviel/control_v11p_sd15_scribble` | FP16 (`torch.float16`) | ~1.5 GB |
| **3D Reconstruction** | `TencentARC/InstantMesh` | FP16 / FP32 | ~6.0 GB |
| **Alpha Matting** | `u2net` via `rembg` | ONNX / FP32 | ~1.0 GB |

## Loading Protocol
Weights are automatically fetched through the Hugging Face Hub cache when connected online, or loaded offline directly from this `checkpoints/` directory if pre-downloaded for offline hackathon environments.

> Model weight binaries (`*.safetensors`, `*.bin`, `*.pth`, `*.onnx`) are excluded by `.gitignore` to prevent repository bloat.
