# Sketch2Shape AI 🖊️ → 🧊

> **TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> Convert hand-drawn sketches into 3-D shapes using a diffusion-guided pipeline.

---

## 🏗️ Architecture & Project Structure

```
Sketch2Shape-AI/
├── engine/                   # Core ML & CV pipeline
│   ├── __init__.py
│   └── preprocessing.py      # Ingest, normalize, letterbox resize, bilateral filter, lineart
├── ui/                       # Front-end / drawing canvas interface
│   └── .gitkeep
├── outputs/                  # Generated images & 3-D assets
│   └── .gitkeep
├── tests/                    # pytest test suite
│   ├── __init__.py
│   ├── test_preprocessing.py # Ingest, color normalisation, resize, filter, lineart
│   └── test_image_utils.py   # Image I/O and visualization helpers
├── utils/                    # Shared helper utilities
│   ├── __init__.py
│   └── image_utils.py        # I/O, PIL conversion, side-by-side display, mask overlay
├── environment.yml           # Conda environment spec (CUDA-enabled PyTorch)
├── requirements.txt          # pip requirements specification
└── .gitignore
```

---

## ⚙️ Quick Start & Environment Configuration

### Option A — Conda (Recommended for CUDA acceleration)

```bash
conda env create -f environment.yml
conda activate sketch2shape
```

### Option B — pip

```bash
# 1. Install PyTorch with matching CUDA runtime (e.g., CUDA 11.8):
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118   # or cu121

# 2. Install remaining core dependencies:
pip install -r requirements.txt
```

---

## 🎨 Preprocessing Pipeline (`engine/preprocessing.py`)

The Phase 1 pipeline handles varied drawing inputs and prepares clean tensors for downstream diffusion & shape models:

1. **Universal Ingestion (`ingest`)**:
   - Local image files (`.png`, `.jpg`, `.bmp`, `.tiff`, etc.)
   - Drawing canvas base64 payloads (`data:image/png;base64,...`)
   - Raw bytes (HTTP multipart uploads / canvas Blob)
   - In-memory `PIL.Image` and `np.ndarray`
2. **Channel Normalisation (`to_rgb`)**:
   - Converts grayscale, single-channel, and RGBA into unified 3-channel RGB.
   - Composites alpha channels cleanly over a neutral white background.
3. **Aspect-Ratio Preserving Resize (`pad_and_resize`)**:
   - Letterboxes images to $512 \times 512$ with configurable padding color.
   - Uses `INTER_AREA` for downscaling and `INTER_CUBIC` for upscaling.
4. **Bilateral Filtering (`bilateral_filter`)**:
   - Smooths background texture and pencil noise while preserving crisp boundary strokes.
5. **Binary Lineart / Scribble Extraction (`to_lineart`)**:
   - Applies Gaussian adaptive thresholding and morphological opening to produce pristine stroke maps for ControlNet scribble conditioning.
6. **Unified Output (`preprocess`)**:
   - Returns normalized PyTorch tensors `(C, 512, 512)` in `[0, 1]` or raw NumPy arrays.

### Preprocessing CLI Usage

```bash
# 1. Standard color preprocessing (512x512 letterboxed + bilateral denoised)
python engine/preprocessing.py path/to/sketch.png
# → outputs/<name>_preprocessed.png

# 2. Binary lineart / scribble generation
python engine/preprocessing.py path/to/sketch.png --lineart
# → outputs/<name>_lineart.png
```

---

## 🧪 Testing

The repository includes a comprehensive `pytest` test suite:

```bash
pytest tests/ -v
```

All 49 unit and integration tests validate:
- File, bytes, and base64 drawing canvas payload ingestion
- Grayscale, RGBA, and float normalisation
- Square, portrait, and landscape letterbox padding
- Bilateral filtering edge retention
- Gaussian adaptive threshold lineart extraction
- PyTorch tensor conversions and shape contracts
- Shared image utility helpers

---

## 👥 Team

| Role | Name |
|---|---|
| Lead Data & Infrastructure Engineer | Puneeth |

---

*TechZephyr 4.0 Hackathon · IIT Bhubaneswar*
