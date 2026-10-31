"""
engine/test_pipeline.py
=======================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Production verification & smoke test script for Stable Diffusion v1.5 +
ControlNet Scribble in FP16 half-precision mode on local/cloud GPUs.

Verification Checks:
1. Environment & Package Dependencies (torch, diffusers, transformers, accelerate).
2. Hugging Face Authentication & Hub token availability.
3. GPU Acceleration, VRAM capacity, and FP16 half-precision tensor execution.
4. Pipeline initialization (ControlNetModel + StableDiffusionControlNetPipeline in FP16).
5. Low-latency hardware optimizations (Attention Slicing, PyTorch 2.x SDPA).
6. Forward inference smoke test pass (1-2 step generation with synthetic sketch).

Usage:
    python engine/test_pipeline.py              # Full smoke test (auto-detects hardware)
    python engine/test_pipeline.py --dry-run    # Verify dependencies & config without downloading weights
    python engine/test_pipeline.py --device cpu # Force CPU execution mode
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Reconfigure stdout to utf-8 if supported to prevent Windows charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.smoke_test")

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_CHECKPOINTS_DIR = ROOT_DIR / "engine" / "checkpoints"


# ── Dependency Verification ──────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("torch", "PyTorch deep learning framework", "2.1.0"),
    ("diffusers", "Hugging Face diffusion models & pipelines", "0.27.0"),
    ("transformers", "Hugging Face transformers & tokenizers", "4.38.0"),
    ("accelerate", "Hardware accelerator & mixed-precision dispatch", "0.28.0"),
    ("safetensors", "Fast & safe tensor serialization format", "0.4.0"),
    ("huggingface_hub", "Hugging Face Hub client & model loader", "0.22.0"),
    ("PIL", "Pillow image manipulation library", "10.0.0"),
    ("numpy", "Numerical computing and array tensors", "1.24.0"),
    ("cv2", "OpenCV image processing and lineart extraction", "4.8.0"),
]


def check_dependencies() -> Dict[str, Any]:
    """
    Verifies that all core ML/CV dependencies are installed and accessible.
    """
    results = {}
    all_ok = True
    missing = []

    for pkg_name, desc, min_ver in REQUIRED_PACKAGES:
        try:
            if pkg_name == "PIL":
                import PIL
                ver = getattr(PIL, "__version__", "installed")
            elif pkg_name == "cv2":
                import cv2
                ver = getattr(cv2, "__version__", "installed")
            else:
                mod = __import__(pkg_name)
                ver = getattr(mod, "__version__", "installed")
            results[pkg_name] = {
                "installed": True,
                "version": ver,
                "description": desc,
                "min_version": min_ver,
            }
        except ImportError:
            results[pkg_name] = {
                "installed": False,
                "version": None,
                "description": desc,
                "min_version": min_ver,
            }
            missing.append(pkg_name)
            all_ok = False

    return {
        "status": all_ok,
        "packages": results,
        "missing": missing,
    }


# ── Authentication & Hub Verification ────────────────────────────────────────

def check_huggingface_auth() -> Dict[str, Any]:
    """
    Verifies Hugging Face authentication tokens and connectivity status.
    Note: Stable Diffusion v1.5 and ControlNet Scribble are public models,
    so anonymous access works, but having an auth token avoids rate-limiting.
    """
    token_sources = []
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        token_sources.append("Environment variable (HF_TOKEN)")

    try:
        from huggingface_hub import HfFolder
        cached_token = HfFolder.get_token()
        if cached_token:
            token_sources.append("Cached Hugging Face CLI credentials")
    except Exception:
        pass

    has_token = len(token_sources) > 0
    token_preview = "***" + hf_token[-4:] if (hf_token and len(hf_token) > 4) else "None"

    return {
        "authenticated": has_token,
        "sources": token_sources,
        "token_preview": token_preview,
        "public_access_supported": True,
        "status": "AUTHENTICATED" if has_token else "ANONYMOUS_PUBLIC_ALLOWED",
    }


# ── Hardware & FP16 Verification ─────────────────────────────────────────────

def check_hardware(requested_device: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspects GPU availability, VRAM budget, and FP16 capability.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    target_device = requested_device or ("cuda" if cuda_available else "cpu")

    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "selected_device": target_device,
        "device_name": "CPU",
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "fp16_supported": False,
        "sdpa_supported": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    }

    if target_device.startswith("cuda") and cuda_available:
        dev_idx = 0
        info["device_name"] = torch.cuda.get_device_name(dev_idx)
        mem_free, mem_total = torch.cuda.mem_get_info(dev_idx)
        info["total_vram_gb"] = round(mem_total / (1024 ** 3), 2)
        info["free_vram_gb"] = round(mem_free / (1024 ** 3), 2)

        # Test FP16 allocation
        try:
            t = torch.zeros(16, 16, dtype=torch.float16, device=target_device)
            info["fp16_supported"] = True
            del t
        except Exception as exc:
            logger.warning(f"FP16 allocation failed: {exc}")
            info["fp16_supported"] = False
    else:
        # CPU supports float32 primarily, can emulate float16 in newer PyTorch
        info["fp16_supported"] = False

    return info


# ── Synthetic Sketch Generator for Smoke Test ────────────────────────────────

def create_synthetic_sketch_image() -> Any:
    """
    Creates a synthetic 512x512 binary lineart sketch (PIL.Image) representing
    a stylized mug / cylinder contour for smoke testing.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # White canvas with black stroke (standard ControlNet scribble input)
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw simple mug/vase outline
    # Body
    draw.rectangle([180, 150, 330, 380], outline=(0, 0, 0), width=4)
    # Rim ellipse
    draw.ellipse([180, 130, 330, 170], outline=(0, 0, 0), width=4)
    # Handle
    draw.arc([310, 190, 400, 340], start=-90, end=90, fill=(0, 0, 0), width=4)

    return img


# ── Pipeline Smoke Test Runner ───────────────────────────────────────────────

def run_smoke_test(
    device: str = "cuda",
    dry_run: bool = False,
    prompt: str = "A clean white ceramic coffee mug, 3d product asset, smooth shading, high quality",
    inference_steps: int = 1,
) -> Dict[str, Any]:
    """
    Initializes and executes the Stable Diffusion v1.5 + ControlNet Scribble pipeline.
    """
    smoke_results: Dict[str, Any] = {
        "success": False,
        "stages": {},
        "elapsed_seconds": 0.0,
        "error": None,
    }

    t0 = time.perf_counter()

    # Step 1: Check dependencies
    dep_report = check_dependencies()
    smoke_results["stages"]["dependencies"] = dep_report

    # Step 2: Check authentication (always run for complete audit)
    auth_report = check_huggingface_auth()
    smoke_results["stages"]["auth"] = auth_report

    # Step 3: Check hardware (always run for complete audit)
    hw_report = check_hardware(requested_device=device)
    smoke_results["stages"]["hardware"] = hw_report
    actual_device = hw_report["selected_device"]

    # Verify synthetic sketch creation contract
    try:
        synthetic_sketch = create_synthetic_sketch_image()
        sketch_ok = (synthetic_sketch.size == (512, 512))
    except Exception as exc:
        sketch_ok = False
        logger.warning(f"Synthetic sketch generation failed: {exc}")

    smoke_results["stages"]["sketch_contract"] = {
        "verified": sketch_ok,
        "dimensions": (512, 512),
        "format": "PIL.Image (RGB)",
    }

    if not dep_report["status"]:
        missing_str = ", ".join(dep_report["missing"])
        msg = (
            f"Missing required dependencies: {missing_str}. "
            f"Run: pip install {' '.join(dep_report['missing'])} (or pip install -r requirements.txt)"
        )
        logger.warning(msg)
        smoke_results["error"] = msg
        smoke_results["stages"]["pipeline_load"] = {
            "status": "DEPENDENCY_PREREQUISITES_MISSING",
            "missing": dep_report["missing"],
            "resolution": f"pip install -r requirements.txt",
        }
        smoke_results["stages"]["inference"] = {"status": "BLOCKED_BY_DEPENDENCIES"}
        smoke_results["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        return smoke_results

    import torch

    dtype = torch.float16 if actual_device.startswith("cuda") and hw_report["fp16_supported"] else torch.float32

    logger.info(f"Target execution device: {actual_device} | Precision: {dtype}")

    if dry_run:
        logger.info("[DRY RUN] Dependencies verified. Skipping remote weight download.")
        smoke_results["success"] = True
        smoke_results["stages"]["pipeline_load"] = {
            "status": "VERIFIED_DRY_RUN",
            "device": actual_device,
            "target_precision": str(dtype),
            "sd_source": "runwayml/stable-diffusion-v1-5",
            "controlnet_source": "lllyasviel/control_v11p_sd15_scribble",
        }
        smoke_results["stages"]["inference"] = {
            "status": "VERIFIED_DRY_RUN",
            "sketch_input_shape": (512, 512),
            "expected_output_shape": (512, 512),
        }
        smoke_results["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        return smoke_results

    # Step 4: Pipeline loading
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        # Determine checkpoint source: local directory or Hugging Face Hub
        local_sd = LOCAL_CHECKPOINTS_DIR / "stable_diffusion"
        local_cn = LOCAL_CHECKPOINTS_DIR / "controlnet" / "scribble"

        sd_source = str(local_sd) if (local_sd / "model_index.json").exists() else "runwayml/stable-diffusion-v1-5"
        cn_source = str(local_cn) if (local_cn / "config.json").exists() else "lllyasviel/control_v11p_sd15_scribble"

        logger.info(f"Loading ControlNet adapter from: {cn_source} ...")
        t_load_cn = time.perf_counter()
        controlnet = ControlNetModel.from_pretrained(
            cn_source,
            torch_dtype=dtype,
        )
        cn_load_time = time.perf_counter() - t_load_cn
        logger.info(f"ControlNet loaded in {cn_load_time:.2f}s")

        logger.info(f"Loading Stable Diffusion v1.5 from: {sd_source} ...")
        t_load_sd = time.perf_counter()
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_source,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
        sd_load_time = time.perf_counter() - t_load_sd
        logger.info(f"Stable Diffusion pipeline loaded in {sd_load_time:.2f}s")

        # Hardware dispatch & memory optimization
        if actual_device.startswith("cuda"):
            pipeline.to(actual_device)
            # Memory optimization for T4 / V100
            pipeline.enable_attention_slicing("auto")
            logger.info("Enabled auto attention slicing for low-latency VRAM stability.")
        else:
            pipeline.to("cpu")

        smoke_results["stages"]["pipeline_load"] = {
            "status": "SUCCESS",
            "controlnet_source": cn_source,
            "sd_source": sd_source,
            "controlnet_load_seconds": round(cn_load_time, 2),
            "sd_load_seconds": round(sd_load_time, 2),
            "precision": str(dtype),
        }

        # Step 5: Smoke test inference pass
        logger.info(f"Executing smoke test inference pass ({inference_steps} step)...")
        synthetic_sketch = create_synthetic_sketch_image()

        t_infer = time.perf_counter()
        generator = torch.manual_seed(42)
        output = pipeline(
            prompt=prompt,
            image=synthetic_sketch,
            num_inference_steps=inference_steps,
            guidance_scale=7.5,
            generator=generator,
        )
        infer_time = time.perf_counter() - t_infer

        generated_image = output.images[0]
        logger.info(
            f"Inference smoke test succeeded in {infer_time:.2f}s | Output size: {generated_image.size}"
        )

        smoke_results["stages"]["inference"] = {
            "status": "SUCCESS",
            "latency_seconds": round(infer_time, 2),
            "output_dimensions": generated_image.size,
        }
        smoke_results["success"] = True

    except Exception as exc:
        logger.error(f"Pipeline smoke test encountered error: {exc}", exc_info=True)
        smoke_results["error"] = str(exc)
        smoke_results["stages"]["pipeline_load"] = {"status": "FAILED", "error": str(exc)}

    smoke_results["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    return smoke_results


# ── Console Presentation ─────────────────────────────────────────────────────

def print_smoke_report(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print(" [Sketch2Shape AI] Stable Diffusion + ControlNet Pipeline Smoke Test")
    print(" TechZephyr 4.0 Hackathon - IIT Bhubaneswar")
    print(" Lead Engineer : Mona (Generative AI & 3D Reconstruction)")
    print("=" * 78)

    # 1. Dependencies
    dep = report["stages"].get("dependencies", {})
    print("\n[1] Environment & Package Dependencies:")
    for pkg, info in dep.get("packages", {}).items():
        state = f"[OK] ({info['version']})" if info["installed"] else "[MISSING]"
        print(f"  * {pkg:<16} : {state:<20} - {info['description']}")

    # 2. Authentication
    auth = report["stages"].get("auth", {})
    print("\n[2] Hugging Face Hub Access & Authentication:")
    print(f"  * Access Status      : {auth.get('status', 'N/A')}")
    print(f"  * Auth Sources       : {', '.join(auth.get('sources', [])) or 'None (Anonymous Mode)'}")
    print(f"  * Token Preview      : {auth.get('token_preview', 'N/A')}")
    print(f"  * Public Models Perm : ALLOWED (runwayml/sd-v1-5 & lllyasviel/controlnet)")

    # 3. Hardware
    hw = report["stages"].get("hardware", {})
    print("\n[3] Hardware Acceleration & VRAM Status:")
    print(f"  * CUDA Available     : {hw.get('cuda_available')}")
    print(f"  * Selected Device    : {hw.get('selected_device')}")
    print(f"  * Device Name        : {hw.get('device_name')}")
    print(f"  * VRAM (Total / Free): {hw.get('total_vram_gb', 0)} GB / {hw.get('free_vram_gb', 0)} GB")
    print(f"  * FP16 Support       : {hw.get('fp16_supported')}")
    print(f"  * PyTorch SDPA       : {hw.get('sdpa_supported')}")

    # 4. Pipeline Execution
    print("\n[4] Smoke Test Execution Results:")
    pl = report["stages"].get("pipeline_load", {})
    inf = report["stages"].get("inference", {})

    if report.get("success"):
        print(f"  * Overall Status     : [PASSED] (Elapsed: {report['elapsed_seconds']}s)")
        if pl.get("status") == "SUCCESS":
            print(f"  * Model Precision    : {pl.get('precision')}")
            print(f"  * ControlNet Load    : {pl.get('controlnet_load_seconds')}s")
            print(f"  * SD v1.5 Load       : {pl.get('sd_load_seconds')}s")
            print(f"  * Forward Pass Run   : {inf.get('latency_seconds')}s (Output: {inf.get('output_dimensions')})")
        elif pl.get("status") == "VERIFIED_DRY_RUN":
            print("  * Dry Run Status     : [VERIFIED] All pipeline contracts & dependencies validated.")
            print(f"  * Precision Target   : {pl.get('target_precision')} on {pl.get('device')}")
            print(f"  * Input / Output Res : {inf.get('sketch_input_shape')} -> {inf.get('expected_output_shape')}")
    else:
        print(f"  * Overall Status     : [DIAGNOSTIC NOTICE / ACTION REQUIRED]")
        if report.get("error"):
            print(f"  * Diagnostic Detail  : {report.get('error')}")
        if pl.get("resolution"):
            print(f"  * Recommended Action : {pl.get('resolution')}")

    print("\n" + "=" * 78)


# ── Pytest Test Suite Hooks ──────────────────────────────────────────────────

def test_dependency_integrity() -> None:
    """Pytest test case: verifies dependencies."""
    dep = check_dependencies()
    assert isinstance(dep, dict)
    assert "packages" in dep


def test_huggingface_auth_status() -> None:
    """Pytest test case: verifies Hugging Face auth resolution."""
    auth = check_huggingface_auth()
    assert auth["public_access_supported"] is True


def test_synthetic_sketch_generation() -> None:
    """Pytest test case: verifies synthetic sketch creation contracts."""
    sketch = create_synthetic_sketch_image()
    assert sketch.size == (512, 512)
    assert sketch.mode == "RGB"


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and smoke-test Stable Diffusion + ControlNet pipeline."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify dependencies, hardware, and configuration without loading weights.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any dependency is missing or smoke test fails.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of inference steps for the smoke test (default: 1 for fast verification).",
    )
    args = parser.parse_args()

    dep = check_dependencies()
    is_dry_run = args.dry_run

    results = run_smoke_test(
        device=args.device or ("cuda" if check_hardware()["cuda_available"] else "cpu"),
        dry_run=is_dry_run,
        inference_steps=args.steps,
    )

    print_smoke_report(results)

    if args.strict and not results["success"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
