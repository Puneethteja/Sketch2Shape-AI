"""
engine/check_gpu.py
===================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

GPU & VRAM Verification Diagnostic Tool.
Validates PyTorch CUDA acceleration, inspects VRAM capacity, and verifies
FP16 half-precision execution readiness for diffusion & 3D backbones.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.check_gpu")

MIN_RECOMMENDED_VRAM_GB: float = 6.0

def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if PyTorch is installed and whether CUDA acceleration is available.

    Returns
    -------
    tuple[bool, str]
        (is_available, status_message)
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed in the active environment."

    if not torch.cuda.is_available():
        cuda_version = getattr(torch.version, "cuda", "None")
        return False, (
            f"PyTorch ({torch.__version__}) is installed, but CUDA is not available. "
            f"Built with CUDA: {cuda_version}."
        )

    device_count = torch.cuda.device_count()
    return True, f"CUDA available with {device_count} device(s). PyTorch version: {torch.__version__}."

def get_gpu_info(device_idx: int = 0) -> Dict[str, Any]:
    """
    Inspect GPU hardware specifications, compute capability, and VRAM memory.

    Parameters
    ----------
    device_idx : int
        Index of the CUDA device to inspect (default: 0).

    Returns
    -------
    dict[str, Any]
        Dictionary of GPU properties and memory statistics.
    """
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch is not installed."}

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
        }

    props = torch.cuda.get_device_properties(device_idx)
    total_mem_bytes = props.total_memory
    allocated_bytes = torch.cuda.memory_allocated(device_idx)
    reserved_bytes = torch.cuda.memory_reserved(device_idx)
    free_mem_bytes = total_mem_bytes - reserved_bytes

    total_gb = round(total_mem_bytes / (1024**3), 2)
    free_gb = round(free_mem_bytes / (1024**3), 2)
    allocated_gb = round(allocated_bytes / (1024**3), 2)

    return {
        "cuda_available": True,
        "device_index": device_idx,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": getattr(props, "multi_processor_count", "N/A"),
        "vram_total_gb": total_gb,
        "vram_free_gb": free_gb,
        "vram_allocated_gb": allocated_gb,
        "meets_min_vram": total_gb >= MIN_RECOMMENDED_VRAM_GB,
    }

def check_fp16_support(device_idx: int = 0) -> Tuple[bool, str]:
    """
    Validate that FP16 half-precision tensor execution functions properly on the device.

    Returns
    -------
    tuple[bool, str]
        (fp16_supported, message)
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed."

    if not torch.cuda.is_available():

        try:
            x = torch.randn(4, 4, dtype=torch.float16)
            y = x @ x
            return False, "CUDA not available; CPU float16 simulation only (slow)."
        except Exception as exc:
            return False, f"CPU float16 unsupported: {exc}"

    device = torch.device(f"cuda:{device_idx}")
    try:
        a = torch.randn(256, 256, device=device, dtype=torch.float16)
        b = torch.randn(256, 256, device=device, dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        del a, b, c
        torch.cuda.empty_cache()
        return True, "FP16 half-precision computation verified successfully on GPU."
    except Exception as exc:
        return False, f"FP16 computation failed on GPU: {exc}"

def run_gpu_diagnostic() -> Dict[str, Any]:
    """
    Run full hardware and inference diagnostic suite.

    Returns
    -------
    dict[str, Any]
        Aggregated report of CUDA readiness, VRAM status, and FP16 support.
    """
    cuda_ok, cuda_msg = check_cuda_availability()
    gpu_info = get_gpu_info(device_idx=0)
    fp16_ok, fp16_msg = check_fp16_support(device_idx=0)

    is_ready_for_production = cuda_ok and gpu_info.get("meets_min_vram", False) and fp16_ok

    report = {
        "cuda_available": cuda_ok,
        "cuda_status": cuda_msg,
        "gpu_info": gpu_info,
        "fp16_supported": fp16_ok,
        "fp16_status": fp16_msg,
        "production_ready": is_ready_for_production,
    }
    return report

def print_diagnostic_report() -> None:
    """Print a clean CLI summary of the hardware verification."""
    print("=" * 65)
    print("  Sketch2Shape AI  |  GPU & Inference Diagnostics")
    print("  TechZephyr 4.0 Hackathon  |  IIT Bhubaneswar")
    print("=" * 65)

    report = run_gpu_diagnostic()
    cuda_ok = report["cuda_available"]
    gpu = report["gpu_info"]
    fp16_ok = report["fp16_supported"]

    print(f"CUDA Available     : {'[PASS] YES' if cuda_ok else '[FAIL] NO'}")
    print(f"Status Details     : {report['cuda_status']}")

    if cuda_ok:
        print("-" * 65)
        print(f"Device Name        : {gpu.get('device_name')}")
        print(f"Compute Capability : {gpu.get('compute_capability')}")
        print(f"Total VRAM         : {gpu.get('vram_total_gb')} GB")
        print(f"Free VRAM          : {gpu.get('vram_free_gb')} GB")
        print(f"Recommended VRAM   : >= {MIN_RECOMMENDED_VRAM_GB} GB "
              f"({'[PASS]' if gpu.get('meets_min_vram') else '[WARN: Low VRAM]'})")
        print("-" * 65)

    print(f"FP16 Execution     : {'[PASS] Verified' if fp16_ok else '[WARN/FAIL] Not ready'}")
    print(f"FP16 Details       : {report['fp16_status']}")
    print("=" * 65)

    if report["production_ready"]:
        print("[SUCCESS] Environment fully configured for FP16 Diffusers + ControlNet inference.")
    else:
        print("[NOTICE] Running in CPU/Fallback mode. GPU acceleration recommended for hackathon demo.")
    print("=" * 65)

if __name__ == "__main__":
    print_diagnostic_report()
