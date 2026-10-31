"""
check_gpu.py
============
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

GPU, VRAM, and FP16 half-precision verification script for local and cloud
hardware environments (NVIDIA T4, V100, A100, RTX series, or CPU fallback).

Verification Scope:
1. PyTorch version, CUDA compilation build, and cuDNN status.
2. CUDA availability and GPU device enumeration.
3. Compute capability and memory allocation (total, reserved, allocated, free VRAM).
4. FP16 half-precision tensor allocation & arithmetic performance benchmark.
5. BF16 (bfloat16) and TensorFloat-32 (TF32) architecture capability check.
6. Target workload suitability evaluation for:
   - Diffusion 2.5D Generation (SD 1.5 + ControlNet in FP16, ~6-8 GB VRAM)
   - 3D Mesh Reconstruction (InstantMesh / Unique3D in FP16, ~10-14 GB VRAM)

Usage:
    python check_gpu.py                # Standard human-readable console report
    python check_gpu.py --json         # Machine-readable JSON output for CI/CD
    python check_gpu.py --benchmark    # Run extended FP16 vs FP32 latency benchmark
    python check_gpu.py --strict       # Exit with code 1 if CUDA is not available
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.check_gpu")


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DeviceVRAMInfo:
    total_mb: float = 0.0
    free_mb: float = 0.0
    used_mb: float = 0.0
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_gb: float = 0.0
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0


@dataclass
class GPUSpec:
    device_id: int
    name: str
    compute_capability: str
    vram: DeviceVRAMInfo
    fp16_supported: bool = False
    fp16_test_passed: bool = False
    fp16_benchmark_ms: Optional[float] = None
    fp32_benchmark_ms: Optional[float] = None
    fp16_speedup: Optional[float] = None
    bf16_supported: bool = False
    tf32_supported: bool = False
    workload_suitability: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEnvironmentReport:
    os_name: str
    os_version: str
    python_version: str
    pytorch_version: str
    cuda_compiled_version: Optional[str]
    cudnn_version: Optional[str]
    cuda_available: bool
    device_count: int
    devices: List[GPUSpec] = field(default_factory=list)
    system_status: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


# ── Verification Functions ───────────────────────────────────────────────────

def test_fp16_arithmetic(
    device: Any,
    benchmark: bool = False,
    matrix_dim: int = 2048,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """
    Allocates half-precision (FP16) tensors and verifies numerical correctness
    and computational stability. Optionally benchmarks FP16 vs FP32 latency.

    Returns:
        (passed, fp16_time_ms, fp32_time_ms, speedup_factor)
    """
    import torch

    try:
        # 1. Functional correctness test
        a_fp16 = torch.randn(256, 256, dtype=torch.float16, device=device)
        b_fp16 = torch.randn(256, 256, dtype=torch.float16, device=device)
        c_fp16 = torch.matmul(a_fp16, b_fp16)

        # Numerical integrity check
        if torch.isnan(c_fp16).any() or torch.isinf(c_fp16).any():
            return False, None, None, None

        if not benchmark:
            return True, None, None, None

        # 2. Performance benchmark: FP16
        x_fp16 = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)
        y_fp16 = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(warmup_iters):
            _ = torch.matmul(x_fp16, y_fp16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = torch.matmul(x_fp16, y_fp16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fp16_ms = ((time.perf_counter() - t0) / bench_iters) * 1000.0

        # Performance benchmark: FP32
        x_fp32 = torch.randn(matrix_dim, matrix_dim, dtype=torch.float32, device=device)
        y_fp32 = torch.randn(matrix_dim, matrix_dim, dtype=torch.float32, device=device)

        for _ in range(warmup_iters):
            _ = torch.matmul(x_fp32, y_fp32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = torch.matmul(x_fp32, y_fp32)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fp32_ms = ((time.perf_counter() - t0) / bench_iters) * 1000.0

        speedup = round(fp32_ms / fp16_ms, 2) if fp16_ms > 0 else 1.0
        return True, round(fp16_ms, 3), round(fp32_ms, 3), speedup

    except Exception as exc:
        logger.warning(f"FP16 arithmetic validation error on {device}: {exc}")
        return False, None, None, None


def evaluate_workload_suitability(vram_gb: float, fp16_ok: bool) -> Dict[str, Any]:
    """
    Evaluates hardware compatibility against Sketch2Shape AI pipeline stages:
    - Preprocessing: Low footprint (~0.5 GB)
    - Diffusion 2.5D (SD 1.5 + ControlNet in FP16): ~6-8 GB VRAM
    - 3D Reconstruction (InstantMesh / Unique3D in FP16): ~10-14 GB VRAM
    """
    diffusion_capable = fp16_ok and vram_gb >= 6.0
    recon3d_capable = fp16_ok and vram_gb >= 10.0
    unified_pipeline_capable = fp16_ok and vram_gb >= 14.0

    recommendations = []
    if vram_gb < 6.0:
        recommendations.append(
            "VRAM is below 6 GB: Enable enable_model_cpu_offload() and attention_slicing for diffusion."
        )
    elif vram_gb < 12.0:
        recommendations.append(
            "VRAM is between 6-12 GB (e.g., T4 16GB shared): Sequential execution recommended. "
            "Unload Diffusion pipeline from VRAM before initializing 3D Reconstruction backbone."
        )
    else:
        recommendations.append(
            "VRAM >= 12 GB (e.g., NVIDIA V100 16GB/32GB, A100, RTX 3090/4090): Full FP16 throughput "
            "with PyTorch 2.x SDPA (Scaled Dot-Product Attention) enabled."
        )

    return {
        "stage1_preprocessing": {
            "supported": True,
            "min_vram_gb": 0.5,
            "status": "READY",
        },
        "stage2_diffusion_sd15_controlnet": {
            "supported": diffusion_capable,
            "min_vram_gb": 6.0,
            "status": "READY" if diffusion_capable else "NEEDS_OFFLOAD",
            "optimizations_needed": ["attention_slicing", "cpu_offload"] if not diffusion_capable else ["sdpa"],
        },
        "stage3_reconstruction_instantmesh_unique3d": {
            "supported": recon3d_capable,
            "min_vram_gb": 10.0,
            "status": "READY" if recon3d_capable else "NEEDS_SEQUENTIAL_UNLOAD",
        },
        "end_to_end_concurrent_pipeline": {
            "supported": unified_pipeline_capable,
            "recommended_vram_gb": 16.0,
            "mode": "CONCURRENT" if unified_pipeline_capable else "SEQUENTIAL_PIPELINE",
        },
        "recommendations": recommendations,
    }


def inspect_system(benchmark: bool = False) -> SystemEnvironmentReport:
    """
    Performs comprehensive hardware and framework inspection, returning a
    structured SystemEnvironmentReport.
    """
    import torch

    os_name = platform.system()
    os_version = platform.release()
    py_version = platform.python_version()
    pytorch_version = torch.__version__
    cuda_compiled_version = torch.version.cuda
    cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    cuda_available = torch.cuda.is_available()

    report = SystemEnvironmentReport(
        os_name=os_name,
        os_version=os_version,
        python_version=py_version,
        pytorch_version=pytorch_version,
        cuda_compiled_version=cuda_compiled_version,
        cudnn_version=cudnn_version,
        cuda_available=cuda_available,
        device_count=torch.cuda.device_count() if cuda_available else 0,
    )

    if not cuda_available:
        report.system_status = "CPU_ONLY"
        report.notes.append(
            "CUDA is not available in current PyTorch installation. "
            "Running in CPU mode. For GPU acceleration (NVIDIA T4 / V100), install CUDA-enabled "
            "PyTorch via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 "
            "or https://download.pytorch.org/whl/cu121"
        )
        return report

    report.system_status = "CUDA_ACCELERATED"

    # Inspect each CUDA device
    for idx in range(report.device_count):
        device = torch.device(f"cuda:{idx}")
        dev_name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        cap_str = f"sm_{cap[0]}{cap[1]}"

        # VRAM stats
        mem_free, mem_total = torch.cuda.mem_get_info(idx)
        mem_allocated = torch.cuda.memory_allocated(idx)
        mem_reserved = torch.cuda.memory_reserved(idx)

        total_mb = round(mem_total / (1024 ** 2), 2)
        free_mb = round(mem_free / (1024 ** 2), 2)
        used_mb = round((mem_total - mem_free) / (1024 ** 2), 2)

        vram_info = DeviceVRAMInfo(
            total_mb=total_mb,
            free_mb=free_mb,
            used_mb=used_mb,
            total_gb=round(total_mb / 1024.0, 2),
            free_gb=round(free_mb / 1024.0, 2),
            used_gb=round(used_mb / 1024.0, 2),
            allocated_mb=round(mem_allocated / (1024 ** 2), 2),
            reserved_mb=round(mem_reserved / (1024 ** 2), 2),
        )

        # Feature checks
        fp16_supported = cap[0] >= 5  # Maxwell or newer supports FP16
        bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        tf32_supported = cap[0] >= 8   # Ampere or newer supports TF32

        # FP16 tensor arithmetic validation
        fp16_ok, fp16_ms, fp32_ms, speedup = test_fp16_arithmetic(
            device=device,
            benchmark=benchmark,
        )

        suitability = evaluate_workload_suitability(vram_info.total_gb, fp16_ok)

        gpu_spec = GPUSpec(
            device_id=idx,
            name=dev_name,
            compute_capability=cap_str,
            vram=vram_info,
            fp16_supported=fp16_supported,
            fp16_test_passed=fp16_ok,
            fp16_benchmark_ms=fp16_ms,
            fp32_benchmark_ms=fp32_ms,
            fp16_speedup=speedup,
            bf16_supported=bf16_supported,
            tf32_supported=tf32_supported,
            workload_suitability=suitability,
        )
        report.devices.append(gpu_spec)

    return report


# ── Console Output Formatting ────────────────────────────────────────────────

# Reconfigure stdout to utf-8 if supported to prevent Windows cp1252 charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_banner() -> None:
    print("=" * 78)
    print(" [Sketch2Shape AI] Hardware & VRAM Verification Script")
    print(" TechZephyr 4.0 Hackathon - IIT Bhubaneswar")
    print(" Lead Engineer : Mona (Generative AI & 3D Reconstruction)")
    print("=" * 78)


def print_report_table(report: SystemEnvironmentReport, benchmark: bool = False) -> None:
    print("\n[1] Framework & Environment:")
    print(f"  * Operating System   : {report.os_name} {report.os_version}")
    print(f"  * Python Runtime     : {report.python_version}")
    print(f"  * PyTorch Version    : {report.pytorch_version}")
    print(f"  * CUDA Built With    : {report.cuda_compiled_version or 'N/A (CPU Build)'}")
    print(f"  * cuDNN Version      : {report.cudnn_version or 'N/A'}")
    print(f"  * CUDA Device Count  : {report.device_count}")
    print(f"  * Execution Status   : {report.system_status}")

    if not report.cuda_available:
        print("\n[!] CUDA Acceleration Unavailable:")
        for note in report.notes:
            print(f"  [-] {note}")
        print("\n" + "=" * 78)
        return

    print("\n[2] CUDA GPU Device Specifications:")
    for dev in report.devices:
        vram = dev.vram
        print(f"\n  -- GPU [{dev.device_id}]: {dev.name} ----------------------------------")
        print(f"  * Compute Capability : {dev.compute_capability}")
        print(f"  * Total VRAM         : {vram.total_gb:.2f} GB ({vram.total_mb:.1f} MB)")
        print(f"  * Free VRAM          : {vram.free_gb:.2f} GB ({vram.free_mb:.1f} MB)")
        print(f"  * Used VRAM          : {vram.used_gb:.2f} GB ({vram.used_mb:.1f} MB)")
        print(f"  * Allocated / Reserved: {vram.allocated_mb:.1f} MB / {vram.reserved_mb:.1f} MB")
        print(f"  * FP16 Half-Precision: {'[OK] SUPPORTED & VERIFIED' if dev.fp16_test_passed else '[FAIL] NOT SUPPORTED'}")
        print(f"  * BF16 Support       : {'[OK] Supported' if dev.bf16_supported else '[-] Not available'}")
        print(f"  * TF32 TensorFloat   : {'[OK] Supported' if dev.tf32_supported else '[-] Not available'}")

        if benchmark and dev.fp16_benchmark_ms:
            print("  * Latency Benchmark  :")
            print(f"      - FP16 GEMM: {dev.fp16_benchmark_ms:.2f} ms")
            print(f"      - FP32 GEMM: {dev.fp32_benchmark_ms:.2f} ms")
            print(f"      - FP16 Acceleration: {dev.fp16_speedup}x speedup")

        print("\n  [3] Pipeline Workload Suitability Matrix:")
        suitability = dev.workload_suitability
        s2 = suitability["stage2_diffusion_sd15_controlnet"]
        s3 = suitability["stage3_reconstruction_instantmesh_unique3d"]
        e2e = suitability["end_to_end_concurrent_pipeline"]

        print(f"  * Stage 2 Diffusion (SD1.5 + ControlNet FP16) : [{s2['status']}] (Min: 6.0 GB)")
        print(f"  * Stage 3 Reconstruction (InstantMesh/Unique3D): [{s3['status']}] (Min: 10.0 GB)")
        print(f"  * Pipeline Concurrency Mode                   : [{e2e['mode']}]")

        print("\n  [4] Hardware Optimization Directives:")
        for rec in suitability["recommendations"]:
            print(f"  >> {rec}")

    print("\n" + "=" * 78)


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify PyTorch CUDA acceleration, VRAM allocation, and FP16 half-precision for Sketch2Shape AI."
    )
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON format.")
    parser.add_argument("--benchmark", action="store_true", help="Run FP16 vs FP32 latency benchmark.")
    parser.add_argument("--strict", action="store_true", help="Exit code 1 if CUDA is unavailable or FP16 fails.")
    args = parser.parse_args()

    report = inspect_system(benchmark=args.benchmark)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_banner()
        print_report_table(report, benchmark=args.benchmark)

    if args.strict:
        if not report.cuda_available:
            logger.error("Strict validation failed: CUDA is not available.")
            return 1
        if any(not d.fp16_test_passed for d in report.devices):
            logger.error("Strict validation failed: One or more GPUs failed FP16 verification.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
