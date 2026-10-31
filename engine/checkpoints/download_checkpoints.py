"""
engine/checkpoints/download_checkpoints.py
=========================================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

CLI and programmatic utility to download, verify, and validate model checkpoints
from the Hugging Face Hub into engine/checkpoints/ for offline low-latency inference.

Models Managed:
- Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- ControlNet Scribble (lllyasviel/control_v11p_sd15_scribble)
- ControlNet Canny (lllyasviel/control_v11p_sd15_canny)
- InstantMesh LRM (TencentARC/InstantMesh)
- Zero123++ v1.2 (sudo-ai/zero123plus-v1.2)
- Unique3D (Wai-Shing/Unique3D)

Usage:
    python engine/checkpoints/download_checkpoints.py --dry-run
    python engine/checkpoints/download_checkpoints.py --verify
    python engine/checkpoints/download_checkpoints.py --stage diffusion
    python engine/checkpoints/download_checkpoints.py --stage all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.checkpoints")

CHECKPOINTS_ROOT = Path(__file__).resolve().parent

# ── Model Registry Specification ─────────────────────────────────────────────

CHECKPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sd15": {
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "category": "diffusion",
        "description": "Stable Diffusion v1.5 base generative pipeline (FP16)",
        "destination": CHECKPOINTS_ROOT / "stable_diffusion",
        "required_files": [
            "model_index.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.fp16.safetensors",
        ],
        "size_gb": 3.4,
    },
    "scribble": {
        "repo_id": "lllyasviel/control_v11p_sd15_scribble",
        "category": "controlnet",
        "description": "ControlNet v1.1 Scribble spatial sketch conditioning adapter",
        "destination": CHECKPOINTS_ROOT / "controlnet" / "scribble",
        "required_files": [
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        ],
        "size_gb": 1.45,
    },
    "canny": {
        "repo_id": "lllyasviel/control_v11p_sd15_canny",
        "category": "controlnet",
        "description": "ControlNet v1.1 Canny edge conditioning adapter",
        "destination": CHECKPOINTS_ROOT / "controlnet" / "canny",
        "required_files": [
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        ],
        "size_gb": 1.45,
    },
    "instantmesh": {
        "repo_id": "TencentARC/InstantMesh",
        "category": "3d",
        "description": "InstantMesh Large Reconstruction Model (LRM) backbone",
        "destination": CHECKPOINTS_ROOT / "reconstruction_3d" / "instantmesh",
        "required_files": [
            "instant_mesh_large.ckpt",
        ],
        "size_gb": 1.6,
    },
    "zero123plus": {
        "repo_id": "sudo-ai/zero123plus-v1.2",
        "category": "3d",
        "description": "Zero123++ v1.2 multi-view elevation/azimuth diffusion model",
        "destination": CHECKPOINTS_ROOT / "reconstruction_3d" / "instantmesh" / "zero123plus-v1.2",
        "required_files": [
            "diffusion_pytorch_model.safetensors",
        ],
        "size_gb": 3.8,
    },
    "unique3d": {
        "repo_id": "Wai-Shing/Unique3D",
        "category": "3d",
        "description": "Unique3D high-resolution mesh reconstruction pipeline (alternative)",
        "destination": CHECKPOINTS_ROOT / "reconstruction_3d" / "unique3d",
        "required_files": [
            "controlnet-tile/diffusion_pytorch_model.safetensors",
        ],
        "size_gb": 4.2,
    },
}


# ── Checkpoint Operations ───────────────────────────────────────────────────

def verify_checkpoints() -> Dict[str, Dict[str, Any]]:
    """
    Checks each model component in the registry and verifies file existence.
    """
    status: Dict[str, Dict[str, Any]] = {}
    for key, spec in CHECKPOINT_REGISTRY.items():
        dest = spec["destination"]
        exists = dest.exists()
        found_files = []
        missing_files = []

        if exists:
            for rel_file in spec["required_files"]:
                full_path = dest / rel_file
                if full_path.exists():
                    found_files.append(rel_file)
                else:
                    missing_files.append(rel_file)
        else:
            missing_files = spec["required_files"].copy()

        is_complete = exists and len(missing_files) == 0
        status[key] = {
            "name": key,
            "category": spec["category"],
            "repo_id": spec["repo_id"],
            "destination": str(dest),
            "complete": is_complete,
            "found_files": found_files,
            "missing_files": missing_files,
            "size_gb": spec["size_gb"],
        }
    return status


def print_verification_report(status: Dict[str, Dict[str, Any]]) -> None:
    print("=" * 78)
    print(" [Sketch2Shape AI] Model Checkpoint Verification Report")
    print(" Lead Engineer : Mona (Generative AI & 3D Reconstruction)")
    print("=" * 78)

    all_complete = True
    total_found_gb = 0.0
    total_missing_gb = 0.0

    for key, info in status.items():
        state = "[PRESENT]" if info["complete"] else "[MISSING / PLACEHOLDER]"
        if not info["complete"]:
            all_complete = False
            total_missing_gb += info["size_gb"]
        else:
            total_found_gb += info["size_gb"]

        print(f"\n  -- Model [{key.upper()}]: {info['repo_id']}")
        print(f"     Category   : {info['category']}")
        print(f"     Destination: {info['destination']}")
        print(f"     Status     : {state}")
        print(f"     Disk Est.  : ~{info['size_gb']} GB")
        if info["missing_files"]:
            print(f"     Missing    : {', '.join(info['missing_files'][:3])}")

    print("\n" + "-" * 78)
    print(f"  Summary: {total_found_gb:.1f} GB present, {total_missing_gb:.1f} GB to download.")
    if all_complete:
        print("  All model checkpoints are present and verified for offline execution!")
    else:
        print("  To download missing assets, run with network access:")
        print("    python engine/checkpoints/download_checkpoints.py --stage all")
    print("=" * 78)


def download_stage(
    stage: str,
    dry_run: bool = False,
    auth_token: Optional[str] = None,
) -> bool:
    """
    Downloads registered checkpoints for a specific stage or all stages.
    """
    targets = []
    for key, spec in CHECKPOINT_REGISTRY.items():
        if stage in ("all", spec["category"], key):
            targets.append((key, spec))

    if not targets:
        logger.error(f"No models match stage or key: '{stage}'")
        return False

    print(f"\n[+] Preparing to download {len(targets)} component(s) for stage: '{stage}'")
    for key, spec in targets:
        print(f"  * {key.upper()} ({spec['repo_id']}) -> {spec['destination']} (~{spec['size_gb']} GB)")

    if dry_run:
        print("\n[DRY RUN] No files were downloaded. Exiting.")
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub package is not installed. Install via: pip install huggingface-hub"
        )
        return False

    token = auth_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for key, spec in targets:
        dest = spec["destination"]
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading {key} from Hugging Face: {spec['repo_id']} ...")
        try:
            snapshot_download(
                repo_id=spec["repo_id"],
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                token=token,
            )
            logger.info(f"Successfully downloaded {key} into {dest}")
        except Exception as exc:
            logger.error(f"Failed to download {key} ({spec['repo_id']}): {exc}")
            return False

    return True


# ── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify Hugging Face model checkpoints for Sketch2Shape AI."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["diffusion", "controlnet", "3d", "sd15", "scribble", "canny", "instantmesh", "all"],
        help="Stage or specific model key to download (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display download targets and local directory status without downloading.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify checkpoint files present in engine/checkpoints/ without downloading.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face personal access token (or set HF_TOKEN environment variable).",
    )
    args = parser.parse_args()

    if args.verify or args.dry_run:
        status = verify_checkpoints()
        print_verification_report(status)
        if args.dry_run and not args.verify:
            download_stage(args.stage, dry_run=True, auth_token=args.token)
        return 0

    success = download_stage(args.stage, dry_run=False, auth_token=args.token)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
