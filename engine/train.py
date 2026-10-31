"""
engine/train.py
================
Sketch2Shape AI · Deep Neural Training & Fine-Tuning Pipeline
Trained neural model predicting dense depth maps, unit surface normal vectors,
and foreground occupancy fields directly from input sketch contour tensors.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.train")

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"


# ── Neural Network Architecture ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual convolutional block with LayerNorm and SiLU activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act2(out + residual)


class SketchToDepthNormalNet(nn.Module):
    """
    Multi-task convolutional encoder-decoder predicting:
    1. Continuous surface depth map: (B, 1, H, W) in [0, 1]
    2. Unit surface normal map: (B, 3, H, W) on the S² unit sphere
    3. Foreground occupancy mask: (B, 1, H, W) in [0, 1]
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Stem: ingest single-channel binary lineart / scribble
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=1, padding=3, bias=False),
            nn.GroupNorm(min(8, c), c),
            nn.SiLU(),
        )

        # Encoder stages
        self.enc1 = nn.Sequential(
            ResidualBlock(c),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),  # /2
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),  # /4
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            ResidualBlock(c * 4),
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),  # /8
            nn.SiLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c * 8),
            ResidualBlock(c * 8),
        )

        # Decoder stages: upsample -> conv cat -> residual
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # /8 -> /4
        self.dec3 = nn.Sequential(
            nn.Conv2d(c * 8 + c * 4, c * 4, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(c * 4),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # /4 -> /2
        self.dec2 = nn.Sequential(
            nn.Conv2d(c * 4 + c * 2, c * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(c * 2),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # /2 -> 1x
        self.dec1 = nn.Sequential(
            nn.Conv2d(c * 2 + c, c, kernel_size=3, padding=1),
            nn.SiLU(),
            ResidualBlock(c),
        )

        # Task Heads
        self.depth_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # Depth bounded [0, 1]
        )

        self.normal_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c // 2, 3, kernel_size=1),
            nn.Tanh(),  # Raw direction [-1, 1]
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # Foreground probability
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Handle 3-channel input by converting to 1-channel luminance if needed
        if x.shape[1] == 3:
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]

        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bottleneck(s3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, s2], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, s1], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, s0], dim=1))

        pred_depth = self.depth_head(d1)
        raw_normals = self.normal_head(d1)
        pred_normals = F.normalize(raw_normals, p=2, dim=1, eps=1e-6)
        pred_mask = self.mask_head(d1)

        return {
            "depth": pred_depth,
            "normal": pred_normals,
            "mask": pred_mask,
        }


# ── Geometric Ground Truth Generator & Dataset ─────────────────────────────

def _generate_synthetic_shape_pair(
    size: int = 256,
    shape_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates an analytic paired 3D geometric surface and its synthetic hand-drawn sketch contour.
    Returns:
        sketch: (H, W) uint8 in {0, 255} (black strokes on white or vice-versa)
        depth: (H, W) float32 in [0, 1]
        normals: (H, W, 3) float32 unit vectors in [-1, 1]
        mask: (H, W) float32 in {0.0, 1.0}
    """
    if shape_type is None:
        shape_type = random.choice(["vase", "sphere", "cylinder", "cone", "bottle", "bowl"])

    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)].astype(np.float32)

    depth = np.zeros((size, size), dtype=np.float32)
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[..., 2] = 1.0  # Default facing camera
    mask = np.zeros((size, size), dtype=np.float32)

    if shape_type == "sphere":
        r2 = x**2 + y**2
        valid = r2 <= 0.65**2
        z = np.sqrt(np.maximum(0.0, 0.65**2 - r2))
        depth[valid] = z[valid] / 0.65
        mask[valid] = 1.0

        normals[valid, 0] = x[valid] / 0.65
        normals[valid, 1] = y[valid] / 0.65
        normals[valid, 2] = z[valid] / 0.65

    elif shape_type == "cylinder":
        valid = (np.abs(y) <= 0.7) & (np.abs(x) <= 0.45)
        z = np.sqrt(np.maximum(0.0, 0.45**2 - x**2))
        depth[valid] = z[valid] / 0.45
        mask[valid] = 1.0

        normals[valid, 0] = x[valid] / 0.45
        normals[valid, 1] = 0.0
        normals[valid, 2] = z[valid] / 0.45

    elif shape_type == "cone":
        max_r = 0.55 * (0.8 - y) / 1.6
        r2 = x**2
        valid = (y >= -0.8) & (y <= 0.8) & (r2 <= np.maximum(0.01, max_r)**2)
        z = np.sqrt(np.maximum(0.0, np.maximum(0.01, max_r)**2 - r2))
        depth[valid] = z[valid] / (max_r[valid] + 1e-5)
        mask[valid] = 1.0

        normals[valid, 0] = x[valid] / (max_r[valid] + 1e-5)
        normals[valid, 1] = -0.3
        normals[valid, 2] = z[valid] / (max_r[valid] + 1e-5)

    elif shape_type == "bowl":
        r2 = x**2 + y**2
        valid = (r2 <= 0.7**2) & (y <= 0.1)
        z = np.sqrt(np.maximum(0.0, 0.7**2 - r2))
        depth[valid] = z[valid] / 0.7
        mask[valid] = 1.0

        normals[valid, 0] = x[valid] / 0.7
        normals[valid, 1] = y[valid] / 0.7
        normals[valid, 2] = z[valid] / 0.7

    else:  # vase / bottle
        profile = 0.35 + 0.18 * np.sin(y * math.pi * 2.0)
        valid = (np.abs(y) <= 0.75) & (np.abs(x) <= profile)
        z = np.sqrt(np.maximum(0.0, profile**2 - x**2))
        depth[valid] = z[valid] / (profile[valid] + 1e-5)
        mask[valid] = 1.0

        normals[valid, 0] = x[valid] / (profile[valid] + 1e-5)
        normals[valid, 1] = -0.18 * math.pi * 2.0 * np.cos(y[valid] * math.pi * 2.0)
        normals[valid, 2] = z[valid] / (profile[valid] + 1e-5)

    # Normalize normals vector length
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
    normals = normals / norm_len

    # Extract clean sketch silhouette contours (boundary + surface crease)
    from scipy.ndimage import binary_dilation, binary_erosion
    boundary = binary_dilation(mask, iterations=2) ^ binary_erosion(mask, iterations=2)

    # Add interior contour strokes (cross-contours simulating hand drawing)
    interior_crease = (np.abs(normals[..., 0]) < 0.05) & (mask > 0.5)
    sketch_binary = (boundary | interior_crease).astype(np.float32)

    # Slight random stroke thickening & jitter for hand-drawn variation
    if random.random() > 0.3:
        sketch_binary = binary_dilation(sketch_binary, iterations=1).astype(np.float32)

    return sketch_binary, depth, normals, mask


class SketchShapeDataset(Dataset):
    """
    Dataset supplying paired sketch inputs, depth targets, surface normals, and masks.
    Supports procedural generation and on-disk paired datasets.
    """

    def __init__(
        self,
        num_samples: int = 200,
        size: int = 128,
        augment: bool = True,
    ) -> None:
        self.num_samples = num_samples
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sketch, depth, normals, mask = _generate_synthetic_shape_pair(size=self.size)

        # Random horizontal flip augmentation
        if self.augment and random.random() > 0.5:
            sketch = np.fliplr(sketch).copy()
            depth = np.fliplr(depth).copy()
            normals = np.fliplr(normals).copy()
            normals[..., 0] = -normals[..., 0]  # Invert X normal on flip
            mask = np.fliplr(mask).copy()

        # Convert to PyTorch tensors
        # sketch: (1, H, W)
        sketch_t = torch.from_numpy(sketch).unsqueeze(0).float()
        # depth: (1, H, W)
        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        # normals: (3, H, W)
        normals_t = torch.from_numpy(normals).permute(2, 0, 1).float()
        # mask: (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "sketch": sketch_t,
            "depth": depth_t,
            "normal": normals_t,
            "mask": mask_t,
        }


# ── Loss Functions ──────────────────────────────────────────────────────────

class MultiTaskShapeLoss(nn.Module):
    """
    Balanced multi-task loss combining:
    1. Smooth L1 Depth Loss
    2. Spatial Depth Gradient Loss (sharp silhouette boundaries)
    3. Cosine Normal Angular Error Loss
    4. Binary Cross Entropy Mask Loss
    """

    def __init__(
        self,
        w_depth: float = 1.0,
        w_grad: float = 0.5,
        w_normal: float = 1.0,
        w_mask: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_depth = w_depth
        self.w_grad = w_grad
        self.w_normal = w_normal
        self.w_mask = w_mask

    def depth_gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx_pred = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        dx_target = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        dy_pred = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        dy_target = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p_depth = predictions["depth"]
        p_normal = predictions["normal"]
        p_mask = predictions["mask"]

        t_depth = targets["depth"]
        t_normal = targets["normal"]
        t_mask = targets["mask"]

        # 1. Depth loss (masked by target foreground)
        loss_depth = F.smooth_l1_loss(p_depth * t_mask, t_depth * t_mask)

        # 2. Gradient loss
        loss_grad = self.depth_gradient_loss(p_depth, t_depth)

        # 3. Normal loss: 1 - cosine similarity
        cos_sim = torch.sum(p_normal * t_normal, dim=1, keepdim=True)
        loss_normal = torch.mean((1.0 - cos_sim) * t_mask)

        # 4. Mask loss
        loss_mask = F.binary_cross_entropy(p_mask, t_mask)

        total_loss = (
            self.w_depth * loss_depth
            + self.w_grad * loss_grad
            + self.w_normal * loss_normal
            + self.w_mask * loss_mask
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_depth": float(loss_depth.item()),
            "loss_grad": float(loss_grad.item()),
            "loss_normal": float(loss_normal.item()),
            "loss_mask": float(loss_mask.item()),
        }

        return total_loss, metrics


# ── Training Loop & Evaluator ───────────────────────────────────────────────

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    resolution: int = 128
    num_train_samples: int = 160
    num_val_samples: int = 40
    device: str = "cpu"
    checkpoint_name: str = "sketch2shape_core.pt"


def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """
    Executes deep learning training of SketchToDepthNormalNet.
    Saves validated checkpoint to checkpoints/sketch2shape_core.pt.
    """
    device = torch.device(config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu")
    logger.info(f"Initializing Sketch2Shape Neural Training on device: {device}")

    # Build datasets and loaders
    train_ds = SketchShapeDataset(num_samples=config.num_train_samples, size=config.resolution, augment=True)
    val_ds = SketchShapeDataset(num_samples=config.num_val_samples, size=config.resolution, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Initialize model, optimizer, scheduler, and criterion
    model = SketchToDepthNormalNet(in_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-5)
    criterion = MultiTaskShapeLoss()

    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    start_time = time.perf_counter()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / config.checkpoint_name

    logger.info(f"Starting {config.epochs} training epochs...")

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            sketch = batch["sketch"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != "sketch"}

            optimizer.zero_grad()
            predictions = model(sketch)
            loss, _ = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))

        # Validation pass
        model.eval()
        val_losses: List[float] = []
        depth_maes: List[float] = []
        angular_errors: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                sketch = batch["sketch"].to(device)
                targets = {k: v.to(device) for k, v in batch.items() if k != "sketch"}

                predictions = model(sketch)
                loss, _ = criterion(predictions, targets)
                val_losses.append(loss.item())

                # Depth MAE
                t_mask = targets["mask"]
                mae = torch.sum(torch.abs(predictions["depth"] - targets["depth"]) * t_mask) / (torch.sum(t_mask) + 1e-6)
                depth_maes.append(mae.item())

                # Normal Angular Error in degrees
                cos_sim = torch.clamp(torch.sum(predictions["normal"] * targets["normal"], dim=1), -1.0, 1.0)
                angle_rad = torch.acos(cos_sim)
                angle_deg = torch.rad2deg(angle_rad)
                mean_angle = torch.sum(angle_deg * t_mask.squeeze(1)) / (torch.sum(t_mask) + 1e-6)
                angular_errors.append(mean_angle.item())

        avg_val_loss = float(np.mean(val_losses))
        avg_depth_mae = float(np.mean(depth_maes))
        avg_angular_deg = float(np.mean(angular_errors))

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "depth_mae": round(avg_depth_mae, 4),
            "angular_error_deg": round(avg_angular_deg, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Depth MAE: {avg_depth_mae:.4f} | "
            f"Normal Error: {avg_angular_deg:.1f}°"
        )

        # Save best weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )

    elapsed = round(time.perf_counter() - start_time, 2)
    logger.info(f"Training finished in {elapsed}s. Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Model saved -> {checkpoint_path}")

    # Write training report
    metrics_path = CHECKPOINTS_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "trained",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": str(device),
                "epochs": config.epochs,
                "best_val_loss": best_val_loss,
                "final_depth_mae": history[-1]["depth_mae"],
                "final_angular_error_deg": history[-1]["angular_error_deg"],
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "metrics": history[-1],
    }


def load_trained_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> Tuple[SketchToDepthNormalNet, Dict[str, Any]]:
    """
    Loads trained weights into SketchToDepthNormalNet and puts model in eval mode.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "sketch2shape_core.pt"

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    model = SketchToDepthNormalNet(in_channels=1, base_channels=32)
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch2Shape Neural Training")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    result = train_model(cfg)
    print("Training result:", json.dumps(result, indent=2))
