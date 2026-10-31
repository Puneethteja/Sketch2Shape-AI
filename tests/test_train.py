"""
tests/test_train.py
===================
Unit tests for Sketch2Shape neural network architecture, multi-task loss functions,
and synthetic dataset generators.
"""

import pytest
import torch
import numpy as np

from engine.train import (
    SketchToDepthNormalNet,
    MultiTaskShapeLoss,
    SketchShapeDataset,
    _generate_synthetic_shape_pair,
)


def test_synthetic_shape_generation():
    sketch, depth, normals, mask = _generate_synthetic_shape_pair(size=64, shape_type="vase")
    assert sketch.shape == (64, 64)
    assert depth.shape == (64, 64)
    assert normals.shape == (64, 64, 3)
    assert mask.shape == (64, 64)

    # Check bounds
    assert np.all(depth >= 0.0) and np.all(depth <= 1.0)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    # Check normal vectors length is approximately 1.0 on mask
    valid = mask > 0.5
    lengths = np.linalg.norm(normals[valid], axis=-1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-3)


def test_dataset_item_format():
    ds = SketchShapeDataset(num_samples=4, size=64, augment=False)
    assert len(ds) == 4
    item = ds[0]
    assert "sketch" in item and "depth" in item and "normal" in item and "mask" in item
    assert item["sketch"].shape == (1, 64, 64)
    assert item["depth"].shape == (1, 64, 64)
    assert item["normal"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)


def test_model_forward_pass():
    model = SketchToDepthNormalNet(in_channels=1, base_channels=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)

    assert "depth" in out
    assert "normal" in out
    assert "mask" in out

    assert out["depth"].shape == (2, 1, 64, 64)
    assert out["normal"].shape == (2, 3, 64, 64)
    assert out["mask"].shape == (2, 1, 64, 64)

    # Depth bounded [0, 1]
    assert torch.all(out["depth"] >= 0.0) and torch.all(out["depth"] <= 1.0)
    # Normals normalized to unit length
    norm_lens = torch.norm(out["normal"], dim=1)
    torch.testing.assert_close(norm_lens, torch.ones_like(norm_lens), atol=1e-4, rtol=1e-4)


def test_multitask_loss_backward():
    model = SketchToDepthNormalNet(in_channels=1, base_channels=16)
    criterion = MultiTaskShapeLoss()

    x = torch.randn(2, 1, 64, 64)
    targets = {
        "depth": torch.rand(2, 1, 64, 64),
        "normal": torch.randn(2, 3, 64, 64),
        "mask": torch.ones(2, 1, 64, 64),
    }
    targets["normal"] = torch.nn.functional.normalize(targets["normal"], dim=1)

    preds = model(x)
    loss, metrics = criterion(preds, targets)

    assert loss.item() > 0.0
    assert "loss_depth" in metrics
    assert "loss_normal" in metrics

    # Test backward pass
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
