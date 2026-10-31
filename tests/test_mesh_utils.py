"""
tests/test_mesh_utils.py
========================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar

Unit tests for utils/mesh_utils.py (trimesh loading, manifold inspection,
7-step repair sequence, floating fragment removal, and multi-format export).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.mesh_utils import (
    export_mesh,
    inspect_mesh,
    is_watertight,
    load_mesh,
    repair_mesh,
    strip_isolated_components,
)


@pytest.fixture
def clean_sphere_mesh() -> trimesh.Trimesh:
    """Create a standard watertight UV sphere."""
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def mesh_with_floating_fragment(clean_sphere_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Combine a sphere with an isolated tiny floating triangle fragment."""
    # A tiny floating disconnected triangle
    floating_vertices = np.array([
        [5.0, 5.0, 5.0],
        [5.1, 5.0, 5.0],
        [5.0, 5.1, 5.0],
    ])
    floating_faces = np.array([[0, 1, 2]])
    frag = trimesh.Trimesh(vertices=floating_vertices, faces=floating_faces)

    combined = trimesh.util.concatenate([clean_sphere_mesh, frag])
    return combined


def test_inspect_mesh_metrics(clean_sphere_mesh: trimesh.Trimesh):
    metrics = inspect_mesh(clean_sphere_mesh)
    assert isinstance(metrics, dict)
    assert metrics["vertex_count"] > 0
    assert metrics["face_count"] > 0
    assert metrics["is_watertight"] is True
    assert metrics["is_winding_consistent"] is True
    assert metrics["volume"] > 0.0
    assert metrics["surface_area"] > 0.0
    assert metrics["body_count"] == 1


def test_is_watertight(clean_sphere_mesh: trimesh.Trimesh):
    assert is_watertight(clean_sphere_mesh) is True

    # Open sheet (not watertight)
    open_mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.0)
    assert is_watertight(open_mesh) is False


def test_strip_isolated_components(mesh_with_floating_fragment: trimesh.Trimesh):
    metrics_before = inspect_mesh(mesh_with_floating_fragment)
    assert metrics_before["body_count"] == 2

    cleaned = strip_isolated_components(mesh_with_floating_fragment, min_face_ratio=0.05)
    metrics_after = inspect_mesh(cleaned)
    assert metrics_after["body_count"] == 1
    assert len(cleaned.faces) < len(mesh_with_floating_fragment.faces)


def test_repair_mesh_sequence(mesh_with_floating_fragment: trimesh.Trimesh):
    repaired = repair_mesh(mesh_with_floating_fragment, strip_floating=True, center_and_scale=True)
    assert isinstance(repaired, trimesh.Trimesh)
    assert repaired.is_watertight is True
    # Bounding box should be centered near origin
    assert np.all(np.abs(repaired.centroid) < 1e-4)


def test_export_and_reload_glb(tmp_path: Path, clean_sphere_mesh: trimesh.Trimesh):
    out_file = tmp_path / "model.glb"
    exported_path = export_mesh(clean_sphere_mesh, out_file)
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    # Reload and verify geometry is intact
    reloaded = load_mesh(exported_path)
    assert len(reloaded.vertices) == len(clean_sphere_mesh.vertices)
    assert len(reloaded.faces) == len(clean_sphere_mesh.faces)


def test_export_obj_with_mtl_bundle(tmp_path: Path, clean_sphere_mesh: trimesh.Trimesh):
    out_obj = tmp_path / "model.obj"
    exported_path = export_mesh(clean_sphere_mesh, out_obj)
    assert exported_path.exists()

    # Verify OBJ text content
    content = exported_path.read_text(encoding="utf-8")
    assert "v " in content
    assert "f " in content


def test_export_ply_and_stl(tmp_path: Path, clean_sphere_mesh: trimesh.Trimesh):
    ply_path = export_mesh(clean_sphere_mesh, tmp_path / "model.ply")
    assert ply_path.exists()

    stl_path = export_mesh(clean_sphere_mesh, tmp_path / "model.stl")
    assert stl_path.exists()
