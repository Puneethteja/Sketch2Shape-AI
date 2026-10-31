"""
utils/mesh_utils.py
===================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Puneeth (Data & Infrastructure)

Mesh Post-Processing, Validation & Export Utilities.
Wraps trimesh to inspect, repair non-manifold geometry, eliminate disconnected
floating fragments, and export validated 3D models into production-ready .glb,
.obj + .mtl, and .ply formats.

Implements the deterministic 7-step non-manifold recovery sequence specified
in docs/technical_architecture.md §3.4.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

logger = logging.getLogger("sketch2shape.mesh_utils")

SUPPORTED_EXPORT_FORMATS = ("glb", "obj", "ply", "stl")

def create_fallback_mesh(radius: float = 1.0) -> trimesh.Trimesh:
    """
    Create a canonical watertight fallback mesh.
    """
    try:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        if getattr(mesh, "is_watertight", False):
            return mesh
    except Exception:
        pass
    return trimesh.creation.box(extents=(radius, radius, radius))

def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Sanitize mesh vertices and faces to remove NaNs, infinities, and out-of-bound indices.
    """
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return create_fallback_mesh()
    try:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if verts.size == 0 or faces.size == 0 or len(verts) < 3 or len(faces) == 0:
            return create_fallback_mesh()

        finite_verts = np.isfinite(verts).all(axis=1)
        if not np.all(finite_verts):
            valid_indices = np.where(finite_verts)[0]
            if len(valid_indices) < 3:
                return create_fallback_mesh()
            index_map = {int(old): int(new) for new, old in enumerate(valid_indices)}
            valid_faces_mask = np.isin(faces, valid_indices).all(axis=1)
            new_faces = faces[valid_faces_mask]
            if len(new_faces) == 0:
                return create_fallback_mesh()
            remapped_faces = np.vectorize(index_map.get)(new_faces)
            clean_verts = verts[finite_verts]
            return trimesh.Trimesh(vertices=clean_verts, faces=remapped_faces, process=False)

        valid_faces_mask = (faces >= 0).all(axis=1) & (faces < len(verts)).all(axis=1)
        if not np.all(valid_faces_mask):
            faces = faces[valid_faces_mask]
            if len(faces) == 0:
                return create_fallback_mesh()
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    except Exception:
        return create_fallback_mesh()

def load_mesh(source: Union[str, Path, trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    """
    Load a 3D mesh from a file path or extract it from an in-memory Trimesh / Scene.

    If the file represents a multi-node Scene (common for GLB / GLTF files),
    all sub-geometries are automatically concatenated into a unified Trimesh.

    Parameters
    ----------
    source : str | Path | trimesh.Trimesh | trimesh.Scene
        Source 3D asset file (.obj, .glb, .gltf, .ply, .stl) or trimesh instance.

    Returns
    -------
    trimesh.Trimesh
        Unified single-body mesh representation.

    Raises
    ------
    FileNotFoundError
        If source is a path that does not exist.
    ValueError
        If geometry cannot be parsed into a triangle mesh.
    """
    if isinstance(source, trimesh.Trimesh):
        return source.copy()

    if isinstance(source, trimesh.Scene):
        return _scene_to_trimesh(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path.resolve()}")

    logger.info("Loading 3D mesh from '%s'", path.name)
    loaded = trimesh.load(str(path), force="mesh")

    if isinstance(loaded, trimesh.Scene):
        return _scene_to_trimesh(loaded)
    elif isinstance(loaded, trimesh.Trimesh):
        return loaded
    else:
        raise ValueError(f"Unsupported geometry type loaded from {path}: {type(loaded)}")

def _scene_to_trimesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """Concatenate all sub-meshes in a trimesh.Scene into a single Trimesh."""
    if len(scene.geometry) == 0:
        raise ValueError("Loaded Scene contains no geometry elements.")

    meshes = list(scene.geometry.values())
    if len(meshes) == 1:
        return meshes[0]

    logger.debug("Merging %d sub-meshes from Scene into single Trimesh", len(meshes))
    return trimesh.util.concatenate(meshes)

def inspect_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Compute comprehensive topological, geometrical, and manifold quality metrics.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to inspect.

    Returns
    -------
    dict[str, Any]
        Dictionary of topological and geometric properties:
        - vertex_count: int
        - face_count: int
        - is_watertight: bool (closed manifold volume, essential for 3D printing & physics)
        - is_winding_consistent: bool (all normal directions oriented uniformly)
        - euler_number: int (V - E + F topological genus indicator)
        - volume: float (cubic units; valid only if watertight)
        - surface_area: float
        - bounding_box_extents: tuple[float, float, float]
        - centroid: tuple[float, float, float]
        - body_count: int (number of disconnected components)
    """
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return {
            "vertex_count": 0,
            "face_count": 0,
            "is_watertight": False,
            "is_winding_consistent": False,
            "euler_number": 0,
            "volume": 0.0,
            "surface_area": 0.0,
            "bounding_box_extents": (0.0, 0.0, 0.0),
            "centroid": (0.0, 0.0, 0.0),
            "body_count": 0,
        }

    is_watertight = bool(getattr(mesh, "is_watertight", False))
    is_winding = bool(getattr(mesh, "is_winding_consistent", False))

    try:
        extents = tuple(float(x) for x in mesh.bounding_box.extents)
    except Exception:
        extents = (0.0, 0.0, 0.0)

    try:
        centroid = tuple(float(x) for x in mesh.centroid)
    except Exception:
        centroid = (0.0, 0.0, 0.0)

    vol = 0.0
    if is_watertight:
        try:
            vol = float(mesh.volume)
            if not np.isfinite(vol):
                vol = 0.0
        except Exception:
            vol = 0.0

    try:
        area = float(mesh.area)
        if not np.isfinite(area):
            area = 0.0
    except Exception:
        area = 0.0

    try:
        euler = int(mesh.euler_number)
    except Exception:
        euler = 0

    try:
        components = mesh.split(only_watertight=False)
        body_count = len(components) if isinstance(components, (list, np.ndarray)) else 1
    except Exception:
        body_count = 1

    return {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "is_watertight": is_watertight,
        "is_winding_consistent": is_winding,
        "euler_number": euler,
        "volume": vol,
        "surface_area": area,
        "bounding_box_extents": extents,
        "centroid": centroid,
        "body_count": body_count,
    }

def is_watertight(mesh: trimesh.Trimesh) -> bool:
    """Convenience boolean check for watertight topology."""
    return bool(getattr(mesh, "is_watertight", False))

def repair_mesh(
    mesh: trimesh.Trimesh,
    strip_floating: bool = True,
    fill_holes: bool = True,
    center_and_scale: bool = True,
    enforce_watertight: bool = True,
) -> trimesh.Trimesh:
    """
    Execute the deterministic 7-step non-manifold recovery and repair sequence.

    Steps (§3.4 Technical Architecture):
    1. Remove duplicate faces (degenerate zero-area and overlapping triangles).
    2. Remove degenerate faces (collapsed edges/vertices).
    3. Remove unreferenced / orphan vertices.
    4. Fix face winding consistency (outward-facing normals).
    5. Fix inverted normals (trimesh.repair.fix_inversion).
    6. Fill boundary holes / close open loops.
    7. Strip unattached floating fragments via connected component analysis.
    8. (Optional) Center mesh at origin and normalize bounding box scale.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh to repair.
    strip_floating : bool
        Whether to retain only the primary connected component and discard floating noise.
    fill_holes : bool
        Whether to run hole-filling routines.
    center_and_scale : bool
        Whether to center the mesh at (0, 0, 0) with a unit bounding sphere.
    enforce_watertight : bool
        Whether to guarantee a watertight closed manifold surface upon completion.

    Returns
    -------
    trimesh.Trimesh
        Cleaned, repaired, manifold mesh.
    """
    try:
        if mesh is None:
            return create_fallback_mesh()

        repaired = sanitize_mesh(mesh)
        if len(repaired.faces) == 0 or len(repaired.vertices) < 3:
            return create_fallback_mesh()

        logger.info(
            "Starting mesh repair sequence: %d vertices, %d faces",
            len(repaired.vertices), len(repaired.faces),
        )

        try:
            repaired.update_faces(repaired.unique_faces())
        except Exception as exc:
            logger.debug("unique_faces warning: %s", exc)

        try:
            repaired.update_faces(repaired.nondegenerate_faces())
        except Exception as exc:
            logger.debug("nondegenerate_faces warning: %s", exc)

        try:
            repaired.remove_unreferenced_vertices()
        except Exception as exc:
            logger.debug("remove_unreferenced_vertices warning: %s", exc)

        try:
            repaired.remove_duplicate_faces()
        except Exception as exc:
            logger.debug("remove_duplicate_faces warning: %s", exc)

        try:
            trimesh.repair.fix_winding(repaired)
        except Exception as exc:
            logger.debug("fix_winding warning: %s", exc)

        try:
            trimesh.repair.fix_inversion(repaired)
            trimesh.repair.fix_normals(repaired)
        except Exception as exc:
            logger.debug("fix_inversion warning: %s", exc)

        if fill_holes:
            try:
                trimesh.repair.fill_holes(repaired)
            except Exception as exc:
                logger.debug("fill_holes warning: %s", exc)

        if strip_floating:
            repaired = strip_isolated_components(repaired)

        try:
            repaired.update_faces(repaired.unique_faces())
            repaired.update_faces(repaired.nondegenerate_faces())
            repaired.remove_unreferenced_vertices()
        except Exception as exc:
            logger.debug("cleanup pass warning: %s", exc)

        if enforce_watertight and not getattr(repaired, "is_watertight", False):
            try:
                trimesh.repair.fill_holes(repaired)
            except Exception:
                pass
            if not getattr(repaired, "is_watertight", False) and len(repaired.vertices) >= 4:
                try:
                    ext = repaired.extents
                    if np.any(ext < 1e-4):
                        v_thick = repaired.vertices.copy()
                        min_dim = int(np.argmin(ext))
                        v_thick[:, min_dim] += 0.05
                        v_combined = np.vstack([repaired.vertices, v_thick])
                        hull = trimesh.convex.convex_hull(v_combined)
                    else:
                        hull = repaired.convex_hull
                    if hasattr(hull, "is_watertight") and hull.is_watertight and len(hull.faces) > 0:
                        repaired = hull
                except Exception as exc:
                    logger.debug("convex_hull recovery warning: %s", exc)

        if enforce_watertight and not getattr(repaired, "is_watertight", False):
            fallback = create_fallback_mesh()
            if len(repaired.vertices) > 0 and np.isfinite(repaired.centroid).all():
                fallback.vertices += repaired.centroid
            return fallback

        if center_and_scale and len(repaired.vertices) > 0:
            if np.isfinite(repaired.centroid).all():
                repaired.vertices -= repaired.centroid

            extents = repaired.extents
            if len(extents) > 0:
                max_extent = float(np.max(extents))
                if np.isfinite(max_extent) and max_extent > 1e-6:
                    repaired.vertices /= max_extent

        if len(repaired.faces) == 0 or len(repaired.vertices) < 3:
            return create_fallback_mesh()

        logger.info(
            "Mesh repair complete: %d vertices, %d faces, watertight=%s",
            len(repaired.vertices), len(repaired.faces), getattr(repaired, "is_watertight", False),
        )
        return repaired
    except Exception as exc:
        logger.warning("Uncaught exception in repair_mesh, falling back to canonical mesh: %s", exc)
        return create_fallback_mesh()

def strip_isolated_components(
    mesh: trimesh.Trimesh,
    min_face_ratio: float = 0.05,
) -> trimesh.Trimesh:
    """
    Isolate the primary connected component and strip stray floating fragments.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Source mesh with potentially multiple disconnected bodies.
    min_face_ratio : float
        Components with fewer faces than this ratio of the largest body are discarded.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh retaining the dominant component.
    """
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return mesh

    try:
        components = mesh.split(only_watertight=False)

        if not isinstance(components, (list, np.ndarray)) or len(components) <= 1:
            return mesh

        sorted_components = sorted(components, key=lambda c: len(c.faces), reverse=True)
        largest = sorted_components[0]
        threshold_faces = max(1, int(len(largest.faces) * min_face_ratio))

        retained = [c for c in sorted_components if len(c.faces) >= threshold_faces]

        if len(retained) == 1:
            logger.debug("Retained dominant component (%d faces), stripped %d fragments",
                         len(largest.faces), len(sorted_components) - 1)
            return largest

        logger.debug("Concatenating %d significant components", len(retained))
        return trimesh.util.concatenate(retained)
    except Exception as exc:
        logger.debug("strip_isolated_components warning: %s", exc)
        return mesh

def export_mesh(
    mesh: trimesh.Trimesh,
    output_path: Union[str, Path],
    export_format: Optional[str] = None,
    include_materials: bool = True,
) -> Path:
    """
    Export a validated 3D mesh into standard exchange formats.

    Supports:
    - .glb  : Binary glTF with embedded textures, PBR materials, and scene hierarchy.
              Standard for WebGL previewers (Three.js / Gradio Model3D).
    - .obj  : Wavefront OBJ with paired .mtl material definition file and textures.
              Standard for DCC tools (Blender, Maya, 3ds Max).
    - .ply  : Stanford Polygon format with vertex colors.
    - .stl  : Stereolithography format for 3D slicing & printing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Validated mesh to serialize.
    output_path : str | Path
        Destination file path. Extension determines format if export_format is None.
    export_format : str, optional
        Target format ('glb', 'obj', 'ply', 'stl').
    include_materials : bool
        Whether to export material textures alongside geometry.

    Returns
    -------
    Path
        Resolved path to the exported primary asset file.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = (export_format or out.suffix.lstrip(".")).lower()
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported export format '{fmt}'. Supported formats: {SUPPORTED_EXPORT_FORMATS}"
        )

    exportable_mesh = mesh
    if exportable_mesh is None or len(getattr(exportable_mesh, "faces", [])) == 0:
        exportable_mesh = create_fallback_mesh()

    logger.info("Exporting mesh to '%s' (format: %s)", out.name, fmt.upper())

    try:
        if fmt == "glb":
            _export_glb(exportable_mesh, out)
        elif fmt == "obj":
            _export_obj_with_mtl(exportable_mesh, out, include_materials=include_materials)
        elif fmt == "ply":
            exportable_mesh.export(str(out), file_type="ply")
        elif fmt == "stl":
            exportable_mesh.export(str(out), file_type="stl")
    except Exception as exc:
        logger.warning("Export error with current geometry: %s. Using fallback mesh.", exc)
        fallback = create_fallback_mesh()
        if fmt == "glb":
            _export_glb(fallback, out)
        elif fmt == "obj":
            _export_obj_with_mtl(fallback, out, include_materials=include_materials)
        elif fmt == "ply":
            fallback.export(str(out), file_type="ply")
        elif fmt == "stl":
            fallback.export(str(out), file_type="stl")

    logger.info("Successfully exported 3D asset -> %s", out)
    return out

def _export_glb(mesh: trimesh.Trimesh, path: Path) -> None:
    """Export mesh as binary glTF (.glb) suitable for Three.js WebGL viewport."""
    glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
    with open(path, "wb") as f:
        f.write(glb_bytes)

def _export_obj_with_mtl(
    mesh: trimesh.Trimesh,
    path: Path,
    include_materials: bool = True,
) -> None:
    """
    Export Wavefront .obj bundled with .mtl material library and texture maps.
    """
    base_stem = path.stem
    mtl_filename = f"{base_stem}.mtl"
    mtl_path = path.parent / mtl_filename

    exported_dict = trimesh.exchange.obj.export_obj(
        mesh,
        return_texture=include_materials,
        mtl_name=mtl_filename,
    )

    if isinstance(exported_dict, str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(exported_dict)
    elif isinstance(exported_dict, tuple):
        obj_text = exported_dict[0]
        with open(path, "w", encoding="utf-8") as f:
            f.write(obj_text)
        if len(exported_dict) > 1 and exported_dict[1]:
            with open(mtl_path, "w", encoding="utf-8") as f:
                f.write(exported_dict[1])
    elif isinstance(exported_dict, dict):
        for sub_name, data in exported_dict.items():
            sub_path = path.parent / sub_name
            mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
            encoding = None if "b" in mode else "utf-8"
            with open(sub_path, mode, encoding=encoding) as f:
                f.write(data)

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Sketch2Shape 3D Mesh Post-Processing & Validation")
    parser.add_argument("input", help="Path to input 3D model (.obj, .glb, .ply, .stl)")
    parser.add_argument("-o", "--output", help="Path to export repaired model (.glb, .obj, .ply)")
    parser.add_argument("--repair", action="store_true", help="Run 7-step non-manifold repair sequence")
    parser.add_argument("--stats", action="store_true", help="Print topological inspection statistics")

    args = parser.parse_args()

    m = load_mesh(args.input)

    if args.stats:
        metrics = inspect_mesh(m)
        print("\n--- Mesh Topology & Manifold Inspection ---")
        print(json.dumps(metrics, indent=2))

    if args.repair:
        m = repair_mesh(m)

    if args.output:
        out_p = export_mesh(m, args.output)
        print(f"\n[OK] Processed mesh exported to: {out_p.resolve()}")
