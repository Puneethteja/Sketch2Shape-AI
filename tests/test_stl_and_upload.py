"""
tests/test_stl_and_upload.py
============================
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Lead Engineer : Mona (Generative AI & 3D Reconstruction)

Verification suite for:
  1. Remove Uploaded File:
     - Clear uploaded file state, preview image, and session references without app restart.
  2. STL Functionality:
     - Valid binary STL export with correct 80-byte header, uint32 facet count,
       and 50-byte IEEE 754 float32 records.
     - Integration with ui.app._run_mesh_export("STL").
     - Integration with WebGL viewer STLExporter.
"""

from __future__ import annotations

import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import ui.app as uiapp
from engine.mesh_utils import TrimeshFallback
from engine.pipeline import Sketch2ShapePipeline


class TestUploadFileRemoval(unittest.TestCase):
    """Verifies file upload removal, preview clearing, and session state reset."""

    def setUp(self):
        uiapp._SESSION["current_source"] = "test_uploaded_file.png"
        uiapp._SESSION["current_mesh"] = None

    def test_clear_uploaded_file_resets_state_and_preview(self):
        file_val, preview_val, status_msg = uiapp._clear_uploaded_file()

        # File input widget should receive None to clear UI state
        self.assertIsNone(file_val)
        # Preprocessed preview should be cleared
        self.assertIsNone(preview_val)
        # Session state should be purged
        self.assertIsNone(uiapp._SESSION.get("current_source"))
        # Informative status message returned
        self.assertIn("Uploaded file removed", status_msg)

    def test_new_upload_after_removal_resolves_cleanly(self):
        # 1. Simulate initial upload
        uiapp._SESSION["current_source"] = "sketch1.png"
        self.assertEqual(uiapp._SESSION["current_source"], "sketch1.png")

        # 2. User clicks remove (x) button
        uiapp._clear_uploaded_file()
        self.assertIsNone(uiapp._SESSION["current_source"])

        # 3. User uploads new image without refreshing
        new_source = uiapp._resolve_input_source(
            canvas_image=None,
            uploaded_file="sketch2.png",
            canvas_data_url=None,
        )
        self.assertEqual(new_source, "sketch2.png")


class TestSTLFunctionality(unittest.TestCase):
    """Verifies valid STL geometry conversion, header, facet data, and downloading."""

    @classmethod
    def setUpClass(cls):
        # Build canonical hollow vase mesh for testing
        cls.pipeline = Sketch2ShapePipeline(force_cpu=True)
        vase_sketch = Path("outputs/samples/sample_01_vase/input_sketch.png")
        if vase_sketch.exists():
            cls.gen_result = cls.pipeline.run(
                sketch_source=str(vase_sketch),
                prompt="ceramic vase",
                seed=101,
            )
            cls.test_mesh = cls.gen_result.mesh
        else:
            verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
            faces = np.array([[0,1,2], [0,2,3], [0,3,1], [1,3,2]], dtype=np.int32)
            cls.test_mesh = TrimeshFallback(verts, faces, is_watertight=True)

    def test_mesh_export_stl_binary_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stl_path = Path(tmpdir) / "test_vase.stl"
            self.test_mesh.export(str(stl_path), file_type="stl")

            self.assertTrue(stl_path.exists())
            file_size = stl_path.stat().st_size
            n_faces = len(self.test_mesh.faces)

            # Standard Binary STL size formula: 80 bytes header + 4 bytes uint32 count + 50 bytes per facet
            expected_size = 80 + 4 + (50 * n_faces)
            self.assertEqual(file_size, expected_size)

            # Verify header does NOT begin with 'solid' to avoid ASCII parser confusion in slicers
            with open(stl_path, "rb") as f:
                header = f.read(80)
                facet_count = struct.unpack("<I", f.read(4))[0]
                self.assertEqual(facet_count, n_faces)
                self.assertFalse(header.lower().startswith(b"solid "))

                # Inspect first facet (50 bytes): normal (3 floats), 3 vertices (9 floats), attr (uint16)
                facet_bytes = f.read(50)
                self.assertEqual(len(facet_bytes), 50)
                normal = struct.unpack("<3f", facet_bytes[:12])
                norm_mag = np.linalg.norm(normal)
                # Normal must be finite and normalized
                self.assertTrue(np.isfinite(norm_mag))
                self.assertAlmostEqual(norm_mag, 1.0, places=3)

    def test_run_mesh_export_stl_integration(self):
        uiapp._LAST_GENERATED_MESH = self.test_mesh
        uiapp._SESSION["current_mesh"] = self.test_mesh

        out_path_str, status_msg = uiapp._run_mesh_export("STL")
        self.assertIsNotNone(out_path_str)
        self.assertTrue(os.path.exists(out_path_str))
        self.assertTrue(out_path_str.endswith(".stl"))
        self.assertIn("[OK] Exported active mesh to STL", status_msg)
        self.assertGreater(os.path.getsize(out_path_str), 1000)

    def test_viewer_js_includes_stl_support(self):
        viewer_js_path = Path("ui/viewer.js")
        self.assertTrue(viewer_js_path.exists())
        content = viewer_js_path.read_text(encoding="utf-8")

        # Must have STLExporter url and method
        self.assertIn("downloadSTL", content)
        self.assertIn("exportSceneToSTL", content)
        self.assertIn("Download .STL", content)
        self.assertIn("cdnStlExporterUrl", content)


if __name__ == "__main__":
    unittest.main()
