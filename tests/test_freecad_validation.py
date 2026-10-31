"""
Unit test for FreeCAD validation script and CAD topology metrics.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.test_with_freecad import compute_mesh_metrics, parse_obj, parse_stl

class TestFreeCADValidation(unittest.TestCase):
    def test_tetrahedron_solid(self):
        # Unit regular tetrahedron (closed solid)
        verts = [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]
        faces = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
        m = compute_mesh_metrics(verts, faces)
        self.assertTrue(m["is_solid"], "Closed tetrahedron must be classified as a solid")
        self.assertEqual(m["boundary_edges"], 0)
        self.assertEqual(m["non_manifold_edges"], 0)
        self.assertEqual(m["euler_characteristic"], 2)
        self.assertGreater(m["volume"], 0.0)

    def test_open_sheet_not_solid(self):
        # Single triangle quad sheet (has boundary edges)
        verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        faces = [[0, 1, 2], [0, 2, 3]]
        m = compute_mesh_metrics(verts, faces)
        self.assertFalse(m["is_solid"], "Open planar sheet must NOT be solid")
        self.assertGreater(m["boundary_edges"], 0)

    def test_sketch2shape_stl_asset(self):
        stl_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "sketch2shape_asset.stl")
        if not os.path.exists(stl_path):
            self.skipTest("sketch2shape_asset.stl not present")
        v, f = parse_stl(stl_path)
        m = compute_mesh_metrics(v, f)
        self.assertTrue(m["is_solid"], "Exported STL asset must be a valid watertight CAD solid")
        self.assertEqual(m["boundary_edges"], 0)
        self.assertEqual(m["non_manifold_edges"], 0)


if __name__ == "__main__":
    unittest.main()
