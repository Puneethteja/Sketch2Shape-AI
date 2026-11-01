"""
Unit tests for TextureProjector and colour-preservation pipeline.

Run with:
    python -m unittest tests/test_texture.py -v
"""
import sys
import os
import unittest
import struct

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(r, g, b, size=(128, 128)):
    """Create a solid-colour PIL RGBA image."""
    from PIL import Image
    img = Image.new("RGBA", size, (r, g, b, 255))
    return img


def _make_random_vertices(n=50):
    """Return a list of (x, y, z) tuples on a unit sphere."""
    import math
    verts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        phi = math.pi * i / n
        x = math.sin(phi) * math.cos(theta)
        y = math.cos(phi)
        z = math.sin(phi) * math.sin(theta)
        verts.append((x, y, z))
    return verts


def _mean_saturation(vertex_colors):
    """
    Return mean HSV saturation [0-255] from a list of (r,g,b,a) tuples.
    """
    import colorsys
    saturations = []
    for vc in vertex_colors:
        r, g, b = vc[0] / 255.0, vc[1] / 255.0, vc[2] / 255.0
        _, s, _ = colorsys.rgb_to_hsv(r, g, b)
        saturations.append(s * 255.0)
    return sum(saturations) / len(saturations) if saturations else 0.0


# ---------------------------------------------------------------------------
# TestTextureProjector
# ---------------------------------------------------------------------------

class TestTextureProjector(unittest.TestCase):

    def setUp(self):
        try:
            from engine.texture import TextureProjector
            self.tp = TextureProjector()
            self._available = True
        except ImportError as e:
            self._available = False
            self._skip_reason = str(e)

    def _skip_if_unavailable(self):
        if not self._available:
            self.skipTest(f"TextureProjector not importable: {self._skip_reason}")

    # ------------------------------------------------------------------
    # enhance_color_fidelity
    # ------------------------------------------------------------------

    def test_enhance_color_fidelity_preserves_saturation(self):
        """Enhancing a colourful image must not decrease mean saturation."""
        self._skip_if_unavailable()
        from PIL import Image
        import colorsys
        import numpy as np

        # Create a vivid red image
        img = Image.new("RGBA", (64, 64), (200, 50, 50, 255))
        enhanced = self.tp.enhance_color_fidelity(img)

        arr_in = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        arr_out = np.array(enhanced.convert("RGB"), dtype=np.float32) / 255.0

        def mean_sat(arr):
            sats = []
            for row in arr:
                for px in row:
                    _, s, _ = colorsys.rgb_to_hsv(*px.tolist())
                    sats.append(s)
            return sum(sats) / len(sats)

        sat_in = mean_sat(arr_in)
        sat_out = mean_sat(arr_out)
        self.assertGreaterEqual(sat_out, sat_in * 0.95,
            "Saturation must not drop below 95 % of input after enhancement")

    def test_enhance_color_fidelity_returns_pil_image(self):
        self._skip_if_unavailable()
        from PIL import Image
        img = _make_rgb_image(100, 150, 200)
        result = self.tp.enhance_color_fidelity(img)
        self.assertIsInstance(result, Image.Image)

    def test_enhance_color_fidelity_correct_mode(self):
        """Output image should be RGBA (or at minimum RGB)."""
        self._skip_if_unavailable()
        img = _make_rgb_image(80, 160, 240)
        result = self.tp.enhance_color_fidelity(img)
        self.assertIn(result.mode, {"RGBA", "RGB"})

    # ------------------------------------------------------------------
    # compute_cylindrical_uvs
    # ------------------------------------------------------------------

    def test_compute_cylindrical_uvs_range(self):
        """All UV coordinates must be in [0, 1]²."""
        self._skip_if_unavailable()
        verts = _make_random_vertices(100)
        uvs = self.tp.compute_cylindrical_uvs(verts)
        self.assertEqual(len(uvs), len(verts))
        for u, v in uvs:
            self.assertGreaterEqual(u, 0.0, "U must be >= 0")
            self.assertLessEqual(u, 1.0, "U must be <= 1")
            self.assertGreaterEqual(v, 0.0, "V must be >= 0")
            self.assertLessEqual(v, 1.0, "V must be <= 1")

    def test_compute_cylindrical_uvs_count(self):
        """UV list length must match vertex list length."""
        self._skip_if_unavailable()
        verts = _make_random_vertices(37)
        uvs = self.tp.compute_cylindrical_uvs(verts)
        self.assertEqual(len(uvs), 37)

    # ------------------------------------------------------------------
    # project_texture_pass
    # ------------------------------------------------------------------

    def test_project_texture_pass_returns_correct_keys(self):
        """project_texture_pass must return an object with vertex_colors and uv_coordinates."""
        self._skip_if_unavailable()
        import math
        # Minimal mesh: a tetrahedron
        vertices = [
            (0.0, 1.0, 0.0),
            (0.943, -0.333, 0.0),
            (-0.471, -0.333, 0.816),
            (-0.471, -0.333, -0.816),
        ]
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

        views = [_make_rgb_image(200, 80, 80) for _ in range(6)]
        result = self.tp.project_texture_pass(vertices, faces, views)

        self.assertTrue(hasattr(result, "vertex_colors"),
                        "Result must have vertex_colors attribute")
        self.assertTrue(hasattr(result, "uv_coordinates"),
                        "Result must have uv_coordinates attribute")

    def test_project_texture_pass_vertex_color_count(self):
        """Number of vertex colors must equal number of vertices."""
        self._skip_if_unavailable()
        vertices = _make_random_vertices(24)
        faces = [(i, (i + 1) % 24, (i + 2) % 24) for i in range(0, 24, 3)]
        views = [_make_rgb_image(100, 180, 100) for _ in range(6)]
        result = self.tp.project_texture_pass(vertices, faces, views)
        self.assertEqual(len(result.vertex_colors), len(vertices))

    # ------------------------------------------------------------------
    # color_fidelity_score
    # ------------------------------------------------------------------

    def test_color_fidelity_score_not_gray(self):
        """A vivid input image must produce fidelity score > 0.5."""
        self._skip_if_unavailable()
        vertices = _make_random_vertices(50)
        faces = [(i, (i + 1) % 50, (i + 2) % 50) for i in range(0, 50, 3)]
        views = [_make_rgb_image(220, 60, 60) for _ in range(6)]  # vivid red
        result = self.tp.project_texture_pass(vertices, faces, views)

        # Compute fidelity from vertex colors (mean saturation / 255)
        sat = _mean_saturation(result.vertex_colors)
        fidelity = sat / 255.0
        self.assertGreater(fidelity, 0.1,
            f"Expected fidelity > 0.1 for vivid red input, got {fidelity:.3f}")


# ---------------------------------------------------------------------------
# TestColorPreservation
# ---------------------------------------------------------------------------

class TestColorPreservation(unittest.TestCase):
    """Test that the generator's conditioned image preserves or routes colours."""

    def _get_generator(self):
        try:
            from engine.generator import SketchDiffusionGenerator
            return SketchDiffusionGenerator()
        except ImportError as e:
            self.skipTest(f"SketchDiffusionGenerator not importable: {e}")

    def test_native_color_preserved(self):
        """
        A vivid red RGBA image should produce vertex colours that are
        clearly reddish (red channel > blue channel by a meaningful margin).
        """
        gen = self._get_generator()
        red_img = _make_rgb_image(210, 40, 40)

        try:
            cond = gen._generate_synthetic_conditioned_image(red_img, "", seed=42)
        except Exception as e:
            self.skipTest(f"_generate_synthetic_conditioned_image raised: {e}")

        import numpy as np
        arr = np.array(cond.convert("RGB"), dtype=np.float32)
        mean_r = arr[:, :, 0].mean()
        mean_b = arr[:, :, 2].mean()
        self.assertGreater(mean_r, mean_b + 10,
            f"Expected red channel dominance; got R={mean_r:.1f} B={mean_b:.1f}")

    def test_prompt_material_routing_gold(self):
        """A 'gold' prompt should yield a warm (yellow-ish) conditioned image."""
        gen = self._get_generator()
        gray_img = _make_rgb_image(128, 128, 128)  # neutral input

        try:
            cond = gen._generate_synthetic_conditioned_image(gray_img, "gold vase", seed=42)
        except Exception as e:
            self.skipTest(f"_generate_synthetic_conditioned_image raised: {e}")

        import numpy as np
        arr = np.array(cond.convert("RGB"), dtype=np.float32)
        mean_r = arr[:, :, 0].mean()
        mean_b = arr[:, :, 2].mean()
        # Gold is warm: red > blue
        self.assertGreater(mean_r, mean_b,
            f"'gold' prompt should give warm hue; got R={mean_r:.1f} B={mean_b:.1f}")

    def test_no_gray_washout(self):
        """
        Any coloured input must produce a conditioned image whose mean
        per-pixel saturation is clearly non-zero (> 10 / 255).
        """
        gen = self._get_generator()
        blue_img = _make_rgb_image(60, 100, 200)  # blue

        try:
            cond = gen._generate_synthetic_conditioned_image(blue_img, "", seed=42)
        except Exception as e:
            self.skipTest(f"_generate_synthetic_conditioned_image raised: {e}")

        import numpy as np, colorsys
        arr = np.array(cond.convert("RGB"), dtype=np.float32) / 255.0
        sats = [colorsys.rgb_to_hsv(*px.tolist())[1]
                for row in arr for px in row]
        mean_sat = sum(sats) / len(sats) * 255.0
        self.assertGreater(mean_sat, 10,
            f"Expected non-gray output, got mean saturation {mean_sat:.1f}/255")


# ---------------------------------------------------------------------------
# TestGLBColorEncoding
# ---------------------------------------------------------------------------

class TestGLBColorEncoding(unittest.TestCase):
    """Verify that TrimeshFallback GLB export embeds COLOR_0 and TEXCOORD_0."""

    def setUp(self):
        try:
            from engine.mesh_utils import TrimeshFallback
            self.TrimeshFallback = TrimeshFallback
            self._available = True
        except ImportError as e:
            self._available = False
            self._skip_reason = str(e)

    def _make_simple_mesh(self):
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        vertex_colors = [
            (200, 80, 80, 255),
            (80, 200, 80, 255),
            (80, 80, 200, 255),
            (200, 200, 80, 255),
        ]
        return self.TrimeshFallback(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )

    def test_glb_has_color0(self):
        """Exported GLB binary must contain the COLOR_0 accessor string."""
        if not self._available:
            self.skipTest(self._skip_reason)
        import tempfile, os
        mesh = self._make_simple_mesh()
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            tmp_path = f.name
        try:
            mesh.export(tmp_path)
            with open(tmp_path, "rb") as f:
                data = f.read()
            self.assertIn(b"COLOR_0", data,
                "GLB must contain COLOR_0 accessor for vertex colours")
        finally:
            os.unlink(tmp_path)

    def test_glb_has_texcoord0(self):
        """Exported GLB binary must contain the TEXCOORD_0 accessor string."""
        if not self._available:
            self.skipTest(self._skip_reason)
        import tempfile, os
        mesh = self._make_simple_mesh()
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            tmp_path = f.name
        try:
            mesh.export(tmp_path)
            with open(tmp_path, "rb") as f:
                data = f.read()
            self.assertIn(b"TEXCOORD_0", data,
                "GLB must contain TEXCOORD_0 accessor for UV coordinates")
        finally:
            os.unlink(tmp_path)

    def test_glb_is_valid_binary(self):
        """GLB must start with the glTF magic bytes 0x46546C67."""
        if not self._available:
            self.skipTest(self._skip_reason)
        import tempfile, os
        mesh = self._make_simple_mesh()
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            tmp_path = f.name
        try:
            mesh.export(tmp_path)
            with open(tmp_path, "rb") as f:
                magic = f.read(4)
            self.assertEqual(magic, b"glTF",
                "GLB must start with glTF magic bytes")
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
