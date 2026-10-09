import base64
import io
import unittest

import numpy as np
from PIL import Image

from backend.main import build_overlay


class FeatheredLungHeatmapTests(unittest.TestCase):
    def test_overlay_uses_soft_lung_roi_and_preserves_heatmap_colors(self):
        original = Image.new("RGB", (100, 100), (0, 0, 0))
        cam = np.ones((100, 100), dtype=np.float32)
        mask = Image.new("L", original.size, 0)
        mask.paste(255, (20, 20, 80, 80))

        result = build_overlay(original, cam, mask, (100, 100), (0, 0))
        encoded_image = base64.b64decode(result.split(",", 1)[1])
        with Image.open(io.BytesIO(encoded_image)) as overlay:
            self.assertEqual(overlay.size, original.size)
            outside = overlay.getpixel((5, 50))
            edge = overlay.getpixel((19, 50))
            inside = overlay.getpixel((50, 50))

            self.assertEqual(outside, (0, 0, 0))
            self.assertGreater(edge[0], 0)
            self.assertLess(edge[0], inside[0])
            self.assertEqual(edge[1:], (0, 0))
            self.assertGreater(inside[0], 0)
            self.assertEqual(inside[1:], (0, 0))


if __name__ == "__main__":
    unittest.main()
