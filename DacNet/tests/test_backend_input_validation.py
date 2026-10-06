import unittest

import numpy as np

from backend.main import has_plausible_bilateral_lung_mask


class BilateralLungMaskValidationTests(unittest.TestCase):
    def test_accepts_a_plausible_pair_of_lungs(self):
        rows, columns = np.ogrid[:100, :100]
        left_lung = ((rows - 50) / 38) ** 2 + ((columns - 31) / 17) ** 2 <= 1
        right_lung = ((rows - 50) / 38) ** 2 + ((columns - 69) / 17) ** 2 <= 1

        self.assertTrue(has_plausible_bilateral_lung_mask(left_lung | right_lung))

    def test_rejects_empty_mask(self):
        self.assertFalse(has_plausible_bilateral_lung_mask(np.zeros((100, 100), dtype=bool)))

    def test_rejects_unilateral_mask(self):
        rows, columns = np.ogrid[:100, :100]
        left_lung = ((rows - 50) / 38) ** 2 + ((columns - 31) / 17) ** 2 <= 1

        self.assertFalse(has_plausible_bilateral_lung_mask(left_lung))

    def test_rejects_mask_that_is_too_small(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 20:40] = True
        mask[45:55, 60:80] = True

        self.assertFalse(has_plausible_bilateral_lung_mask(mask))


if __name__ == "__main__":
    unittest.main()
