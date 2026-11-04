import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lr_schedule import round_lr  # noqa: E402


class RoundLrTest(unittest.TestCase):
    def test_constant_is_identity(self):
        for r in range(1, 10):
            self.assertEqual(round_lr(5e-4, r, 9, "constant"), 5e-4)

    def test_cosine_first_round_is_base_lr(self):
        self.assertAlmostEqual(round_lr(5e-4, 1, 9, "cosine"), 5e-4)

    def test_cosine_monotone_decreasing_and_positive(self):
        lrs = [round_lr(5e-4, r, 9, "cosine") for r in range(1, 10)]
        self.assertTrue(all(a > b for a, b in zip(lrs, lrs[1:])))
        self.assertGreater(lrs[-1], 0.0)
        self.assertLess(lrs[-1], 0.05 * 5e-4)

    def test_cosine_midpoint(self):
        # R=10, r=6 -> progress 0.5 -> 절반
        self.assertAlmostEqual(round_lr(1.0, 6, 10, "cosine"), 0.5)

    def test_min_ratio_floor(self):
        lrs = [round_lr(1.0, r, 9, "cosine", min_ratio=0.1) for r in range(1, 10)]
        self.assertTrue(all(l >= 0.1 for l in lrs))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            round_lr(1.0, 1, 9, "linear")
        with self.assertRaises(ValueError):
            round_lr(1.0, 10, 9, "cosine")


if __name__ == "__main__":
    unittest.main()
