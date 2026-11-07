import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personalize_utils import (  # noqa: E402
    balanced_accuracy,
    class_counts,
    local_class_alpha,
    mask_probs,
    present_mask,
    select_config,
)


class PersonalizeUtilsTest(unittest.TestCase):
    def test_counts_and_present(self):
        c = class_counts([0, 0, 1, 4, 4, 4])
        self.assertEqual(c.tolist(), [2, 1, 0, 0, 3, 0, 0, 0])
        self.assertEqual(present_mask(c).tolist(), [True, True, False, False, True, False, False, False])

    def test_local_alpha_normalized(self):
        c = np.array([6, 3, 0, 0, 1, 0, 0, 0])
        a = local_class_alpha(c)
        self.assertTrue(np.all(a[c == 0] == 0))
        freq = c / c.sum()
        self.assertAlmostEqual(float((freq * a).sum()), 1.0, places=6)
        # 희귀 클래스일수록 가중치가 크다
        self.assertGreater(a[4], a[1])
        self.assertGreater(a[1], a[0])

    def test_local_alpha_clipped(self):
        c = np.array([10000, 1, 0, 0, 0, 0, 0, 0])
        self.assertLessEqual(local_class_alpha(c, max_alpha=20.0).max(), 20.0)

    def test_mask_probs(self):
        probs = np.array([[0.5, 0.1, 0.4], [0.1, 0.2, 0.7]])
        present = np.array([True, True, False])
        out = mask_probs(probs, present)
        np.testing.assert_allclose(out.sum(axis=1), 1.0)
        self.assertEqual(out.argmax(axis=1).tolist(), [0, 1])
        self.assertTrue(np.all(out[:, 2] == 0))

    def test_mask_probs_all_zero_row(self):
        probs = np.array([[0.0, 0.0, 1.0]])
        out = mask_probs(probs, np.array([True, True, False]))
        np.testing.assert_allclose(out, [[0.5, 0.5, 0.0]])

    def test_balanced_accuracy(self):
        y_true = np.array([0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 0])
        # recall: class0 0.5, class1 0.75
        self.assertAlmostEqual(balanced_accuracy(y_true, y_pred), 0.625)

    def test_select_config_prefers_higher_mean_then_smaller(self):
        scores = {
            ("head", "global", 50): [0.5, 0.5],
            ("all", "local", 100): [0.7, 0.7],
            ("head", "local", 100): [0.7, 0.7],
            ("head", "local", 200): [0.7, 0.7],
        }
        key, mean = select_config(scores)
        self.assertEqual(key, ("head", "local", 100))
        self.assertAlmostEqual(mean, 0.7)


if __name__ == "__main__":
    unittest.main()
