"""Independent small-data correctness checks. No test-framework dependency."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from exp_1 import (knn_predict, load_data, loo_eval, loo_predictions,
                   predict_from_distances, squared_distances, write_arff)


def reference_loo(X, y, k):
    """Literal leave-one-out with direct Euclidean norms, no matrix identity."""
    predictions = []
    for held_out in range(len(X)):
        candidates = [i for i in range(len(X)) if i != held_out]
        candidates.sort(key=lambda i: (np.linalg.norm(X[held_out] - X[i]), i))
        counts = np.bincount(y[candidates[:k]], minlength=10)
        predictions.append(int(np.argmax(counts)))
    return np.asarray(predictions)


class KNNTests(unittest.TestCase):
    def test_matches_literal_loo_for_several_k(self):
        rng = np.random.default_rng(7)
        X = rng.integers(0, 2, size=(19, 12)).astype(float)
        y = rng.integers(0, 10, size=19)
        for k in (1, 3, 5, 18):
            with self.subTest(k=k):
                expected = reference_loo(X, y, k)
                np.testing.assert_array_equal(loo_predictions(X, y, k), expected)
                self.assertEqual(loo_eval(X, y, k), np.mean(expected == y))

    def test_excludes_self_but_keeps_other_identical_rows(self):
        X = np.array([[0.], [0.], [10.]])
        y = np.array([1, 2, 3])
        np.testing.assert_array_equal(loo_predictions(X, y, 1), [2, 1, 1])

    def test_distance_ties_use_row_order(self):
        np.testing.assert_array_equal(knn_predict([[1], [-1]], [8, 2], [[0]], 1), [8])

    def test_vote_ties_and_seeded_randomness(self):
        d = np.tile([0., 1.], (100, 1))
        y = np.array([8, 2])
        np.testing.assert_array_equal(predict_from_distances(d, y, 2), np.full(100, 2))
        a = predict_from_distances(d, y, 2, tie_break="random", seed=5)
        b = predict_from_distances(d, y, 2, tie_break="random", seed=5)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(set(a), {2, 8})

    def test_kth_distance_boundary_can_change_vote(self):
        distances = [[0., 1., 1., 1.]]
        labels = [9, 1, 1, 1]
        self.assertEqual(predict_from_distances(distances, labels, 2)[0], 1)
        self.assertEqual(predict_from_distances(distances, labels, 1)[0], 9)
        tied = [[1., 1., 1.]]
        self.assertEqual(predict_from_distances(tied, [9, 1, 1], 1)[0], 9)
        self.assertEqual(predict_from_distances(tied, [9, 1, 1], 1, include_boundary=True)[0], 1)

    def test_unsigned_binary_pixels_do_not_overflow(self):
        X = np.array([np.ones(256), np.zeros(256)], dtype=np.uint8)
        np.testing.assert_array_equal(squared_distances(X, X), [[0, 256], [256, 0]])

    def test_invalid_k_and_insufficient_neighbours(self):
        for k in (0, -1, 3, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                loo_predictions([[0], [1], [2]], [0, 1, 2], k)
        with self.assertRaises(ValueError):
            predict_from_distances([[0, np.inf]], [0, 1], 2)

    def test_one_hot_validation_and_nominal_arff(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.txt"
            raw = np.zeros((2, 266))
            raw[0, 256 + 7] = 1
            raw[1, 256 + 2] = 1
            np.savetxt(path, raw)
            X, y = load_data(path)
            np.testing.assert_array_equal(y, [7, 2])
            arff = Path(temp) / "data.arff"
            write_arff(arff, X, y)
            self.assertEqual(arff.read_text().count("@attribute"), 257)
            self.assertIn("@attribute digit {0,1,2,3,4,5,6,7,8,9}", arff.read_text())
            raw[0, 256] = 1
            np.savetxt(path, raw)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
