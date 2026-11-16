"""Numerical tests tied to active Chapter 3, not generic named algorithms."""
import unittest

import numpy as np
from scipy import sparse

from afs_rgmnb.methods.feature_scoring import jmh_scores
from afs_rgmnb.methods.adaptive_sampling import (
    initial_probabilities, inclusion_proportions, update_probabilities, weight_change,
)
from afs_rgmnb.methods.random_subspace import sample_subspace


class Chapter3MathTests(unittest.TestCase):
    def test_two_class_jeffreys_uses_occurrences(self):
        X = sparse.csr_matrix([[9, 1], [1, 9]])
        scores = jmh_scores(X, ["a", "b"], alpha=1)
        expected = (10 / 12 - 2 / 12) * np.log((10 / 12) / (2 / 12)) * 2
        np.testing.assert_allclose(scores, [expected, expected])
        # Both features are present in both samples: binarization would give zero.
        self.assertTrue(np.all(scores > 0))

    def test_multiclass_empirical_complement_and_unweighted_outer_sum(self):
        X = np.array([[8, 1, 1], [1, 8, 1], [1, 7, 2], [1, 1, 8]])
        y = np.array([0, 1, 1, 2]); expected = np.zeros(3)
        p = np.array([[9, 2, 2], [3, 16, 4], [2, 2, 9]], dtype=float)
        p /= p.sum(axis=1)[:, None]
        priors = np.array([.25, .5, .25])
        for cls in range(3):
            other = [i for i in range(3) if i != cls]
            q = sum(priors[i] * p[i] for i in other) / sum(priors[i] for i in other)
            expected += p[cls] * np.log(p[cls] / q) + (1 - p[cls]) * np.log((1 - p[cls]) / (1 - q))
        np.testing.assert_allclose(jmh_scores(sparse.csr_matrix(X), y, alpha=1), expected)
        np.testing.assert_allclose(jmh_scores(X, y, alpha=1), expected)

    def test_equal_feature_probabilities_and_zero_score_gap(self):
        for classes in [2, 3, 7]:
            scores = jmh_scores(np.tile([2, 2, 2], (classes, 1)), np.arange(classes), alpha=1)
            np.testing.assert_array_equal(scores, [0, 0, 0])
            with self.assertRaisesRegex(ValueError, "undefined"):
                initial_probabilities(scores, epsilon=.1)

    def test_scoring_invalid_domains(self):
        for X, y, alpha in [([[-1, 2], [1, 2]], [0, 1], 1),
                            ([[.5, 2], [1, 2]], [0, 1], 1),
                            ([[np.inf, 2], [1, 2]], [0, 1], 1),
                            ([[1, 2], [1, 2]], [0, 0], 1),
                            ([[1, 2], [1, 2]], [0], 1),
                            ([[1, 2], [1, 2]], [0, 1], 0)]:
            with self.assertRaises(ValueError):
                jmh_scores(X, y, alpha=alpha)

    def test_initial_weights_and_exploration_floor(self):
        q = initial_probabilities([0, 1, 3], epsilon=.3)
        np.testing.assert_allclose(q, [.1, .275, .625])
        self.assertAlmostEqual(q.sum(), 1)
        self.assertTrue(np.all(q >= .3 / 3))

    def test_exact_inclusion_threshold_and_new_distribution(self):
        selected = [[0, 1], [0, 2]]
        np.testing.assert_allclose(inclusion_proportions(selected, n_features=4), [1, .5, .5, 0])
        # threshold=.5 exactly: the two r=.5 features must take the lower branch.
        c0 = .5 * np.log(4)
        raw = np.array([1, c0 / 4, c0 / 4, c0 / 4])
        q = update_probabilities(selected, n_features=4, c0=c0)
        np.testing.assert_allclose(q, raw / raw.sum())
        self.assertTrue(np.all(q > 0))
        self.assertAlmostEqual(weight_change(q, q), 0)
        self.assertTrue(0 <= weight_change(q, np.full(4, .25)) <= 2)

    def test_subspace_validation_and_fixed_m(self):
        for selected in [[], [[0, 0]], [[0, 5]], [[0, 1], [0, 1, 2]], [[0., 1.]]]:
            with self.assertRaises(ValueError):
                update_probabilities(selected, n_features=4, c0=1)

    def test_sampling_conditional_probabilities_are_renormalized(self):
        class RecordedDraws:
            def __init__(self):
                self.probabilities = []
            def choice(self, n, p):
                self.probabilities.append(p.copy())
                return 1 if n == 3 else 0
        rng = RecordedDraws()
        q = np.array([.1, .3, .6])
        chosen = sample_subspace(q, 2, rng=rng)
        np.testing.assert_array_equal(chosen, [1, 0])
        np.testing.assert_allclose(rng.probabilities[0], q)
        np.testing.assert_allclose(rng.probabilities[1], [1 / 7, 6 / 7])
        np.testing.assert_array_equal(q, [.1, .3, .6])

    def test_seed_reproducibility_and_full_selection(self):
        q = np.array([.1, .2, .3, .4])
        a = sample_subspace(q, 3, rng=np.random.default_rng(42))
        b = sample_subspace(q, 3, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 3)
        self.assertEqual(set(sample_subspace(q, 4, rng=np.random.default_rng(42))), set(range(4)))


if __name__ == "__main__":
    unittest.main()
