"""Hand-calculated validation of active Chapter 3 equations (3.7)-(3.8)."""
import unittest

import numpy as np
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from afs_rgmnb.methods.mnb import (
    ClassStatistics, MNBParameters, SubspaceMNB, class_log_scores,
    class_statistics, estimate_parameters,
)


class SubspaceMNBTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[2, 0, 100], [0, 2, 0], [1, 1, 3]])
        self.y = np.array(["a", "b", "a"])

    def test_class_statistics_count_occurrences_and_empirical_priors(self):
        stats = class_statistics(sparse.csr_matrix(self.X), self.y)
        np.testing.assert_array_equal(stats.classes, ["a", "b"])
        np.testing.assert_array_equal(stats.class_counts, [2, 1])
        np.testing.assert_array_equal(stats.feature_counts.toarray(), [[3, 1, 103], [0, 2, 0]])
        self.assertEqual(stats.feature_counts.dtype, np.float64)
        self.assertEqual(stats.n_samples, 3)
        self.assertEqual(stats.n_features, 3)

    def test_hand_calculated_smoothing_and_subspace_normalization(self):
        params = estimate_parameters(class_statistics(self.X, self.y), [0, 1], alpha=1)
        np.testing.assert_allclose(params.class_prior, [2 / 3, 1 / 3])
        np.testing.assert_allclose(params.feature_prob, [[4 / 6, 2 / 6], [1 / 4, 3 / 4]])
        np.testing.assert_allclose(params.feature_prob.sum(axis=1), [1, 1])
        self.assertEqual(params.feature_prob.dtype, np.float64)
        # The huge omitted column must not affect the denominator.
        changed = self.X.copy(); changed[:, 2] = 0
        equivalent = SubspaceMNB(alpha=1).fit(changed, self.y, [0, 1])
        np.testing.assert_array_equal(equivalent.feature_prob_, params.feature_prob)

    def test_prediction_scores_use_selected_columns_and_zero_prior_rule(self):
        model = SubspaceMNB(alpha=1).fit(self.X, self.y, [0, 1])
        query = np.array([[3, 0, 999], [0, 3, 999], [0, 0, 999]])
        expected = query[:, :2] @ np.log([[4 / 6, 2 / 6], [1 / 4, 3 / 4]]).T
        expected += np.log([2 / 3, 1 / 3])
        np.testing.assert_allclose(model.predict_log_scores(query), expected)
        np.testing.assert_array_equal(model.predict(query), ["a", "b", "a"])

    def test_feature_draw_order_is_preserved(self):
        model = SubspaceMNB(alpha=1).fit(self.X, self.y, [1, 0])
        np.testing.assert_array_equal(model.feature_indices_, [1, 0])
        np.testing.assert_allclose(model.feature_prob_, [[2 / 6, 4 / 6], [3 / 4, 1 / 4]])
        other = SubspaceMNB(alpha=1).fit(self.X, self.y, [0, 1])
        np.testing.assert_allclose(model.predict_log_scores(self.X), other.predict_log_scores(self.X))

    def test_sparse_dense_equivalence_and_reused_statistics(self):
        dense = SubspaceMNB(alpha=.5).fit(self.X, self.y, [2, 0])
        sparse_model = SubspaceMNB.from_statistics(
            class_statistics(sparse.csr_matrix(self.X), self.y), [2, 0], alpha=.5)
        np.testing.assert_array_equal(dense.feature_prob_, sparse_model.feature_prob_)
        np.testing.assert_allclose(dense.predict_log_scores(self.X),
                                   sparse_model.predict_log_scores(sparse.csr_matrix(self.X)))
        copied = SubspaceMNB.from_parameters(dense.parameters_)
        np.testing.assert_array_equal(copied.predict(self.X), dense.predict(self.X))

    def test_zero_class_feature_counts_are_uniform_after_smoothing(self):
        model = SubspaceMNB(alpha=2).fit([[0, 0], [1, 3]], [0, 1], [0, 1])
        np.testing.assert_allclose(model.feature_prob_, [[.5, .5], [3 / 8, 5 / 8]])
        np.testing.assert_allclose(model.class_prior_, [.5, .5])

    def test_ties_follow_sorted_or_explicit_class_order(self):
        X = [[0, 0], [0, 0]]
        default = SubspaceMNB(alpha=1).fit(X, ["z", "a"], [0, 1])
        explicit = SubspaceMNB(alpha=1, class_order=["z", "a"]).fit(X, ["z", "a"], [0, 1])
        np.testing.assert_array_equal(default.predict([[4, 4]]), ["a"])
        np.testing.assert_array_equal(explicit.predict([[4, 4]]), ["z"])

    def test_equivalence_to_sklearn_when_columns_smoothing_and_priors_match(self):
        indices = [2, 0]
        ours = SubspaceMNB(alpha=.7).fit(sparse.csr_matrix(self.X), self.y, indices)
        reference = MultinomialNB(alpha=.7, fit_prior=True).fit(self.X[:, indices], self.y)
        np.testing.assert_array_equal(ours.classes_, reference.classes_)
        np.testing.assert_allclose(ours.class_prior_, np.exp(reference.class_log_prior_))
        np.testing.assert_allclose(ours.feature_prob_, np.exp(reference.feature_log_prob_))
        np.testing.assert_array_equal(ours.predict(self.X), reference.predict(self.X[:, indices]))

    def test_invalid_count_inputs_and_labels_are_rejected(self):
        for X, y in [([[-1, 0], [0, 1]], [0, 1]),
                     ([[.5, 0], [0, 1]], [0, 1]),
                     ([[np.nan, 0], [0, 1]], [0, 1]),
                     ([[np.inf, 0], [0, 1]], [0, 1]),
                     ([[1j, 0], [0, 1]], [0, 1]),
                     ([[0, 0], [0, 1]], [0, None]),
                     ([[0, 0], [0, 1]], [0, np.nan]),
                     ([[0, 0], [0, 1]], ["a", ""]),
                     ([[0, 0], [0, 1]], [1, "a"]),
                     ([[0, 0], [0, 1]], [0, 0]),
                     ([[0, 0], [0, 1]], [[0], [1]]),
                     ([[0, 0], [0, 1]], [0])]:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                class_statistics(X, y)
        for order in [["a"], ["a", "a"], ["a", "b", "c"]]:
            with self.assertRaises(ValueError):
                class_statistics(self.X, self.y, class_order=order)

    def test_invalid_alpha_and_subspaces_are_rejected(self):
        stats = class_statistics(self.X, self.y)
        for alpha in [0, -1, np.inf, np.nan, True, [1]]:
            with self.assertRaises(ValueError):
                estimate_parameters(stats, [0, 1], alpha=alpha)
        for indices in [[0], [], [0, 0], [0, 3], [-1, 1], [0., 1.], [True, False], [[0, 1]]]:
            with self.assertRaises(ValueError):
                estimate_parameters(stats, indices, alpha=1)

    def test_invalid_explicit_parameters_and_prediction_inputs(self):
        model = SubspaceMNB(alpha=1).fit(self.X, self.y, [0, 1])
        params = model.parameters_
        bad = MNBParameters(params.classes, params.feature_indices,
                            np.array([.5, .6]), params.feature_prob, 3)
        with self.assertRaises(ValueError):
            class_log_scores(self.X, bad)
        with self.assertRaises(ValueError):
            model.predict([[0, 0]])
        with self.assertRaises(ValueError):
            model.predict([[0, -.5, 0]])
        with self.assertRaises(ValueError):
            SubspaceMNB(alpha=1).predict(self.X)
        with self.assertRaises(ValueError):
            SubspaceMNB.from_statistics(None, [0, 1], alpha=1)
        with self.assertRaises(ValueError):
            SubspaceMNB.from_parameters(None)
        self.assertEqual(model.predict(np.empty((0, 3))).shape, (0,))


if __name__ == "__main__":
    unittest.main()
