import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from afs_rgmnb.methods.feature_scoring import jmh_scores
from afs_rgmnb.methods.random_subspace import select_subspaces


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.X = sparse.csr_matrix([[10, 0, 1, 0], [8, 1, 0, 1],
                                    [0, 10, 1, 0], [1, 8, 0, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.kwargs = dict(n_groups=2, candidates_per_group=3, subspace_size=2,
                           alpha=1, epsilon=.05, seed=42)

    def test_hand_computed_selection_and_first_tie(self):
        draws = [np.array(x) for x in [[2, 3], [0, 1], [0, 1]] * 2]
        with patch('afs_rgmnb.methods.random_subspace.sample_subspace', side_effect=draws), \
             patch('afs_rgmnb.methods.feature_scoring.jmh_scores', wraps=jmh_scores) as scorer:
            result = select_subspaces(self.X, self.y, self.X, self.y, **self.kwargs)
        self.assertEqual(scorer.call_count, 1)
        self.assertEqual(len(result.groups), 2)
        for group in result.groups:
            self.assertEqual([x.accuracy for x in group.candidates], [.5, 1, 1])
            self.assertEqual(group.selected_index, 1)
            self.assertEqual(group.tied_best_indices, (1, 2))
            np.testing.assert_array_equal(group.selected_subspace, [0, 1])
            self.assertEqual(len(group.candidates), 3)

    def test_actual_draws_reproduce_and_have_fixed_unique_size(self):
        a = select_subspaces(self.X, self.y, self.X, self.y, **self.kwargs)
        b = select_subspaces(self.X, self.y, self.X, self.y, **self.kwargs)
        np.testing.assert_array_equal(a.feature_scores, b.feature_scores)
        np.testing.assert_array_equal(a.sampling_weights, b.sampling_weights)
        for ga, gb in zip(a.groups, b.groups):
            self.assertEqual(ga.selected_index, gb.selected_index)
            for ca, cb in zip(ga.candidates, gb.candidates):
                np.testing.assert_array_equal(ca.feature_indices, cb.feature_indices)
                self.assertEqual(len(set(ca.feature_indices)), 2)
                self.assertEqual(ca.accuracy, cb.accuracy)

    def test_search_changes_evaluation_but_not_parameter_estimation(self):
        fixed = lambda *args, **kwargs: np.array([0, 1])
        with patch('afs_rgmnb.methods.random_subspace.sample_subspace', side_effect=fixed):
            a = select_subspaces(self.X, self.y, self.X, self.y, **self.kwargs)
            b = select_subspaces(self.X, self.y, self.X, 1 - self.y, **self.kwargs)
        self.assertEqual(a.groups[0].candidates[0].accuracy, 1)
        self.assertEqual(b.groups[0].candidates[0].accuracy, 0)
        np.testing.assert_array_equal(a.groups[0].model.feature_prob_, b.groups[0].model.feature_prob_)
        np.testing.assert_array_equal(a.groups[0].model.class_prior_, b.groups[0].model.class_prior_)

    def test_invalid_configuration_and_search_classes(self):
        for changes in [dict(n_groups=0), dict(candidates_per_group=0), dict(subspace_size=1),
                        dict(seed=-1), dict(alpha=0), dict(epsilon=1)]:
            with self.assertRaises(ValueError):
                select_subspaces(self.X, self.y, self.X, self.y, **(self.kwargs | changes))
        with self.assertRaises(ValueError):
            select_subspaces(self.X, self.y, self.X[:1], self.y[:1], **self.kwargs)
        with self.assertRaises(ValueError):
            select_subspaces(self.X, self.y, self.X[:, :3], self.y, **self.kwargs)
