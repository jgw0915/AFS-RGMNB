"""Group-disjoint splitting and fit-only feature mapping acceptance tests."""
import unittest

import numpy as np
from scipy import sparse

from afs_rgmnb.data.splitting import (GroupSplit, group_count_vectors,
    group_aware_splits, prepare_feature_partitions)


class GroupSplittingTests(unittest.TestCase):
    def dataset(self):
        # One cross-label group, one same-label group, and many distinct groups.
        X, y = [], []
        for cls, size in [(0, 50), (1, 65), (2, 80)]:
            for index in range(size):
                X.append([cls + 1, index + 1, (index + 1) * (cls + 1)])
                y.append(cls)
        X.extend([X[0], X[0], X[60]])
        y.extend([0, 1, 1])
        return sparse.csr_matrix(X), np.asarray(y)

    def test_identity_ignores_storage_dtype_order_and_explicit_zeros(self):
        X = sparse.csr_matrix((np.array([2, 0, 1, 2, 1, 3]),
                               np.array([1, 2, 0, 1, 0, 0]),
                               np.array([0, 3, 5, 6])), shape=(3, 3))
        groups = group_count_vectors(X)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])
        np.testing.assert_array_equal(groups, group_count_vectors(X.toarray().astype(float)))
        # Two entries in one feature canonicalize to their sum.
        duplicated = sparse.csr_matrix(([1, 1, 2], [0, 0, 1], [0, 3]), shape=(1, 3))
        self.assertEqual(group_count_vectors(duplicated)[0], group_count_vectors([[2, 2, 0]])[0])
        np.testing.assert_array_equal(groups, group_count_vectors(X.tocsc()))
        self.assertEqual(group_count_vectors(duplicated.tocoo())[0],
                         group_count_vectors([[2, 2, 0]])[0])

    def test_groups_stable_when_sample_order_changes(self):
        X, _ = self.dataset()
        order = np.random.default_rng(4).permutation(X.shape[0])
        np.testing.assert_array_equal(group_count_vectors(X)[order], group_count_vectors(X[order]))

    def test_nested_splits_preserve_all_records_labels_and_groups(self):
        X, y = self.dataset()
        ids = [f"original-{i}" for i in range(X.shape[0])]
        plan = group_aware_splits(X, y, n_splits=5, search_fraction=.2, seed=42, sample_ids=ids,
                                  class_order=[2, 0, 1])
        self.assertEqual(plan.classes.tolist(), [2, 0, 1])
        self.assertEqual(plan.sample_ids.tolist(), ids)
        self.assertEqual(plan.audit["duplicate_groups"], 2)
        self.assertEqual(plan.audit["conflicting_label_groups"], 1)
        self.assertEqual(plan.audit["conflicting_label_rows"], 3)
        all_test = []
        for split in plan.folds:
            combined = np.concatenate([split.fit_indices, split.search_indices, split.test_indices])
            np.testing.assert_array_equal(np.sort(combined), np.arange(X.shape[0]))
            self.assertEqual(max(split.audit["sample_overlaps"].values()), 0)
            self.assertEqual(max(split.audit["group_overlaps"].values()), 0)
            for rows in [split.fit_indices, split.search_indices, split.test_indices]:
                self.assertEqual(set(y[rows]), {0, 1, 2})
            all_test.extend(split.test_indices.tolist())
        np.testing.assert_array_equal(np.sort(all_test), np.arange(X.shape[0]))
        self.assertTrue(plan.audit["outer_test_coverage_once"])

    def test_same_seed_reproduces_rows_and_audit(self):
        X, y = self.dataset()
        first = group_aware_splits(X, y, seed=16)
        second = group_aware_splits(X, y, seed=16)
        self.assertEqual(first.audit, second.audit)
        for a, b in zip(first.folds, second.folds):
            self.assertEqual(a.audit, b.audit)
            np.testing.assert_array_equal(a.fit_indices, b.fit_indices)
            np.testing.assert_array_equal(a.search_indices, b.search_indices)
            np.testing.assert_array_equal(a.test_indices, b.test_indices)

    def test_search_majority_fraction_still_keeps_groups_disjoint(self):
        X, y = self.dataset()
        plan = group_aware_splits(X, y, search_fraction=.8)
        for split in plan.folds:
            self.assertGreater(len(split.search_indices), len(split.fit_indices))
            self.assertEqual(max(split.audit["group_overlaps"].values()), 0)

    def test_fit_only_vocabulary_preserves_original_ids(self):
        X = sparse.csr_matrix([[2, 0, 1, 0], [0, 0, 2, 0], [1, 9, 0, 0], [0, 0, 1, 8]])
        split = GroupSplit(np.array([0, 1]), np.array([2]), np.array([3]))
        features = prepare_feature_partitions(X, split, source_feature_ids=[10, 20, 30, 40])
        self.assertEqual(features.columns.tolist(), [0, 2])
        self.assertEqual(features.source_feature_ids.tolist(), [10, 30])
        np.testing.assert_array_equal(features.X_search.toarray(), [[1, 0]])
        np.testing.assert_array_equal(features.X_test.toarray(), [[0, 1]])
        self.assertEqual(X.shape[1], 4)
        with self.assertRaisesRegex(ValueError, "No observed features"):
            prepare_feature_partitions(sparse.csr_matrix([[0, 0], [1, 0], [0, 1]]),
                GroupSplit(np.array([0]), np.array([1]), np.array([2])))

    def test_infeasible_classes_fail_without_deletion(self):
        X = sparse.csr_matrix([[1, 0]] * 10 + [[0, 1]] * 10)
        with self.assertRaisesRegex(ValueError, "Too few identity groups"):
            group_aware_splits(X, [0] * 10 + [1] * 10)
        X = sparse.csr_matrix([[i, 1] for i in range(20)])
        with self.assertRaisesRegex(ValueError, "no classes were removed"):
            group_aware_splits(X, [0] * 4 + [1] * 16)

    def test_invalid_counts_labels_ids_and_settings_fail(self):
        X, y = self.dataset()
        for invalid in [[[-1, 2]], [[.5, 1]], [[np.inf, 1]], [[1j, 2]]]:
            with self.assertRaises(ValueError):
                group_count_vectors(invalid)
        for values in [[2, -1], [np.iinfo(np.int64).max, 1]]:
            invalid_coo = sparse.coo_matrix((values, ([0, 0], [0, 0])), shape=(1, 2))
            with self.assertRaises(ValueError):
                group_count_vectors(invalid_coo)
        for kwargs in [dict(n_splits=1), dict(search_fraction=0), dict(seed=-1),
                       dict(sample_ids=["same"] * len(y)), dict(class_order=[0, 1])]:
            with self.assertRaises(ValueError):
                group_aware_splits(X, y, **kwargs)
        with self.assertRaises(ValueError):
            group_aware_splits(X, [None] * len(y))
        for labels in [[0] + ["a"] * (len(y) - 1),
                       np.asarray([np.nan] + y[1:].tolist(), dtype=object),
                       [" "] + ["valid"] * (len(y) - 1)]:
            with self.assertRaises(ValueError):
                group_aware_splits(X, labels)
        for order in [[0, "1", 2], np.asarray([0, 1, np.nan], dtype=object)]:
            with self.assertRaises(ValueError):
                group_aware_splits(X, y, class_order=order)

    def test_feature_mapping_rejects_overlaps_and_duplicate_feature_ids(self):
        X = sparse.csr_matrix([[1, 0], [0, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "no index overlap"):
            prepare_feature_partitions(X, GroupSplit(np.array([0]), np.array([0]), np.array([2])))
        with self.assertRaisesRegex(ValueError, "source_feature_ids"):
            prepare_feature_partitions(X, GroupSplit(np.array([0]), np.array([1]), np.array([2])),
                                       source_feature_ids=[1, 1])


if __name__ == "__main__":
    unittest.main()
