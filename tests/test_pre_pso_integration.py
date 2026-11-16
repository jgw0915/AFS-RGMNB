import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from afs_rgmnb.experiments.pre_pso import PrePSOConfig, run_pre_pso, save_pre_pso
from afs_rgmnb.methods.random_subspace import select_subspaces


class PrePSOIntegrationTests(unittest.TestCase):
    def dataset(self):
        n = 150
        X = sparse.lil_matrix((n, n + 3), dtype=np.int64)
        y = np.repeat(np.arange(3), n // 3)
        for row in range(n):
            X[row, int(y[row])] = 10
            X[row, row + 3] = 1  # unique identity feature; heldout-only columns exist
        X = sparse.vstack([X.tocsr(), X[0], X[0]]).tocsr()
        y = np.concatenate([y, [0, 1]])  # same vector includes conflicting labels
        return X, y, [f'original-{i}' for i in range(len(y))]

    def config(self):
        return PrePSOConfig(n_groups=2, candidates_per_group=3, subspace_size=3,
                            alpha=1., epsilon=.05, seed=42, outer_folds=5,
                            search_fraction=.2, outer_fold_index=0)

    def test_end_to_end_roles_mapping_and_private_artifacts(self):
        X, y, ids = self.dataset()
        with patch('afs_rgmnb.experiments.pre_pso.select_subspaces', wraps=select_subspaces) as selector:
            result = run_pre_pso(X, y, config=self.config(), sample_ids=ids,
                                 source_feature_ids=np.arange(1, X.shape[1] + 1))
        plan, parts, selection, report = result
        split = plan.folds[0]
        self.assertFalse(report['test_evaluated'])
        self.assertEqual(plan.audit['conflicting_label_groups'], 1)
        self.assertEqual(sum(report['selected_fold']['sample_overlaps'].values()), 0)
        self.assertEqual(sum(report['selected_fold']['group_overlaps'].values()), 0)
        np.testing.assert_array_equal(parts.columns, np.unique(X[split.fit_indices].indices))
        self.assertLess(len(parts.columns), X.shape[1])
        np.testing.assert_array_equal(parts.source_feature_ids, parts.columns + 1)
        # The model-selection boundary receives exactly fit/search and their own labels.
        passed = selector.call_args.args
        self.assertEqual(len(passed), 4)
        np.testing.assert_array_equal(passed[0].toarray(), X[split.fit_indices][:, parts.columns].toarray())
        np.testing.assert_array_equal(passed[1], y[split.fit_indices])
        np.testing.assert_array_equal(passed[2].toarray(), X[split.search_indices][:, parts.columns].toarray())
        np.testing.assert_array_equal(passed[3], y[split.search_indices])
        self.assertEqual(len(selection.selected_subspaces), 2)
        with tempfile.TemporaryDirectory() as directory:
            save_pre_pso(Path(directory), result, sample_ids=ids, sample_labels=y)
            saved = json.loads((Path(directory) / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['groups'], report['groups'])
            self.assertEqual(len((Path(directory) / 'selected_fold_records.tsv').read_text().splitlines()), len(y) + 1)
            self.assertTrue((Path(directory) / 'split_indices.npz').is_file())

    def test_same_seed_reproduces_complete_synthetic_pipeline(self):
        X, y, ids = self.dataset()
        arguments = dict(config=self.config(), sample_ids=ids, source_feature_ids=np.arange(X.shape[1]))
        a = run_pre_pso(X, y, **arguments)
        b = run_pre_pso(X, y, **arguments)
        self.assertEqual(a[-1]['groups'], b[-1]['groups'])
        self.assertEqual(a[-1]['split_plan'], b[-1]['split_plan'])
        np.testing.assert_array_equal(a[2].sampling_weights, b[2].sampling_weights)

    def test_invalid_development_config_rejected_before_data_work(self):
        from dataclasses import asdict
        base = asdict(self.config())
        for changes in [dict(n_groups=0), dict(alpha=0), dict(epsilon=1),
                        dict(outer_fold_index=5), dict(seed=True)]:
            with self.assertRaises(ValueError):
                PrePSOConfig(**(base | changes))
