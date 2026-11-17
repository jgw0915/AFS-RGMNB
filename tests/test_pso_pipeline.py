"""Integration of the verified data/selection stages with fresh PSO per group."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from afs_rgmnb.experiments.pre_pso import PrePSOConfig
from afs_rgmnb.experiments.pso_search import (
    optimize_selected_subspaces, run_pso_pipeline, save_pso_pipeline,
)
from afs_rgmnb.methods.pso import PSOConfig, optimize_mnb


class PSOPipelineTests(unittest.TestCase):
    def dataset(self):
        y = np.repeat(np.arange(3), 30)
        X = np.zeros((len(y), 4), dtype=np.int64)
        X[np.arange(len(y)), y] = 5
        X[:, 3] = np.arange(1, len(y) + 1)
        X = sparse.vstack([sparse.csr_matrix(X), sparse.csr_matrix(X[:1])]).tocsr()
        y = np.concatenate([y, [1]])
        return X, y, [f'source-{i}' for i in range(len(y))]

    def configuration(self):
        pre = PrePSOConfig(n_groups=2, candidates_per_group=2, subspace_size=2,
                           alpha=1, epsilon=.05, seed=42, outer_folds=5,
                           search_fraction=.2, outer_fold_index=0)
        pso = PSOConfig(n_particles=3, max_iterations=3, c1=1.4, c2=1.4, vmax=.001)
        return pre, pso

    def test_only_search_enters_one_fresh_swarm_per_group_and_artifacts_roundtrip(self):
        X, y, ids = self.dataset()
        pre, pso = self.configuration()
        with patch('afs_rgmnb.methods.pso.optimize_mnb', wraps=optimize_mnb) as optimizer:
            result = run_pso_pipeline(X, y, pre_config=pre, pso_config=pso, pso_seed=42,
                                      sample_ids=ids, source_feature_ids=np.arange(1, 5))
        pre_result, stage, report = result
        plan, parts, selection, _ = pre_result
        split = plan.folds[0]
        self.assertEqual(optimizer.call_count, 2)
        self.assertEqual(len(stage.results), 2)
        self.assertEqual(len(set(stage.seeds)), 2)
        self.assertFalse(report['test_evaluated'])
        for index, call in enumerate(optimizer.call_args_list):
            self.assertEqual(len(call.args), 2)
            np.testing.assert_array_equal(call.args[0].toarray(), parts.X_search.toarray())
            np.testing.assert_array_equal(call.args[1], y[split.search_indices])
            np.testing.assert_array_equal(call.kwargs['feature_indices'], selection.selected_subspaces[index])
            self.assertEqual(call.kwargs['seed'], stage.seeds[index])
            found = stage.results[index]
            self.assertGreaterEqual(found.best_fitness, found.history[0].best_fitness)
            self.assertTrue(np.all(found.model.class_prior_ > 0))
            np.testing.assert_allclose(found.model.feature_prob_.sum(axis=1), 1)
            # Fresh random priors are not replaced by the fitted candidate priors.
            self.assertFalse(np.allclose(found.model.class_prior_, selection.groups[index].model.class_prior_))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_pso_pipeline(output, result, sample_ids=ids, sample_labels=y)
            saved = json.loads((output / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['pso'], json.loads(json.dumps(report['pso'])))
            for index, found in enumerate(stage.results):
                with np.load(output / f'group_{index:02d}_best_parameters.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['feature_prob'], found.model.feature_prob_)
                    np.testing.assert_array_equal(data['class_prior'], found.model.class_prior_)

    def test_complete_seeded_pipeline_repeats_models_and_histories(self):
        X, y, ids = self.dataset()
        pre, pso = self.configuration()
        kwargs = dict(pre_config=pre, pso_config=pso, pso_seed=9, sample_ids=ids,
                      source_feature_ids=np.arange(1, 5))
        a, b = run_pso_pipeline(X, y, **kwargs), run_pso_pipeline(X, y, **kwargs)
        self.assertEqual(a[1].seeds, b[1].seeds)
        for ra, rb in zip(a[1].results, b[1].results):
            self.assertEqual([asdict(h) for h in ra.history], [asdict(h) for h in rb.history])
            np.testing.assert_array_equal(ra.model.class_prior_, rb.model.class_prior_)
            np.testing.assert_array_equal(ra.model.feature_prob_, rb.model.feature_prob_)
            np.testing.assert_array_equal(ra.personal_best_positions, rb.personal_best_positions)

    def test_group_stage_rejects_missing_subspaces_bad_seed_and_variable_m(self):
        X, y, _ = self.dataset()
        _, pso = self.configuration()
        base = dict(classes=[0, 1, 2], selected_subspaces=[[0, 1]],
                    n_features_in=4, config=pso, seed=42)
        for change in [dict(selected_subspaces=[]), dict(seed=-1),
                       dict(selected_subspaces=[[0, 1], [0, 1, 2]])]:
            with self.assertRaises(ValueError):
                optimize_selected_subspaces(X, y, **(base | change))
