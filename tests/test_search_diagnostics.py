"""Controls and telemetry must leave the fitness, fitted parameters and split unchanged."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

from afs_rgmnb.experiments import diagnostics as d
from afs_rgmnb.experiments.baselines import bootstrap_indices, document_frequencies
from afs_rgmnb.experiments.pso_search import optimize_selected_subspaces
from afs_rgmnb.methods.adaptive_search import AdaptiveConfig
from afs_rgmnb.methods.pso import PSOConfig
from scripts.run_search_diagnostics import load_training_roles


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fit = sparse.csr_matrix([[11, 0, 1], [4, 1, 1], [0, 8, 0], [1, 5, 0]])
        self.search = sparse.csr_matrix([[1, 7, 2], [9, 0, 0], [5, 1, 0]])
        self.data = d.SearchData(
            self.fit, np.array([4, 4, 9, 9]), self.search, np.array([9, 4, 4])
        )
        self.adaptive = AdaptiveConfig(2, 2, 2, 1.0, 0.05, 1.0, 0.1, 1, 2)
        self.pso = PSOConfig(4, 3, 1.4, 1.4, 0.01)

    def test_reference_counts_roles_and_class_order(self):
        controls = d.reference_controls(self.data, 1.0)
        self.assertEqual(controls["classes"], [4, 9])
        self.assertEqual(controls["majority"]["correct"], 2)
        self.assertEqual(controls["full_mnb"]["confusion"], [[2, 0], [0, 1]])
        self.assertEqual(controls["full_mnb"]["accuracy"], 1)
        self.assertEqual(controls["full_mnb"]["sklearn_disagreements"], 0)

    def test_diagnostics_do_not_change_rng_or_model(self):
        kwargs = dict(
            classes=[4, 9],
            selected_subspaces=[[2, 0], [0, 1]],
            n_features_in=3,
            config=self.pso,
            seed=3,
        )
        a = optimize_selected_subspaces(self.search, self.data.y_search, **kwargs)
        b = optimize_selected_subspaces(
            self.search, self.data.y_search, diagnostics=True, **kwargs
        )
        for previous, observed in zip(a.results, b.results):
            self.assertIsNone(previous.diagnostics)
            self.assertEqual(previous.history, observed.history)
            np.testing.assert_array_equal(previous.positions, observed.positions)
            np.testing.assert_array_equal(
                previous.model.feature_prob_, observed.model.feature_prob_
            )
            _, record = d.swarm_metrics(observed, self.data)
            self.assertEqual(len(record["movement_by_step"]), observed.iterations)
            self.assertGreater(record["mean_abs_change"], 0)
            self.assertEqual(record["final"]["correct"] / 3, observed.best_fitness)

    def test_fixed_subspaces_across_pso_conditions(self):
        selected, seconds = d.select_features(self.data, self.adaptive, 7)

        def run(config):
            return d.compare_models(
                self.data, selected, self.adaptive, config, 31, seconds
            )

        low, left = run(self.pso)
        high, right = run(replace(self.pso, vmax=0.2))
        repeat, _ = run(self.pso)
        self.assertEqual(low["members"], repeat["members"])
        self.assertEqual(low["fitted_vote"], high["fitted_vote"])
        for i, cols in enumerate(selected.selected_subspaces):
            np.testing.assert_array_equal(left.members[i].feature_indices_, cols)
            np.testing.assert_array_equal(right.members[i].feature_indices_, cols)
            self.assertEqual(
                low["members"][i]["initial_positions_sha256"],
                high["members"][i]["initial_positions_sha256"],
            )
        self.assertAlmostEqual(
            low["seconds"]["build"],
            sum(low["seconds"][k] for k in ("feature", "pso", "construct")),
            places=6,
        )

    def test_round_audits_sparse_coverage_and_all_draws(self):
        selected, _ = d.select_features(self.data, self.adaptive, 3)
        rounds = d.feature_diagnostics(self.data, selected, self.adaptive.c0)
        for h in rounds:
            self.assertEqual(h["evaluations"], 4)
            for g, group in enumerate(h["draws"]):
                for j, cols in enumerate(group):
                    nnz = np.count_nonzero(self.search[:, cols].toarray(), axis=1)
                    cov = h["search_coverage"][g][j]
                    self.assertEqual(cov["zeros"], sum(nnz == 0))
                    self.assertEqual(cov["nnz_histogram"], np.bincount(nnz).tolist())

    def test_random_budget_and_prefixes_are_separate_controls(self):
        selected, seconds = d.select_features(self.data, self.adaptive, 7)
        pso = replace(self.pso, n_particles=1)
        # Initialization is the whole budget for a single converged particle.
        record, _ = d.compare_models(
            self.data,
            selected,
            self.adaptive,
            pso,
            9,
            seconds,
            random_control=True,
            prefix_sizes=[1, 2],
        )
        self.assertEqual(record["budget_matched_random_vote"], record["initial_vote"])
        self.assertEqual(
            record["fixed_member_prefixes"]["2"]["correct"],
            record["final_vote"]["correct"],
        )
        self.assertEqual(record["fixed_member_prefixes"]["1"]["ties"], 0)

    def test_loader_accesses_fit_search_only_and_rejects_index_drift(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "roles.npz"
            fit, search, test = np.arange(4), np.arange(4, 7), np.array([7])
            # No eighth row/label is supplied. Test-only content cannot enter the diagnostic API.
            X = sparse.hstack(
                [sparse.vstack([self.fit, self.search]), sparse.csr_matrix((7, 12000))],
                format="csr",
            )
            y = np.concatenate((self.data.y_fit, self.data.y_search))
            np.savez(
                f,
                fit=fit,
                search=search,
                test=test,
                fit_columns=np.arange(3),
                source_feature_ids=np.arange(1, 4),
            )
            hashes = {
                k: d.digest(v, "<i8")
                for k, v in dict(fit=fit, search=search, test=test).items()
            }
            data, _ = load_training_roles(SimpleNamespace(X=X, y=y), f, hashes)
            self.assertTrue(sparse.isspmatrix_csr(data.X_search))
            np.testing.assert_array_equal(data.y_search, self.data.y_search)
            with self.assertRaises(ValueError):
                load_training_roles(
                    SimpleNamespace(X=X, y=y), f, hashes | {"fit": "wrong"}
                )

    def test_presence_and_bootstrap_primitives(self):
        freq = document_frequencies(self.fit, self.data.y_fit)
        np.testing.assert_array_equal(
            freq.feature_counts.toarray(), [[2, 1, 2], [1, 2, 0]]
        )
        self.assertEqual(self.fit[0, 0], 11)
        rows = list(
            bootstrap_indices(
                3, sample_size=10, n_members=2, rng=np.random.default_rng(7)
            )
        )
        other = list(
            bootstrap_indices(
                3, sample_size=10, n_members=2, rng=np.random.default_rng(7)
            )
        )
        np.testing.assert_array_equal(rows, other)
        self.assertEqual(np.shape(rows), (2, 10))
        self.assertTrue(np.all((np.array(rows) >= 0) & (np.array(rows) < 3)))
        with self.assertRaises(ValueError):
            list(
                bootstrap_indices(
                    3, sample_size=0, n_members=2, rng=np.random.default_rng(7)
                )
            )
