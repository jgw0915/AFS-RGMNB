"""Numerical and stage-boundary tests for the November 25 Chapter 3."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from afs_rgmnb.data.splitting import GroupSplit
from afs_rgmnb.experiments import runner
from afs_rgmnb.methods import adaptive_search as search
from afs_rgmnb.methods import pso
from afs_rgmnb.methods.adaptive_sampling import update_probabilities
from afs_rgmnb.methods.ensemble import VotingEnsemble
from afs_rgmnb.methods.mnb import MNBParameters, SubspaceMNB


class RevisedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile([11, 27, 43], 20)
        rows = np.arange(len(self.labels))
        self.X = sparse.csr_matrix((rows + 1, (rows, rows % 6)), shape=(len(rows), 8000))
        self.split = GroupSplit(rows[:36], rows[36:48], rows[48:])
        self.config = search.AdaptiveConfig(3, 4, 2, 1., .05, 1., .01, 2, 4)
        self.pso = pso.PSOConfig(3, 2, 1.4, 1.4, .0001)

    def options(self):
        return dict(adaptive=self.config, pso=self.pso, seed=42, fold_index=0,
                    source_feature_ids=np.arange(self.X.shape[1]) + 101)

    def test_stage_order_actual_calls_and_final_subspaces(self):
        events = []
        seen = []
        select = search.select_round
        optimize = pso.optimize_mnb
        original_fit = SubspaceMNB.from_statistics

        def one_round(*args, **kwargs):
            result = select(*args, **kwargs)
            seen.append(result)
            events.append("round")
            return result

        def one_swarm(*args, **kwargs):
            index = events.count("swarm")
            self.assertEqual(events.count("round"), 4)
            np.testing.assert_array_equal(kwargs["feature_indices"], seen[-1][index].selected_subspace)
            self.assertEqual(set(kwargs), {"classes", "feature_indices", "n_features_in", "config", "seed"})
            result = optimize(*args, **kwargs)
            events.append("swarm")
            return result

        class TracedVote(VotingEnsemble):
            def __init__(self, members):
                super().__init__(members)
                events.append("ensemble")

            def predict(self, X):
                events.append("predict")
                return super().predict(X)

        with patch.object(search, "select_round", side_effect=one_round) as rounds, \
             patch.object(search, "jmh_scores", wraps=search.jmh_scores) as scores, \
             patch.object(search, "class_statistics", wraps=search.class_statistics) as statistics, \
             patch.object(SubspaceMNB, "from_statistics", side_effect=original_fit) as fitted, \
             patch.object(search, "weight_change", return_value=.5), \
             patch.object(pso, "optimize_mnb", side_effect=one_swarm), \
             patch.object(runner, "VotingEnsemble", TracedVote):
            record = runner.evaluate_fold(self.X, self.labels, self.split, **self.options())
        self.assertEqual(events, ["round"] * 4 + ["swarm"] * 3 + ["ensemble", "predict"])
        self.assertEqual((scores.call_count, statistics.call_count, fitted.call_count), (1, 1, 48))
        self.assertEqual((record["candidate_evaluations"], record["independent_searches"],
                          record["ensemble_members"], record["stop"]), (48, 3, 3, "tmax"))
        rng = rounds.call_args_list[0].kwargs["rng"]
        for i in range(1, 4):
            expected = update_probabilities([g.selected_subspace for g in seen[i-1]], n_features=6, c0=1.)
            np.testing.assert_array_equal(rounds.call_args_list[i].kwargs["weights"], expected)
            self.assertIs(rounds.call_args_list[i].kwargs["rng"], rng)

    def test_inclusive_consecutive_condition_and_reset(self):
        cfg = replace(self.config, delta_q=.25, tmax=7)
        with patch.object(search, "weight_change", side_effect=[.25, .26, .25, .24]):
            result = search.search_subspaces(self.X[:36], self.labels[:36], self.X[36:48],
                                             self.labels[36:48], config=cfg, rng=np.random.default_rng(9))
        self.assertEqual(result.stop_reason, "consecutive_stability")
        self.assertEqual([h["consecutive"] for h in result.history], [1, 0, 1, 2])
        self.assertEqual(len(result.initial_weights), 8000)

    def test_control_domains(self):
        for changes in ({"rq": 0}, {"rq": True}, {"rq": 2.5}, {"tmax": 1},
                        {"delta_q": 0}, {"delta_q": 2}, {"delta_q": float("nan")}, {"c0": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_fixed_partition_test_values_and_labels_isolated_and_serializable(self):
        modified = self.X.tolil()
        modified[48:, 7000] = 99
        modified[48:, :6] = 7
        labels = self.labels.copy()
        labels[48:] = np.roll(labels[48:], 1)
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            first = runner.evaluate_fold(self.X, self.labels, self.split, output=a, **self.options())
            second = runner.evaluate_fold(modified.tocsr(), labels, self.split, output=b, **self.options())
            self.assertEqual(first["weight_history"], second["weight_history"])
            self.assertEqual(first["final_source_feature_ids"], second["final_source_feature_ids"])
            self.assertEqual([r["history"] for r in first["pso"]], [r["history"] for r in second["pso"]])
            with np.load(a / "ensemble.npz") as original, np.load(b / "ensemble.npz") as repeat:
                for k in original.files:
                    np.testing.assert_array_equal(original[k], repeat[k])
            with np.load(a / "fold.npz") as original, np.load(b / "fold.npz") as repeat:
                for k in ("fit_columns", "source_feature_ids", "initial_weights", "next_weights"):
                    np.testing.assert_array_equal(original[k], repeat[k])
            model = VotingEnsemble.load(a / "ensemble.npz")
            with np.load(a / "fold.npz") as saved:
                prediction = model.predict(self.X[saved["test"]][:, saved["fit_columns"]])
                np.testing.assert_array_equal(prediction, saved["predictions"])
                np.testing.assert_array_equal(saved["source_feature_ids"], np.arange(101, 107))

    def test_each_swarm_generates_fresh_parameters_after_search(self):
        # A one-particle swarm stops at initialization, exposing exactly the
        # randomly generated parameters supplied to each retained model.
        cfg = replace(self.config, rq=1, tmax=1, subspace_size=6)
        small = self.X[:, :6]
        draws = []
        original = pso.generate_parameters

        def generate(*args, **kwargs):
            self.assertEqual(rounds.call_count, 1)
            result = original(*args, **kwargs)
            draws.append(result)
            return result

        with patch.object(search, "select_round", wraps=search.select_round) as rounds, \
             patch.object(pso, "generate_parameters", side_effect=generate):
            ensemble, _, _, _ = runner.fit_ensemble(
                small[:36], self.labels[:36], small[36:48], self.labels[36:48],
                adaptive=cfg, pso=replace(self.pso, n_particles=1), search_seed=3, pso_seed=4)
        self.assertEqual(len(draws), cfg.n_groups)
        for draw, member in zip(draws, ensemble.members):
            np.testing.assert_array_equal(draw.class_prior, member.class_prior_)
            np.testing.assert_array_equal(draw.feature_prob, member.feature_prob_)
            self.assertFalse(np.array_equal(draw.class_prior, np.full(3, 1/3)))
        self.assertEqual(len({d.class_prior.tobytes() for d in draws}), cfg.n_groups)

    def test_equal_votes_member_columns_and_fixed_class_order(self):
        # Three different magnitudes. Two weak votes for class 9 defeat one strong vote for 2.
        classes = np.array([9, 2, 15])
        def member(features, p):
            return SubspaceMNB.from_parameters(MNBParameters(
                classes, np.asarray(features), np.full(3, 1/3), np.column_stack((p, 1-np.array(p))), 5))
        weak = member([0, 1], [.51, .50, .49])
        strong = member([2, 3], [.001, .999, .5])
        third = member([4, 1], [.50, .49, .51])
        X = sparse.csr_matrix([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
        np.testing.assert_array_equal(VotingEnsemble([weak, strong, third]).predict(X), [9, 9])
        np.testing.assert_array_equal(VotingEnsemble([weak, strong, weak]).predict(X), [9, 15])

    def test_full_cv_aggregation_and_fold_independent_seeds(self):
        cfg = replace(self.config, tmax=1, rq=1, n_groups=1, candidates_per_group=1)
        kwargs = dict(adaptive=cfg, pso=self.pso, n_splits=3, search_fraction=.5, seed=8,
                      sample_ids=np.arange(len(self.labels)), source_feature_ids=np.arange(8000))
        with patch.object(search, "jmh_scores", wraps=search.jmh_scores) as scores:
            all_folds = runner.run_cv(self.X, self.labels, **kwargs)
        self.assertEqual(scores.call_count, 3)
        acc = [r["test_accuracy"] for r in all_folds["folds"]]
        self.assertEqual(all_folds["cv_mean_accuracy"], np.mean(acc))
        self.assertEqual(all_folds["cv_std_accuracy_ddof0"], np.std(acc))
        subset = runner.run_cv(self.X, self.labels, folds=[2], **kwargs)
        self.assertIsNone(subset["cv_mean_accuracy"])
        self.assertEqual(subset["folds"][0]["weight_history"], all_folds["folds"][2]["weight_history"])
