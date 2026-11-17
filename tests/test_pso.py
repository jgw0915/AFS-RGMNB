"""Numerical examples for active Ch3 PSO, with Q07's explicit position map."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy import sparse

from afs_rgmnb.methods.pso import (
    PSOConfig, _move_swarm, _velocity_candidate, optimize_mnb,
)
from afs_rgmnb.methods.random_generation import parameters_to_position


class PSOTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array(["a", "b"])
        self.indices = np.array([0, 1])
        self.z0 = np.array([.6, .4, .7, .3, .2, .8])
        self.z1 = np.array([.4, .6, .3, .7, .8, .2])
        self.X = sparse.csr_matrix([[0, 0], [0, 0]])
        self.y = np.array(["a", "b"])

    def optimize(self, *, config=None, **kwargs):
        options = dict(classes=self.classes, feature_indices=self.indices,
                       n_features_in=2, config=PSOConfig(2, 1, .5, .5, 1) if config is None else config,
                       seed=71)
        options.update(kwargs)
        return optimize_mnb(self.X, self.y, **options)

    def test_equation_315_hand_calculated_scalar_rhos_and_fixed_inertia(self):
        positions = np.array([[.2, .3, .4], [.7, .6, .5]])
        velocities = np.array([[.1, -.2, .3], [-.1, .2, -.3]])
        personal = np.array([[.3, .5, .7], [.5, .4, .3]])
        global_best = np.array([.4, .4, .4])
        # For particle 0: .9*v + .5*(p-z) + .1*(g-z).
        # For particle 1: .9*v + 1.5*(p-z) + .3*(g-z).
        result = _velocity_candidate(
            positions, velocities, personal, global_best, c1=2, c2=.4,
            rhos=np.array([[.25, .25], [.75, .75]]))
        np.testing.assert_allclose(result, [[.16, -.07, .42], [-.48, -.18, -.60]],
                                   rtol=0, atol=1e-15)

    def test_all_particles_use_old_global_best_before_any_fitness_update(self):
        rng = Mock()
        rng.random.return_value = np.full((2, 2), .5)
        initial = [(self.z0.copy(), np.zeros(6)), (self.z1.copy(), np.zeros(6))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng), \
                patch("afs_rgmnb.methods.pso._search_fitness", side_effect=[.25, .5, 1., .5]):
            result = self.optimize()
        # Particle 0 becomes the best after moving one quarter toward old p_*.
        # Particle 1 remains at the old p_*, despite particle 0's improvement.
        expected_first = np.array([.55, .45, .6, .4, .35, .65])
        np.testing.assert_allclose(result.positions, [expected_first, self.z1], atol=1e-15)
        np.testing.assert_allclose(result.personal_best_positions, [expected_first, self.z1])
        self.assertEqual(result.history[0].best_particle_index, 1)
        self.assertEqual(result.history[1].best_particle_index, 0)
        self.assertEqual(result.best_fitness, 1.)
        self.assertEqual(result.fitness_evaluation_count, 4)
        rng.random.assert_called_once_with((2, 2))  # Two scalar draws per particle.

    def test_speed_outside_bound_redraws_only_that_coordinate_then_block_normalizes(self):
        z = np.array([[.5, .5, .4, .6, .7, .3]])
        velocity = np.array([[2., 0, 0, 0, 0, 0]])
        rng = Mock()
        rng.random.return_value = np.zeros((1, 2))
        rng.uniform.return_value = np.array([.2])
        config = PSOConfig(1, 1, 1, 1, 1)
        next_z, next_v, redraws, resets = _move_swarm(
            z, velocity, z.copy(), z[0].copy(), classes=self.classes,
            feature_indices=self.indices, n_features_in=2, config=config, rng=rng)
        # .9*2=1.8 exceeds vmax; it is redrawn as .2 rather than clipped to 1.
        np.testing.assert_array_equal(next_v, [[.2, 0, 0, 0, 0, 0]])
        np.testing.assert_allclose(next_z, [[7 / 12, 5 / 12, .4, .6, .7, .3]])
        rng.uniform.assert_called_once_with(-1., 1., size=1)
        self.assertEqual((redraws, resets), (1, 0))
        np.testing.assert_array_equal(z, [[.5, .5, .4, .6, .7, .3]])
        np.testing.assert_array_equal(velocity, [[2., 0, 0, 0, 0, 0]])

    def test_speed_at_both_bounds_is_retained(self):
        z = np.array([[.5, .5, .4, .6, .7, .3]])
        rng = Mock()
        rng.random.return_value = np.zeros((1, 2))
        with patch("afs_rgmnb.methods.pso._velocity_candidate",
                   return_value=np.array([[.1, -.1, 0, 0, 0, 0]])):
            result = _move_swarm(
                z, np.zeros_like(z), z.copy(), z[0].copy(), classes=self.classes,
                feature_indices=self.indices, n_features_in=2,
                config=PSOConfig(1, 1, 1, 1, .1), rng=rng)
        np.testing.assert_array_equal(result[1], [[.1, -.1, 0, 0, 0, 0]])
        self.assertEqual(result[2:], (0, 0))
        rng.uniform.assert_not_called()

    def test_zero_or_one_trial_coordinate_resets_whole_position_and_velocity(self):
        z = np.array([[.5, .5, .4, .6, .7, .3]])
        replacement_velocity = np.array([-.2, .1, .2, -.1, .1, -.2])
        for first_velocity in [-.5, .5]:
            with self.subTest(trial_coordinate=.5 + first_velocity):
                rng = Mock()
                rng.random.return_value = np.zeros((1, 2))
                candidate = np.array([[first_velocity, .1, 0, 0, 0, 0]])
                personal = z.copy()
                global_best = z[0].copy()
                with patch("afs_rgmnb.methods.pso._velocity_candidate", return_value=candidate), \
                        patch("afs_rgmnb.methods.pso._initialize_particle",
                              return_value=(self.z1.copy(), replacement_velocity.copy())) as initialize:
                    next_z, next_v, redraws, resets = _move_swarm(
                        z, np.zeros_like(z), personal, global_best, classes=self.classes,
                        feature_indices=self.indices, n_features_in=2,
                        config=PSOConfig(1, 1, 1, 1, 1), rng=rng)
                np.testing.assert_array_equal(next_z[0], self.z1)
                np.testing.assert_array_equal(next_v[0], replacement_velocity)
                np.testing.assert_array_equal(personal, z)
                np.testing.assert_array_equal(global_best, z[0])
                self.assertEqual((redraws, resets), (0, 1))
                initialize.assert_called_once()

    def test_whole_particle_reset_retains_better_personal_and_global_history(self):
        rng = Mock()
        rng.random.return_value = np.zeros((2, 2))
        replacement = np.full(6, .5)
        initial = [(self.z0.copy(), np.array([-.8, 0, 0, 0, 0, 0])),
                   (self.z1.copy(), np.zeros(6)), (replacement, np.full(6, .1))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng), \
                patch("afs_rgmnb.methods.pso._search_fitness", side_effect=[.75, .25, .1, .25]):
            result = self.optimize()
        np.testing.assert_array_equal(result.positions[0], replacement)
        np.testing.assert_array_equal(result.velocities[0], np.full(6, .1))
        np.testing.assert_array_equal(result.personal_best_positions[0], self.z0)
        np.testing.assert_array_equal(parameters_to_position(result.model.parameters_), self.z0)
        np.testing.assert_array_equal(result.personal_best_scores, [.75, .25])
        self.assertEqual(result.particle_reset_count, 1)
        self.assertEqual(result.velocity_redraw_count, 0)
        self.assertEqual(result.history[1].particle_reset_count, 1)
        self.assertEqual(result.boundary_counts["whole_particles_reinitialized"], 1)

    def test_equal_accuracy_keeps_old_personal_bests_and_first_global_particle(self):
        rng = Mock()
        rng.random.return_value = np.zeros((2, 2))
        initial = [(self.z0.copy(), np.array([.01, -.01, 0, 0, 0, 0])),
                   (self.z1.copy(), np.zeros(6))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng):
            result = self.optimize(config=PSOConfig(2, 3, .5, .5, 1))
        self.assertFalse(np.array_equal(result.positions[0], self.z0))
        np.testing.assert_array_equal(result.personal_best_positions, [self.z0, self.z1])
        np.testing.assert_array_equal(parameters_to_position(result.model.parameters_), self.z0)
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.fitness_evaluation_count, 8)
        for entry in result.history:
            self.assertEqual(entry.fitness_scores, (.5, .5))
            self.assertEqual(entry.personal_best_scores, (.5, .5))
            self.assertEqual(entry.best_particle_index, 0)

    def test_exact_position_equality_stops_at_initialization_including_one_particle(self):
        for n_particles in [1, 2]:
            with self.subTest(n_particles=n_particles), \
                    patch("afs_rgmnb.methods.pso._initialize_particle",
                          return_value=(self.z0.copy(), np.zeros(6))):
                result = self.optimize(config=PSOConfig(n_particles, 5, 1, 1, .1))
            self.assertEqual(result.stop_reason, "positions_identical")
            self.assertEqual(result.iterations, 0)
            self.assertEqual(result.fitness_evaluation_count, n_particles)
            self.assertEqual(len(result.history), 1)

    def test_exact_position_equality_after_update_stops_before_iteration_limit(self):
        first = np.array([.4, .6, .5, .5, .5, .5])
        second = np.array([.6, .4, .5, .5, .5, .5])
        rng = Mock()
        rng.random.return_value = np.array([[0., 0.], [0., 1.]])
        initial = [(first.copy(), np.zeros(6)), (second.copy(), np.zeros(6))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng):
            result = self.optimize(config=PSOConfig(2, 5, 1, 1, .5))
        np.testing.assert_array_equal(result.positions, [first, first])
        self.assertEqual(result.stop_reason, "positions_identical")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.fitness_evaluation_count, 4)
        self.assertFalse(result.history[0].positions_identical)
        self.assertTrue(result.history[1].positions_identical)

    def test_approximately_equal_positions_do_not_trigger_convergence(self):
        first = np.full(6, .5)
        second = first.copy()
        second[0] = np.nextafter(.5, 1.)
        self.assertTrue(np.allclose(first, second))
        rng = Mock()
        rng.random.return_value = np.zeros((2, 2))
        initial = [(first, np.zeros(6)), (second, np.zeros(6))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng):
            result = self.optimize()
        self.assertEqual(result.stop_reason, "max_iterations")
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.history[0].positions_identical)
        self.assertFalse(result.history[1].positions_identical)

    def test_fitness_is_natural_sample_accuracy_with_fixed_class_order(self):
        self.X = sparse.csr_matrix((4, 2))
        self.y = np.array(["a", "a", "a", "b"])
        rng = Mock()
        rng.random.return_value = np.zeros((2, 2))
        initial = [(self.z0.copy(), np.zeros(6)), (self.z1.copy(), np.zeros(6))]
        with patch("afs_rgmnb.methods.pso._initialize_particle", side_effect=initial), \
                patch("afs_rgmnb.methods.pso.np.random.default_rng", return_value=rng):
            result = self.optimize()
        self.assertEqual(result.history[0].fitness_scores, (.75, .25))
        self.assertEqual(result.best_fitness, .75)
        np.testing.assert_array_equal(result.model.predict(self.X), ["a"] * 4)

    def test_repeatability_sparse_multiclass_mapping_counts_and_probability_domains(self):
        X = sparse.csr_matrix([[5, 0, 0, 2], [0, 5, 0, 1], [0, 0, 5, 2],
                               [3, 0, 0, 1], [0, 3, 0, 2], [0, 0, 3, 1]])
        y = np.array(["a", "b", "c", "a", "b", "c"])
        options = dict(classes=["c", "a", "b"], feature_indices=[2, 0, 1],
                       n_features_in=4, config=PSOConfig(4, 3, .7, .8, .05), seed=123)
        # A high-dimensional sample matrix must never be densified for PSO.
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense matrix")):
            first = optimize_mnb(X, y, **options)
            repeat = optimize_mnb(X, y, **options)
        np.testing.assert_array_equal(first.positions, repeat.positions)
        np.testing.assert_array_equal(first.velocities, repeat.velocities)
        np.testing.assert_array_equal(first.personal_best_positions, repeat.personal_best_positions)
        np.testing.assert_array_equal(first.personal_best_scores, repeat.personal_best_scores)
        self.assertEqual(first.history, repeat.history)
        self.assertEqual(first.seeds, {"pso": 123})
        self.assertEqual(first.stop_reason, "max_iterations")
        self.assertEqual(first.fitness_evaluation_count, 4 * (1 + 3))
        np.testing.assert_array_equal(first.model.feature_indices_, [2, 0, 1])
        np.testing.assert_array_equal(first.model.classes_, ["c", "a", "b"])
        self.assertEqual(first.model.n_features_in_, 4)
        for population in [first.positions, first.personal_best_positions]:
            self.assertEqual(population.shape, (4, 12))
            self.assertTrue(np.all((population > 0) & (population < 1)))
            np.testing.assert_allclose(population[:, :3].sum(axis=1), 1, atol=1e-15)
            np.testing.assert_allclose(population[:, 3:].reshape(4, 3, 3).sum(axis=2), 1, atol=1e-15)
        self.assertTrue(np.all(np.abs(first.velocities) <= .05))
        self.assertTrue(np.all((first.personal_best_scores >= 0) & (first.personal_best_scores <= 1)))
        self.assertEqual(first.velocity_redraw_count,
                         sum(entry.velocity_redraw_count for entry in first.history))
        self.assertEqual(first.particle_reset_count,
                         sum(entry.particle_reset_count for entry in first.history))
        self.assertEqual(first.best_fitness, np.mean(first.model.predict(X) == y))

    def test_configuration_and_data_domains_fail_clearly(self):
        for field in ["n_particles", "max_iterations"]:
            for bad in [0, -1, 1.5, True, np.nan]:
                config = dict(n_particles=2, max_iterations=1, c1=1, c2=1, vmax=.1)
                config[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    PSOConfig(**config)
        for field in ["c1", "c2", "vmax"]:
            for bad in [0, -1, np.inf, np.nan, True, [1], 1j]:
                config = dict(n_particles=2, max_iterations=1, c1=1, c2=1, vmax=.1)
                config[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    PSOConfig(**config)
        for bad_seed in [-1, 1.5, True, None]:
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                self.optimize(seed=bad_seed)
        for changes in [dict(config={}), dict(classes=["a", "c"]), dict(classes=["a", "a"]),
                        dict(classes=["a"]), dict(feature_indices=[0]),
                        dict(feature_indices=[0, 0]), dict(feature_indices=[0., 1.]),
                        dict(n_features_in=3), dict(n_features_in=True)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.optimize(**changes)
        for X, y in [([[0, -.1], [0, 0]], ["a", "b"]),
                     ([[0, np.inf], [0, 0]], ["a", "b"]),
                     (np.empty((0, 2)), []), ([[0, 0]], ["a"]),
                     ([[0, 0], [0, 0]], ["a"]),
                     ([[0, 0], [0, 0]], ["a", None])]:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                optimize_mnb(X, y, classes=self.classes, feature_indices=self.indices,
                             n_features_in=2, config=PSOConfig(2, 1, 1, 1, .1), seed=1)


if __name__ == "__main__":
    unittest.main()
