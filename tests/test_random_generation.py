"""Numerical checks of Chapter 3 (3.10)-(3.11) and particle serialization."""
import unittest

import numpy as np

from afs_rgmnb.methods.mnb import MNBParameters, SubspaceMNB
from afs_rgmnb.methods.random_generation import (
    generate_parameters, normalize_position, parameters_to_position,
    position_to_parameters,
)


class ScriptedGenerator(np.random.Generator):
    """A Generator with recorded, specified draws for equation-level tests."""

    def __init__(self, draws):
        super().__init__(np.random.PCG64(0))
        self.draws = iter(draws)
        self.calls = []

    def random(self, size=None):
        self.calls.append(size)
        result = np.array(next(self.draws), dtype=np.float64)
        expected_shape = (size,) if isinstance(size, int) else size
        if result.shape != expected_shape:
            raise AssertionError(f"Expected draw shape {expected_shape}, got {result.shape}")
        return result


class RandomGenerationTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array(["z", "a", "m"])
        self.indices = np.array([4, 1])
        self.parameters = MNBParameters(
            self.classes.copy(), self.indices.copy(), np.array([.2, .3, .5]),
            np.array([[.25, .75], [.6, .4], [.1, .9]]), 5)

    def test_hand_draws_use_independent_priors_and_per_class_blocks(self):
        # Equation (3.11): each denominator includes only its own block.
        rng = ScriptedGenerator([[.2, .3, .5], [[.1, .3], [.8, .2], [.3, .6]]])
        parameters = generate_parameters(self.classes, self.indices, 5, rng=rng)
        self.assertEqual(rng.calls, [3, (3, 2)])
        np.testing.assert_array_equal(parameters.classes, self.classes)
        np.testing.assert_array_equal(parameters.feature_indices, [4, 1])
        np.testing.assert_allclose(parameters.class_prior, [.2, .3, .5])
        np.testing.assert_allclose(parameters.feature_prob,
                                   [[1 / 4, 3 / 4], [4 / 5, 1 / 5], [1 / 3, 2 / 3]])
        np.testing.assert_allclose(parameters.feature_prob.sum(axis=1), np.ones(3))
        self.assertEqual(parameters.n_features_in, 5)

    def test_exact_zero_draws_are_redrawn_without_changing_other_coordinates(self):
        rng = ScriptedGenerator([
            [0, .5], [0], [.25],
            [[.2, 0], [.3, .6]], [.8],
        ])
        parameters = generate_parameters([2, 1], [3, 0], 4, rng=rng)
        self.assertEqual(rng.calls, [2, 1, 1, (2, 2), 1])
        np.testing.assert_allclose(parameters.class_prior, [1 / 3, 2 / 3])
        np.testing.assert_allclose(parameters.feature_prob, [[.2, .8], [1 / 3, 2 / 3]])
        self.assertTrue(np.all(parameters_to_position(parameters) > 0))

    def test_seeded_draws_match_normalized_uniform_reference_and_repeat(self):
        reference = np.random.default_rng(20261117)
        u = reference.random(3)
        v = reference.random((3, 2))
        actual = generate_parameters(self.classes, self.indices, 5,
                                     rng=np.random.default_rng(20261117))
        repeated = generate_parameters(self.classes, self.indices, 5,
                                       rng=np.random.default_rng(20261117))
        np.testing.assert_array_equal(actual.class_prior, u / u.sum())
        np.testing.assert_array_equal(actual.feature_prob, v / v.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(parameters_to_position(actual), parameters_to_position(repeated))
        rng = np.random.default_rng(20261117)
        first = generate_parameters(self.classes, self.indices, 5, rng=rng)
        second = generate_parameters(self.classes, self.indices, 5, rng=rng)
        self.assertFalse(np.array_equal(parameters_to_position(first), parameters_to_position(second)))

    def test_flat_layout_and_roundtrip_preserve_metadata_and_predictions(self):
        expected = [.2, .3, .5, .25, .75, .6, .4, .1, .9]
        position = parameters_to_position(self.parameters)
        np.testing.assert_array_equal(position, expected)
        self.assertEqual(position.shape, (3 * (1 + 2),))
        self.assertEqual(position.dtype, np.float64)
        restored = position_to_parameters(position, self.classes, self.indices, 5)
        np.testing.assert_array_equal(parameters_to_position(restored), position)
        np.testing.assert_array_equal(restored.classes, self.classes)
        np.testing.assert_array_equal(restored.feature_indices, self.indices)
        query = [[0, 1, 0, 0, 4], [0, 3, 0, 0, 0]]
        np.testing.assert_array_equal(SubspaceMNB.from_parameters(self.parameters).predict(query),
                                      SubspaceMNB.from_parameters(restored).predict(query))

    def test_serialization_and_generation_return_independent_float64_arrays(self):
        position = parameters_to_position(self.parameters)
        restored = position_to_parameters(position, self.classes, self.indices, 5)
        position[:] = .01
        self.classes[0] = "x"
        self.indices[0] = 0
        np.testing.assert_array_equal(restored.classes, ["z", "a", "m"])
        np.testing.assert_array_equal(restored.feature_indices, [4, 1])
        np.testing.assert_array_equal(restored.class_prior, [.2, .3, .5])
        self.assertEqual(restored.class_prior.dtype, np.float64)
        self.assertEqual(restored.feature_prob.dtype, np.float64)
        restored.class_prior[0] = .9
        self.assertEqual(self.parameters.class_prior[0], .2)

    def test_normalization_uses_every_prior_and_separate_conditional_rows(self):
        position = np.array([1, 2, 3, 2, 6, 3, 1, 8, 2])
        saved = position.copy()
        normalized = normalize_position(position, 3, 2)
        np.testing.assert_allclose(normalized, [1 / 6, 2 / 6, 3 / 6,
                                                .25, .75, .75, .25, .8, .2])
        np.testing.assert_array_equal(position, saved)
        self.assertEqual(normalized.dtype, np.float64)
        position_to_parameters(normalized, self.classes, self.indices, 5)

    def test_invalid_classes_indices_dimensions_and_rng_are_rejected(self):
        invalid_classes = [[], ["a"], ["a", "a"], ["a", ""], [1, "a"],
                           [1, None], [1, np.nan], [1, np.inf], [["a"], ["b"]]]
        for classes in invalid_classes:
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                generate_parameters(classes, [0, 1], 3, rng=np.random.default_rng(0))
        for indices in [[], [0], [0, 0], [-1, 1], [0, 5], [0., 1.],
                        [True, False], [[0, 1]]]:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                generate_parameters(self.classes, indices, 5, rng=np.random.default_rng(0))
        for n_features in [1, 2.5, True, np.bool_(True), "5", None]:
            with self.subTest(n_features=n_features), self.assertRaises(ValueError):
                generate_parameters(self.classes, self.indices, n_features,
                                    rng=np.random.default_rng(0))
        for rng in [None, 1, np.random.RandomState(0), object()]:
            with self.subTest(rng=rng), self.assertRaises(ValueError):
                generate_parameters(self.classes, self.indices, 5, rng=rng)
        for dimension in [0, 1, True, np.bool_(True), 2.0, None]:
            for dimensions in [(dimension, 2), (3, dimension)]:
                with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                    normalize_position(np.ones(9), *dimensions)

    def test_invalid_positions_are_rejected_without_implicit_normalization(self):
        valid = parameters_to_position(self.parameters)
        invalid = [[], valid[:-1], valid.reshape(3, 3), valid.astype(complex), [object()] * 9]
        for value in [0, -1, np.nan, np.inf]:
            changed = valid.copy()
            changed[4] = value
            invalid.append(changed)
        for position in invalid:
            for operation in [lambda: normalize_position(position, 3, 2),
                              lambda: position_to_parameters(position, self.classes, self.indices, 5)]:
                with self.subTest(position=position), self.assertRaises(ValueError):
                    operation()
        for coordinate in [0, 3]:
            unnormalized = valid.copy()
            unnormalized[coordinate] += .1
            with self.assertRaises(ValueError):
                position_to_parameters(unnormalized, self.classes, self.indices, 5)
        with self.assertRaises(ValueError):
            normalize_position(np.full(9, np.finfo(np.float64).max), 3, 2)

    def test_packing_validates_probabilities_and_parameter_type(self):
        for value in [None, {}, [1, 2]]:
            with self.assertRaises(ValueError):
                parameters_to_position(value)
        for prior, conditional in [([.2, .3, .6], self.parameters.feature_prob),
                                   ([0, .5, .5], self.parameters.feature_prob),
                                   (self.parameters.class_prior, [[.5, .5]]),
                                   (self.parameters.class_prior, np.ones((3, 2))),
                                   ([np.nan, .5, .5], self.parameters.feature_prob)]:
            parameters = MNBParameters(self.classes, self.indices, prior, conditional, 5)
            with self.assertRaises(ValueError):
                parameters_to_position(parameters)


if __name__ == "__main__":
    unittest.main()
