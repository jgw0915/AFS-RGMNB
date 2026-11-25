"""Independent-uniform MNB initialization, Chapter 3 equations (3.14)-(3.15).

The fixed particle layout is all M priors followed by the M-by-m conditional
probabilities in row-major order. Feature and class order are preserved, so a
position always contains exactly M(1+m) coordinates.
"""
from __future__ import annotations

import numpy as np

from .mnb import MNBParameters, _class_labels, _indices, _validated_parameters


def _dimension(value, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 2):
        raise ValueError(f"{name} must be an integer at least two")
    return int(value)


def _layout(classes, feature_indices, n_features_in):
    classes = _class_labels(classes, name="classes")
    if len(classes) < 2 or np.unique(classes).size != len(classes):
        raise ValueError("classes must contain at least two unique class labels")
    n_features_in = _dimension(n_features_in, "n_features_in")
    return classes.copy(), _indices(feature_indices, n_features_in), n_features_in


def _positive_position(position, n_classes: int, subspace_size: int) -> np.ndarray:
    if not np.isrealobj(position):
        raise ValueError("Position coordinates must be real")
    try:
        result = np.asarray(position, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Position coordinates must be finite positive numbers") from exc
    if (result.shape != (n_classes * (1 + subspace_size),)
            or not np.all(np.isfinite(result)) or np.any(result <= 0)):
        raise ValueError("Position must be flat, correctly sized, finite and strictly positive")
    return result.copy()


def normalize_position(position, n_classes: int, subspace_size: int) -> np.ndarray:
    """Normalize positive coordinates by prior and per-class conditional blocks.

    Range checks for a PSO proposed position precede this operation in the
    search procedure. This helper accepts any finite positive block weights;
    it never clips, smooths or replaces invalid coordinates.
    """
    n_classes = _dimension(n_classes, "n_classes")
    subspace_size = _dimension(subspace_size, "subspace_size")
    result = _positive_position(position, n_classes, subspace_size)
    prior = result[:n_classes]
    conditional = result[n_classes:].reshape(n_classes, subspace_size)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        prior /= prior.sum()
        conditional /= conditional.sum(axis=1, keepdims=True)
    if (not np.all(np.isfinite(result)) or np.any(result <= 0)
            or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-12)
            or not np.allclose(conditional.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Probability blocks cannot be normalized safely in float64")
    return result


def _open_uniform(rng: np.random.Generator, size) -> np.ndarray:
    # Generator.random is [0,1); rejection of zero implements the open interval
    # without clipping it to an arbitrary lower probability or redrawing a block.
    draws = rng.random(size)
    zero = draws == 0
    while np.any(zero):
        draws[zero] = rng.random(int(np.count_nonzero(zero)))
        zero = draws == 0
    return draws


def generate_parameters(classes, feature_indices, n_features_in, *,
                        rng: np.random.Generator) -> MNBParameters:
    """Draw independent U_y and V_yj and normalize exactly as equation (3.15).

    No class counts or estimated parameters enter this initialization. The
    caller owns and seeds the Generator; independent calls consume new draws.
    """
    classes, indices, n_features_in = _layout(classes, feature_indices, n_features_in)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    n_classes, subspace_size = len(classes), len(indices)
    prior = _open_uniform(rng, n_classes)
    conditional = _open_uniform(rng, (n_classes, subspace_size))
    position = normalize_position(np.concatenate((prior, conditional.ravel())),
                                  n_classes, subspace_size)
    return position_to_parameters(position, classes, indices, n_features_in)


def parameters_to_position(parameters: MNBParameters) -> np.ndarray:
    """Validate and serialize priors first, then each class's theta row."""
    parameters = _validated_parameters(parameters)
    return np.concatenate((parameters.class_prior, parameters.feature_prob.ravel()))


def position_to_parameters(position, classes, feature_indices, n_features_in) -> MNBParameters:
    """Deserialize an already normalized positive particle into an MNB model.

    Class and feature metadata are supplied explicitly and retain their order.
    This operation validates probability sums rather than normalizing them.
    """
    classes, indices, n_features_in = _layout(classes, feature_indices, n_features_in)
    n_classes, subspace_size = len(classes), len(indices)
    position = _positive_position(position, n_classes, subspace_size)
    return _validated_parameters(MNBParameters(
        classes, indices, position[:n_classes],
        position[n_classes:].reshape(n_classes, subspace_size), n_features_in))
