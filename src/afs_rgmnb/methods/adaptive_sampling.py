"""Chapter 3 initial weights, inclusion updates and L1 change.

No model fitness is used in the inclusion update. No old weights/JMH scores are
mixed into the later RaSE update. All-zero initial scores fail explicitly because
the active equation has no defined fallback (see methodology question Q01).
"""
from __future__ import annotations

import numpy as np


def validate_probabilities(values) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if (weights.ndim != 1 or weights.size < 2 or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Weights must be a positive finite distribution of length at least two")
    return weights


def initial_probabilities(scores, *, epsilon: float) -> np.ndarray:
    """Equation (3.4), preserving the epsilon/d exploration floor."""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or scores.size < 2 or not np.all(np.isfinite(scores))
            or np.any(scores < 0)):
        raise ValueError("scores must be finite nonnegative values for at least two features")
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    # Scale first to avoid overflow in sum while preserving score ratios.
    scale = float(scores.max())
    if scale == 0:
        raise ValueError("All JMH scores are zero; active Chapter 3 equation (3.4) is undefined")
    normalized = scores / scale
    normalized /= normalized.sum()
    return validate_probabilities((1 - epsilon) * normalized + epsilon / scores.size)


def inclusion_proportions(selected_subspaces, *, n_features: int) -> np.ndarray:
    """Equation (3.17), with fixed m and one selected subspace per group."""
    if isinstance(n_features, bool) or not isinstance(n_features, (int, np.integer)) or n_features < 2:
        raise ValueError("n_features must be an integer at least two")
    selected_subspaces = list(selected_subspaces)
    if not selected_subspaces:
        raise ValueError("At least one selected subspace is required")
    counts = np.zeros(n_features, dtype=np.float64)
    fixed_size = None
    for subspace in selected_subspaces:
        indices = np.asarray(subspace)
        if (indices.ndim != 1 or indices.size < 2 or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= n_features)
                or np.unique(indices).size != indices.size):
            raise ValueError("Subspaces must contain at least two unique valid integer indices")
        if fixed_size is not None and fixed_size != indices.size:
            raise ValueError("Chapter 3 requires all subspaces to have the same fixed size m")
        fixed_size = indices.size
        counts[indices] += 1
    return counts / len(selected_subspaces)


def update_probabilities(selected_subspaces, *, n_features: int, c0: float) -> np.ndarray:
    """Equations (3.17)-(3.19); threshold uses strict > C0/log(d)."""
    if not np.isfinite(c0) or c0 <= 0:
        raise ValueError("c0 must be finite and positive")
    proportions = inclusion_proportions(selected_subspaces, n_features=n_features)
    thresholded = np.where(proportions > c0 / np.log(n_features),
                           proportions, c0 / n_features)
    # Stable normalization without overflow for finite large c0.
    thresholded /= thresholded.max()
    return validate_probabilities(thresholded / thresholded.sum())


def weight_change(current, next_weights) -> float:
    """Equation (3.20); no stopping-policy tolerance is invented here."""
    current = validate_probabilities(current)
    next_weights = validate_probabilities(next_weights)
    if current.shape != next_weights.shape:
        raise ValueError("Successive distributions must use the same feature columns")
    return float(np.abs(next_weights - current).sum())
