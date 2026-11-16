"""Exact sequential weighted sampling without replacement, Ch3 equation (3.6)."""
from __future__ import annotations

import numpy as np

from .adaptive_sampling import validate_probabilities


def sample_subspace(weights, size: int, *, rng: np.random.Generator) -> np.ndarray:
    """Draw in order, renormalizing remaining weights after each feature.

    RNG is supplied by the experiment; independent calls may yield identical
    sets. Returns zero-based indices in draw order, without changing weights.
    """
    weights = validate_probabilities(weights)
    if (isinstance(size, bool) or not isinstance(size, (int, np.integer))
            or not 2 <= size <= len(weights)):
        raise ValueError("Fixed subspace size must be an integer in [2,d]")
    remaining = np.arange(len(weights))
    chosen = np.empty(size, dtype=np.int64)
    for step in range(size):
        conditional = weights[remaining]
        conditional = conditional / conditional.sum()
        position = rng.choice(len(remaining), p=conditional)
        chosen[step] = remaining[position]
        remaining = np.delete(remaining, position)
    return chosen
