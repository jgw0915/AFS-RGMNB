"""Feature-subspace construction in §3.3. Model search begins after this returns."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .adaptive_sampling import initial_probabilities, update_probabilities, weight_change
from .feature_scoring import jmh_scores
from .mnb import class_statistics
from .random_subspace import select_round


@dataclass(frozen=True)
class AdaptiveConfig:
    n_groups: int
    candidates_per_group: int
    subspace_size: int
    alpha: float
    epsilon: float
    c0: float
    delta_q: float
    rq: int
    tmax: int

    def __post_init__(self):
        for name in ("n_groups", "candidates_per_group", "subspace_size", "rq", "tmax"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < (2 if name == "subspace_size" else 1)):
                raise ValueError(f"Invalid positive integer setting: {name}")
        for name in ("alpha", "epsilon", "c0", "delta_q"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        if self.delta_q >= 2 or self.epsilon >= 1 or self.rq > self.tmax:
            raise ValueError("Require delta_q < 2, epsilon < 1, and tmax >= rq")


@dataclass
class AdaptiveResult:
    classes: np.ndarray
    selected_subspaces: tuple[np.ndarray, ...]
    history: list[dict]
    initial_weights: np.ndarray
    next_weights: np.ndarray
    stop_reason: str


def search_subspaces(X_fit, y_fit, X_search, y_search, *, config: AdaptiveConfig,
                     rng: np.random.Generator, class_order=None, retain_draws=False) -> AdaptiveResult:
    stats = class_statistics(X_fit, y_fit, class_order=class_order)
    scores = jmh_scores(X_fit, y_fit, alpha=config.alpha)
    initial = initial_probabilities(scores, epsilon=config.epsilon)
    weights = initial
    history = []
    consecutive = 0
    for iteration in range(1, config.tmax + 1):
        groups = select_round(stats, X_search, y_search, weights=weights, rng=rng,
                              n_groups=config.n_groups, candidates_per_group=config.candidates_per_group,
                              subspace_size=config.subspace_size, alpha=config.alpha)
        selected = tuple(group.selected_subspace for group in groups)
        updated = update_probabilities(selected, n_features=stats.n_features, c0=config.c0)
        delta = weight_change(weights, updated)
        consecutive = consecutive + 1 if delta <= config.delta_q else 0
        history.append({
            "iteration": iteration, "delta": delta, "consecutive": consecutive,
            "sampling_weights_sha256": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
            "groups": [{"accuracies": [c.accuracy for c in group.candidates],
                        "selected_index": group.selected_index, "ties": group.tied_best_indices,
                        "selected_fit_columns": group.selected_subspace.tolist()} for group in groups],
        })
        if retain_draws:
            history[-1]["draws"] = [[c.feature_indices.tolist() for c in g.candidates] for g in groups]
        weights = updated
        if consecutive >= config.rq:
            break
    reason = "consecutive_stability" if consecutive >= config.rq else "tmax"
    return AdaptiveResult(stats.classes, selected, history, initial, weights, reason)
