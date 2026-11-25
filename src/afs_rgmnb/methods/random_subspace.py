"""Exact sequential weighted sampling without replacement, Ch3 equation (3.6)."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import time

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


@dataclass
class CandidateResult:
    feature_indices: np.ndarray
    accuracy: float
    correct_predictions: int
    search_samples: int


@dataclass
class SelectedGroup:
    candidates: tuple[CandidateResult, ...]
    selected_index: int
    tied_best_indices: tuple[int, ...]
    model: object

    @property
    def selected_subspace(self) -> np.ndarray:
        return self.candidates[self.selected_index].feature_indices.copy()


@dataclass
class CandidateSelection:
    groups: tuple[SelectedGroup, ...]
    classes: np.ndarray
    feature_scores: np.ndarray
    sampling_weights: np.ndarray
    seed: int
    timings: dict[str, float]

    @property
    def selected_subspaces(self) -> tuple[np.ndarray, ...]:
        return tuple(group.selected_subspace for group in self.groups)


def select_subspaces(X_fit, y_fit, X_search, y_search, *, n_groups: int,
                     candidates_per_group: int, subspace_size: int, alpha: float,
                     epsilon: float, seed: int, class_order=None) -> CandidateSelection:
    """Chapter 3 (3.1)-(3.9), exactly one best search-accuracy candidate/group.

    JMH is calculated once on fit only; all candidates reuse fit statistics.
    Independent draws may repeat sets. First-candidate ties are deterministic.
    No test-data argument is accepted. These are fitted candidates, not random
    MNB parameter generation or a completed AFS-RGMNB classifier.
    """
    from .feature_scoring import jmh_scores
    from .adaptive_sampling import initial_probabilities
    from .mnb import class_statistics

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    start = time.perf_counter()
    stats = class_statistics(X_fit, y_fit, class_order=class_order)
    statistics_seconds = time.perf_counter() - start
    score_start = time.perf_counter()
    scores = jmh_scores(X_fit, y_fit, alpha=alpha)
    weights = initial_probabilities(scores, epsilon=epsilon)
    scoring_seconds = time.perf_counter() - score_start
    selection_start = time.perf_counter()
    groups = select_round(stats, X_search, y_search, weights=weights,
                          rng=np.random.default_rng(seed), n_groups=n_groups,
                          candidates_per_group=candidates_per_group,
                          subspace_size=subspace_size, alpha=alpha)
    return CandidateSelection(tuple(groups), stats.classes.copy(), scores, weights,
                              int(seed), {"fit_statistics_seconds": statistics_seconds,
                              "jmh_initial_weights_seconds": scoring_seconds,
                              "sampling_fit_search_selection_seconds": time.perf_counter() - selection_start})


def select_round(stats, X_search, y_search, *, weights, rng, n_groups: int,
                 candidates_per_group: int, subspace_size: int, alpha: float):
    """Equations (3.6)–(3.9). Draw and score one round using the supplied q and RNG."""
    from scipy import sparse
    from .mnb import SubspaceMNB, _class_labels

    weights = validate_probabilities(weights)
    if len(weights) != stats.n_features:
        raise ValueError("Weights must align with fit columns")
    for name, value in (("n_groups", n_groups), ("candidates_per_group", candidates_per_group)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(subspace_size, bool) or not isinstance(subspace_size, (int, np.integer))
            or not 2 <= subspace_size <= stats.n_features):
        raise ValueError("subspace_size must be a fixed integer in [2,d]")
    X_search = sparse.csr_matrix(X_search)
    truth = _class_labels(y_search, name="y_search")
    if X_search.shape[0] == 0 or X_search.shape[1] != stats.n_features:
        raise ValueError("Search data must be nonempty and use the fit feature columns")
    if truth.ndim != 1 or len(truth) != X_search.shape[0]:
        raise ValueError("Search labels must be one-dimensional and aligned")
    try:
        if set(truth.tolist()) != set(stats.classes.tolist()):
            raise ValueError("Search data must represent exactly the fit classes")
    except TypeError as exc:
        raise ValueError("Invalid search class labels") from exc
    groups = []
    for _ in range(n_groups):
        candidates = []
        best_correct, best_index, best_model = -1, -1, None
        for index in range(candidates_per_group):
            features = sample_subspace(weights, subspace_size, rng=rng)
            model = SubspaceMNB.from_statistics(stats, features, alpha=alpha)
            correct = int(np.count_nonzero(model.predict(X_search) == truth))
            candidates.append(CandidateResult(features.copy(), correct / len(truth), correct, len(truth)))
            if correct > best_correct:  # strict improvement: first candidate wins ties
                best_correct, best_index, best_model = correct, index, model
        ties = tuple(i for i, candidate in enumerate(candidates)
                     if candidate.correct_predictions == best_correct)
        groups.append(SelectedGroup(tuple(candidates), best_index, ties, best_model))
    return tuple(groups)
