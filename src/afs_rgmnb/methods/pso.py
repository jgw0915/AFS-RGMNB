"""Chapter 3 (3.14)-(3.19) PSO on a fixed MNB feature subspace.

Fitness uses Dsearch accuracy only. Positions store all class priors first,
then row-major class-conditional probabilities. The position map is addition
followed by strict (0, 1) validation and whole-particle redraw; this is the
supporting clarification recorded in methodology open question Q07, rather
than an additional equation claimed to appear in the active chapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mnb import (
    MNBParameters, SubspaceMNB, _classes_and_encoding, _counts, _indices,
    class_log_scores,
)
from .random_generation import (
    generate_parameters, normalize_position, parameters_to_position,
    position_to_parameters,
)


INERTIA_WEIGHT = 0.9


def _positive_scalar(value, name: str) -> float:
    try:
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
            raise ValueError
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return converted


@dataclass(frozen=True)
class PSOConfig:
    """Required search settings; inertia is fixed at 0.9 by the active text."""

    n_particles: int
    max_iterations: int
    c1: float
    c2: float
    vmax: float

    def __post_init__(self):
        for name in ("n_particles", "max_iterations"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("c1", "c2", "vmax"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))


@dataclass(frozen=True)
class PSOIteration:
    """Lightweight history; iteration 0 is initialization, not a PSO update.

    Boundary counts in an entry describe that update; the fitness evaluation
    count is cumulative. Scores describe the current population and its saved
    personal bests. No sample matrices or full position history are retained.
    """

    iteration: int
    best_fitness: float
    best_particle_index: int
    fitness_scores: tuple[float, ...]
    personal_best_scores: tuple[float, ...]
    fitness_evaluation_count: int
    velocity_redraw_count: int
    particle_reset_count: int
    positions_identical: bool


@dataclass
class SwarmDiagnostics:
    """Optional observations; neither selection nor random draws depend on these."""
    initial_positions: np.ndarray
    steps: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class PSOResult:
    model: SubspaceMNB
    best_fitness: float
    history: tuple[PSOIteration, ...]
    positions: np.ndarray
    velocities: np.ndarray
    personal_best_positions: np.ndarray
    personal_best_scores: np.ndarray
    fitness_scores: np.ndarray
    stop_reason: str
    fitness_evaluation_count: int
    velocity_redraw_count: int
    particle_reset_count: int
    seeds: dict[str, int]
    config: PSOConfig
    diagnostics: SwarmDiagnostics | None = None

    @property
    def iterations(self) -> int:
        return len(self.history) - 1

    @property
    def boundary_counts(self) -> dict[str, int]:
        return {"velocity_coordinates_redrawn": self.velocity_redraw_count,
                "whole_particles_reinitialized": self.particle_reset_count}


def _initialize_particle(classes, feature_indices, n_features_in, *, config, rng):
    parameters = generate_parameters(classes, feature_indices, n_features_in, rng=rng)
    position = parameters_to_position(parameters)
    velocity = rng.uniform(-config.vmax, config.vmax, size=position.size)
    return position, velocity


def _velocity_candidate(positions, velocities, personal_best, global_best, *,
                        c1: float, c2: float, rhos: np.ndarray) -> np.ndarray:
    """Equation (3.19): two scalars per particle, each shared by its coordinates."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (INERTIA_WEIGHT * velocities
                + c1 * rhos[:, 0, None] * (personal_best - positions)
                + c2 * rhos[:, 1, None] * (global_best - positions))


def _move_swarm(positions, velocities, personal_best, global_best, *, classes,
                feature_indices, n_features_in, config, rng, trace=None):
    """Move the complete population using the same old historical bests.

    Speed repair redraws only out-of-range coordinates. Position repair
    redraws the entire particle and its velocity using the initial law, while
    this function never mutates personal/global best arrays.
    """
    rhos = rng.random((config.n_particles, 2))
    next_velocities = _velocity_candidate(
        positions, velocities, personal_best, global_best,
        c1=config.c1, c2=config.c2, rhos=rhos)
    next_positions = np.empty_like(positions)
    velocity_redraw_count = particle_reset_count = 0
    repaired_particles = 0
    accepted_movement = 0.0
    for particle in range(config.n_particles):
        velocity = next_velocities[particle]
        invalid_speed = ~np.isfinite(velocity) | (np.abs(velocity) > config.vmax)
        n_invalid = int(np.count_nonzero(invalid_speed))
        if n_invalid:
            repaired_particles += 1
            velocity[invalid_speed] = rng.uniform(-config.vmax, config.vmax, size=n_invalid)
            velocity_redraw_count += n_invalid
        with np.errstate(over="ignore", invalid="ignore"):
            trial = positions[particle] + velocity
        if np.any(~np.isfinite(trial) | (trial <= 0) | (trial >= 1)):
            next_positions[particle], next_velocities[particle] = _initialize_particle(
                classes, feature_indices, n_features_in, config=config, rng=rng)
            particle_reset_count += 1
        else:
            next_positions[particle] = normalize_position(trial, len(classes), len(feature_indices))
            if trace is not None:
                accepted_movement += float(np.abs(next_positions[particle] - positions[particle]).sum())
    if trace is not None:
        trace.steps.append({"repaired_velocities": repaired_particles,
                            "absolute_movement": float(np.abs(next_positions - positions).sum()),
                            "accepted_absolute_movement": accepted_movement})
    return next_positions, next_velocities, velocity_redraw_count, particle_reset_count


def _search_fitness(position, X_selected, truth, *, classes) -> float:
    """Use the existing sparse MNB scorer on a once-selected column matrix."""
    n_classes, n_selected = len(classes), X_selected.shape[1]
    parameters = MNBParameters(
        classes, np.arange(n_selected), position[:n_classes],
        position[n_classes:].reshape(n_classes, n_selected), n_selected)
    scores = class_log_scores(X_selected, parameters)
    predicted = classes[np.argmax(scores, axis=1)]
    return int(np.count_nonzero(predicted == truth)) / len(truth)


def _positions_identical(positions) -> bool:
    # Exact positions, not approximate equality or equality of fitness.
    return bool(np.all(positions == positions[0]))


def optimize_mnb(X_search, y_search, *, classes, feature_indices, n_features_in,
                 config: PSOConfig, seed: int, diagnostics: bool = False) -> PSOResult:
    """Search one selected subspace and return its historical-best MNB model.

    All classes must be represented in search data. No Dfit or Dtest argument
    is accepted, no parameters are estimated from counts, and only accuracy
    determines personal/global bests. A single seeded Generator supplies the
    sequential initialization, rho, and boundary-redraw draws. Each iteration
    draws the population's Np pairs of scalar rhos before boundary handling.

    Exact equality is checked at initialization and after completed updates.
    A single-particle swarm therefore stops at initialization, with one fitness
    evaluation. Otherwise at most max_iterations *updates* are performed in
    addition to initialization. Personal-best ties keep the previous position;
    global-best ties select the first particle in the fixed particle order.
    """
    if not isinstance(config, PSOConfig):
        raise ValueError("config must be PSOConfig")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(n_features_in, (bool, np.bool_))
            or not isinstance(n_features_in, (int, np.integer)) or n_features_in < 2):
        raise ValueError("n_features_in must be an integer at least two")
    if classes is None:
        raise ValueError("classes must specify the fixed class order")
    X = _counts(X_search, n_features=int(n_features_in))
    classes, _ = _classes_and_encoding(y_search, X.shape[0], class_order=classes)
    truth = np.asarray(y_search)
    indices = _indices(feature_indices, int(n_features_in))
    X_selected = X[:, indices].tocsr()
    del X  # Full sample matrices are not retained in the search result.
    rng = np.random.default_rng(seed)
    dimension = len(classes) * (1 + len(indices))
    positions = np.empty((config.n_particles, dimension), dtype=np.float64)
    velocities = np.empty_like(positions)
    for particle in range(config.n_particles):
        positions[particle], velocities[particle] = _initialize_particle(
            classes, indices, int(n_features_in), config=config, rng=rng)
    fitness = np.asarray([
        _search_fitness(position, X_selected, truth, classes=classes)
        for position in positions])
    personal_best = positions.copy()
    trace = SwarmDiagnostics(positions.copy()) if diagnostics else None
    personal_best_scores = fitness.copy()
    evaluations = config.n_particles
    total_velocity_redraws = total_particle_resets = 0
    history = []

    def record(iteration, redraws, resets):
        best_index = int(np.argmax(personal_best_scores))
        entry = PSOIteration(
            iteration, float(personal_best_scores[best_index]), best_index,
            tuple(float(score) for score in fitness),
            tuple(float(score) for score in personal_best_scores), evaluations,
            redraws, resets, _positions_identical(positions))
        history.append(entry)
        return entry

    entry = record(0, 0, 0)
    for iteration in range(1, config.max_iterations + 1):
        if entry.positions_identical:
            break
        # Snapshot p_* before any movement or fitness evaluation. _move_swarm
        # uses the unchanged p_a array and never updates either historical best.
        global_best = personal_best[entry.best_particle_index].copy()
        positions, velocities, redraws, resets = _move_swarm(
            positions, velocities, personal_best, global_best, classes=classes,
            feature_indices=indices, n_features_in=int(n_features_in), config=config, rng=rng,
            **({"trace": trace} if diagnostics else {}))
        fitness = np.asarray([
            _search_fitness(position, X_selected, truth, classes=classes)
            for position in positions])
        evaluations += config.n_particles
        improved = fitness > personal_best_scores
        personal_best[improved] = positions[improved]
        personal_best_scores[improved] = fitness[improved]
        total_velocity_redraws += redraws
        total_particle_resets += resets
        entry = record(iteration, redraws, resets)

    best_parameters = position_to_parameters(
        personal_best[entry.best_particle_index], classes, indices, int(n_features_in))
    return PSOResult(
        SubspaceMNB.from_parameters(best_parameters), entry.best_fitness, tuple(history),
        positions, velocities, personal_best, personal_best_scores, fitness,
        "positions_identical" if entry.positions_identical else "max_iterations",
        evaluations, total_velocity_redraws, total_particle_resets,
        {"pso": int(seed)}, config, trace)
