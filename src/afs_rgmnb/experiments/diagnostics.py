"""Search-partition diagnostics for the revised method on fixed fit/search data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from ..methods.adaptive_sampling import update_probabilities
from ..methods.adaptive_search import search_subspaces
from ..methods.ensemble import VotingEnsemble
from ..methods.mnb import SubspaceMNB, class_statistics
from ..methods.pso import _search_fitness
from ..methods.random_generation import (
    generate_parameters,
    parameters_to_position,
    position_to_parameters,
)
from .pso_search import optimize_selected_subspaces


def digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


@dataclass
class SearchData:
    X_fit: object
    y_fit: np.ndarray
    X_search: object
    y_search: np.ndarray


def classification(truth, prediction, classes):
    matrix = confusion_matrix(truth, prediction, labels=classes)
    if matrix.sum() != len(truth) or np.any(matrix.sum(axis=1) == 0):
        raise ValueError("Expected aligned predictions and all search classes")
    correct = int(matrix.trace())
    return {
        "correct": correct,
        "n": len(truth),
        "accuracy": correct / len(truth),
        "confusion": matrix.tolist(),
        "predicted_counts": matrix.sum(axis=0).tolist(),
        "recall": (matrix.diagonal() / matrix.sum(axis=1)).tolist(),
    }


def fitted_control(data, stats, features, alpha):
    model = SubspaceMNB.from_statistics(stats, features, alpha=alpha)
    predicted = model.predict(data.X_search)
    other = MultinomialNB(alpha=alpha).fit(data.X_fit[:, features], data.y_fit)
    alternative = other.predict(data.X_search[:, features])
    metrics = classification(data.y_search, predicted, stats.classes)
    metrics["sklearn_disagreements"] = int(np.count_nonzero(alternative != predicted))
    return model, metrics


def reference_controls(data, alpha):
    stats = class_statistics(data.X_fit, data.y_fit)
    label = stats.classes[np.argmax(stats.class_counts)]
    _, full = fitted_control(data, stats, np.arange(stats.n_features), alpha)
    return {
        "classes": stats.classes.tolist(),
        "majority": classification(
            data.y_search, np.full(len(data.y_search), label), stats.classes
        ),
        "full_mnb": full,
    }


def weight_summary(q):
    return {
        "min": float(q.min()),
        "max": float(q.max()),
        "mean": float(q.mean()),
        "effective_features": float(1 / (q @ q)),
        "top_1pct_mass": float(np.sort(q)[-int(np.ceil(0.01 * len(q))) :].sum()),
    }


def coverage(matrix, features):
    nnz = np.diff(matrix[:, features].indptr)
    hist = np.bincount(nnz)
    return {
        "zeros": int(hist[0]),
        "zero_fraction": float(hist[0] / len(nnz)),
        "nnz_histogram": hist.tolist(),
        "mean_nnz": float(nnz.mean()),
    }


def feature_diagnostics(data, result, c0):
    q = result.initial_weights
    rounds = []
    for h in result.history:
        selected = [g["selected_fit_columns"] for g in h["groups"]]
        sets = [set(s) for s in selected]
        accuracies = [a for g in h["groups"] for a in g["accuracies"]]
        next_q = update_probabilities(selected, n_features=len(q), c0=c0)
        rounds.append(
            {
                "iteration": h["iteration"],
                "before": weight_summary(q),
                "after": weight_summary(next_q),
                "delta": h["delta"],
                "evaluations": len(accuracies),
                "min": min(accuracies),
                "mean": float(np.mean(accuracies)),
                "max": max(accuracies),
                "selected_accuracies": [max(g["accuracies"]) for g in h["groups"]],
                "union": len(set.union(*sets)),
                "pair_jaccard": [
                    len(a & b) / len(a | b) for a, b in combinations(sets, 2)
                ],
                "groups": h["groups"],
                "draws": h["draws"],
                "search_coverage": [
                    [coverage(data.X_search, s) for s in group] for group in h["draws"]
                ],
                "fit_coverage": [
                    [coverage(data.X_fit, s) for s in group] for group in h["draws"]
                ],
            }
        )
        if digest(q, "<f8") != h["sampling_weights_sha256"]:
            raise ValueError("Reconstructed sampling weights differ")
        q = next_q
    np.testing.assert_array_equal(q, result.next_weights)
    return rounds


def select_features(data, config, seed):
    start = perf_counter()
    result = search_subspaces(
        data.X_fit,
        data.y_fit,
        data.X_search,
        data.y_search,
        config=config,
        rng=np.random.default_rng(seed),
        retain_draws=True,
    )
    elapsed = perf_counter() - start
    return result, elapsed


def ensemble_metrics(models, data):
    vote = VotingEnsemble(models)
    predictions = [m.predict(data.X_search) for m in models]
    disagreements = [float(np.mean(a != b)) for a, b in combinations(predictions, 2)]
    metrics = classification(data.y_search, vote.predict(data.X_search), vote.classes)
    metrics["pair_disagreement"] = disagreements
    return metrics


def swarm_metrics(result, data):
    model, diag = result.model, result.diagnostics
    history = result.history
    initial = diag.initial_positions
    best = initial[history[0].best_particle_index]
    initial_model = SubspaceMNB.from_parameters(
        position_to_parameters(
            best, model.classes_, model.feature_indices_, model.n_features_in_
        )
    )
    initial_metrics = classification(
        data.y_search, initial_model.predict(data.X_search), model.classes_
    )
    final_metrics = classification(
        data.y_search, model.predict(data.X_search), model.classes_
    )
    if (
        initial_metrics["accuracy"] != history[0].best_fitness
        or final_metrics["accuracy"] != result.best_fitness
    ):
        raise ValueError("Decoded particle predictions disagree with optimizer fitness")
    particle_updates = result.iterations * len(initial)
    dim = initial.shape[1]
    valid_updates = particle_updates - result.particle_reset_count
    personal = np.array([h.personal_best_scores for h in history])
    member = {
        "seed": result.seeds["pso"],
        "initial_positions_sha256": digest(initial, "<f8"),
        "initial": initial_metrics,
        "final": final_metrics,
        "history": [asdict(h) for h in history],
        "movement_by_step": diag.steps,
        "dimension": dim,
        "particle_updates": particle_updates,
        "gain": result.best_fitness - history[0].best_fitness,
        "stop": result.stop_reason,
        "fitness_evaluations": result.fitness_evaluation_count,
        "resets": result.particle_reset_count,
        "redrawn_coords": result.velocity_redraw_count,
        "repaired_velocities": sum(s["repaired_velocities"] for s in diag.steps),
        "pbest_improvements": int(np.count_nonzero(np.diff(personal, axis=0) > 0)),
        "gbest_improvements": int(np.count_nonzero(np.diff(personal.max(axis=1)) > 0)),
        "mean_abs_change": sum(s["absolute_movement"] for s in diag.steps)
        / (dim * particle_updates)
        if particle_updates
        else None,
        "accepted_mean_abs_change": sum(
            s["accepted_absolute_movement"] for s in diag.steps
        )
        / (dim * valid_updates)
        if valid_updates
        else None,
    }
    return initial_model, member


def random_budget_control(data, result):
    """Start with the same particles, spend the remaining evaluations on independent draws."""
    model = result.model
    trace = result.diagnostics
    first = result.history[0]
    best = trace.initial_positions[first.best_particle_index]
    best_accuracy = first.best_fitness
    rng = np.random.default_rng(np.random.SeedSequence([result.seeds["pso"], 117]))
    X = data.X_search[:, model.feature_indices_]
    for _ in range(result.fitness_evaluation_count - len(trace.initial_positions)):
        params = generate_parameters(
            model.classes_, model.feature_indices_, model.n_features_in_, rng=rng
        )
        position = parameters_to_position(params)
        fitness = _search_fitness(position, X, data.y_search, classes=model.classes_)
        if fitness > best_accuracy:
            best, best_accuracy = position, fitness
    return SubspaceMNB.from_parameters(
        position_to_parameters(
            best, model.classes_, model.feature_indices_, model.n_features_in_
        )
    )


def prefix_votes(models, data, sizes):
    """Descriptive fixed-order voting controls; no member ranking or reselection."""
    if any(size < 1 or size > len(models) for size in sizes):
        raise ValueError("Invalid prefix size")
    order = models[0].classes_
    votes = np.zeros((len(data.y_search), len(order)), dtype=int)
    rows = np.arange(len(votes))
    summary = {}
    for size, model in enumerate(models, 1):
        votes[rows, np.argmax(model.predict_log_scores(data.X_search), axis=1)] += 1
        if size in sizes:
            metric = classification(data.y_search, order[votes.argmax(axis=1)], order)
            tied = (votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1
            summary[str(size)] = {"correct": metric["correct"], "ties": int(tied.sum())}
    return summary


def compare_models(
    data,
    selected,
    adaptive,
    pso,
    pso_seed,
    feature_seconds,
    *,
    random_control=False,
    prefix_sizes=(),
):
    """Run identical final subspaces through fitted controls and fresh swarms."""
    start = perf_counter()
    swarms = optimize_selected_subspaces(
        data.X_search,
        data.y_search,
        classes=selected.classes,
        selected_subspaces=selected.selected_subspaces,
        n_features_in=data.X_fit.shape[1],
        config=pso,
        seed=pso_seed,
        diagnostics=True,
    )
    end = perf_counter()
    final_models = [r.model for r in swarms.results]
    ensemble = VotingEnsemble(final_models)
    construction = perf_counter() - end
    predicted = ensemble.predict(data.X_search)
    stats = class_statistics(data.X_fit, data.y_fit)
    initial_models, fitted_models, members = [], [], []
    for indices, result in zip(selected.selected_subspaces, swarms.results):
        fitted, metrics = fitted_control(data, stats, indices, adaptive.alpha)
        initial_model, member = swarm_metrics(result, data)
        member["fitted"] = metrics
        initial_models.append(initial_model)
        fitted_models.append(fitted)
        members.append(member)
    final = classification(data.y_search, predicted, selected.classes)
    final["pair_disagreement"] = [
        float(np.mean(a != b))
        for a, b in combinations([m.predict(data.X_search) for m in final_models], 2)
    ]
    controls = {}
    if random_control:
        random_models = [
            random_budget_control(data, result) for result in swarms.results
        ]
        controls["budget_matched_random_vote"] = ensemble_metrics(random_models, data)
    if prefix_sizes:
        controls["fixed_member_prefixes"] = prefix_votes(
            final_models, data, prefix_sizes
        )
    return {
        "members": members,
        "classes": selected.classes.tolist(),
        **controls,
        "fitted_vote": ensemble_metrics(fitted_models, data),
        "initial_vote": ensemble_metrics(initial_models, data),
        "final_vote": final,
        "seconds": {
            "feature": feature_seconds,
            "pso": end - start,
            "construct": construction,
            "build": feature_seconds + (end - start) + construction,
        },
    }, ensemble
