"""Revised Chapter 3 orchestration on integer count matrices."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ..data.splitting import group_aware_splits, prepare_feature_partitions
from ..methods.adaptive_search import search_subspaces
from ..methods.ensemble import VotingEnsemble
from ..utils.reproducibility import environment_record, peak_rss_bytes
from .pso_search import optimize_selected_subspaces


def fit_ensemble(X_fit, y_fit, X_search, y_search, *, adaptive, pso, search_seed, pso_seed):
    """Build H using fit/search only. Pass final feature indices across the stage boundary."""
    start = perf_counter()
    selected = search_subspaces(X_fit, y_fit, X_search, y_search, config=adaptive,
                                rng=np.random.default_rng(search_seed))
    search_end = perf_counter()
    swarms = optimize_selected_subspaces(X_search, y_search, classes=selected.classes,
                                         selected_subspaces=selected.selected_subspaces,
                                         n_features_in=X_fit.shape[1], config=pso, seed=pso_seed)
    pso_end = perf_counter()
    ensemble = VotingEnsemble(result.model for result in swarms.results)
    end = perf_counter()
    timings = {"feature_search": search_end - start, "random_generation_and_pso": pso_end - search_end,
               "ensemble_construction": end - pso_end, "model_building": end - start}
    return ensemble, selected, swarms, timings


def evaluate_fold(X, y, split, *, adaptive, pso, seed, fold_index, source_feature_ids,
                  output: Path | None = None):
    start = perf_counter()
    mapped = prepare_feature_partitions(X, split, source_feature_ids=source_feature_ids)
    labels = np.asarray(y)
    search_seed, pso_seed = (int(child.generate_state(1)[0]) for child in
                             np.random.SeedSequence([seed, fold_index]).spawn(2))
    ensemble, selected, swarms, timings = fit_ensemble(
        mapped.X_fit, labels[split.fit_indices], mapped.X_search, labels[split.search_indices],
        adaptive=adaptive, pso=pso, search_seed=search_seed, pso_seed=pso_seed)
    prediction_start = perf_counter()
    predicted = ensemble.predict(mapped.X_test)
    timings["ensemble_prediction"] = perf_counter() - prediction_start
    correct = int(np.count_nonzero(predicted == labels[split.test_indices]))
    timings["fold_runtime"] = perf_counter() - start
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        ensemble.save(output / "ensemble.npz")
        np.savez_compressed(output / "fold.npz", fit=split.fit_indices, search=split.search_indices,
                            test=split.test_indices, fit_columns=mapped.columns,
                            source_feature_ids=mapped.source_feature_ids, predictions=predicted,
                            initial_weights=selected.initial_weights, next_weights=selected.next_weights)
    return {
        "fold": fold_index, "search_seed": search_seed, "pso_seed": pso_seed,
        "split": split.audit, "fit_features": len(mapped.columns), "classes": selected.classes.tolist(),
        "weight_iterations": len(selected.history), "stop": selected.stop_reason,
        "weight_history": selected.history,
        "candidate_evaluations": sum(len(g["accuracies"]) for h in selected.history for g in h["groups"]),
        "final_source_feature_ids": [mapped.source_feature_ids[s].tolist() for s in selected.selected_subspaces],
        "independent_searches": len(swarms.results),
        "pso": [{"seed": child, "stop": r.stop_reason, "fitness_evaluations": r.fitness_evaluation_count,
                 "best_search_accuracy": r.best_fitness, "seconds": elapsed,
                 "history": [asdict(step) for step in r.history]}
                for child, r, elapsed in zip(swarms.seeds, swarms.results, swarms.elapsed_seconds)],
        "ensemble_members": len(ensemble.members), "test_correct": correct,
        "test_samples": len(predicted), "test_accuracy": correct / len(predicted), "seconds": timings,
    }


def run_cv(X, y, *, adaptive, pso, n_splits, search_fraction, seed, sample_ids,
           source_feature_ids, folds=None, output: Path | None = None):
    """Plan all grouped folds once. An explicit fold subset yields development results."""
    start = perf_counter()
    plan = group_aware_splits(X, y, n_splits=n_splits, search_fraction=search_fraction,
                              seed=seed, sample_ids=sample_ids)
    split_seconds = perf_counter() - start
    indices = list(range(n_splits) if folds is None else folds)
    if (not indices or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                           or not 0 <= i < n_splits for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError("folds must contain distinct zero-based outer fold indices")
    records = [evaluate_fold(X, y, plan.folds[i], adaptive=adaptive, pso=pso, seed=seed, fold_index=i,
                             source_feature_ids=source_feature_ids,
                             output=None if output is None else output / f"fold_{i:02d}") for i in indices]
    accuracies = [r["test_accuracy"] for r in records]
    complete = len(records) == n_splits
    return {"environment": environment_record(), "adaptive": asdict(adaptive), "pso_config": asdict(pso),
            "split_protocol": plan.audit, "folds": records, "complete_cv": complete,
            "cv_mean_accuracy": float(np.mean(accuracies)) if complete else None,
            "cv_std_accuracy_ddof0": float(np.std(accuracies)) if complete else None,
            "split_seconds": split_seconds, "cv_runtime_seconds": perf_counter() - start,
            "peak_process_rss_bytes": peak_rss_bytes()}
