"""Thesis PSO on the preselected subspaces; no voting or outer AFS loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .pre_pso import PrePSOConfig, run_pre_pso, save_pre_pso
from ..utils.reproducibility import peak_rss_bytes


@dataclass
class GroupModelSearch:
    results: tuple
    seeds: tuple[int, ...]
    elapsed_seconds: tuple[float, ...]


def optimize_selected_subspaces(X_search, y_search, *, classes, selected_subspaces,
                                n_features_in: int, config, seed: int) -> GroupModelSearch:
    """One fresh swarm per selected subspace, in group order, using search only.

    Child seeds come from SeedSequence(seed).spawn(G); each swarm initializes
    random priors and conditionals independently. Fitted candidate parameters
    never warm-start the swarm. Returned models are separate, not a voting model.
    """
    from ..methods.pso import optimize_mnb

    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("PSO stage seed must be an integer in [0,2**32)")
    subspaces = list(selected_subspaces)
    if not subspaces:
        raise ValueError("At least one selected subspace is required")
    if len({len(indices) for indices in subspaces}) != 1:
        raise ValueError("Selected subspaces must use the same fixed m")
    children = np.random.SeedSequence(int(seed)).spawn(len(subspaces))
    seeds = tuple(int(child.generate_state(1, dtype=np.uint32)[0]) for child in children)
    results, elapsed = [], []
    for indices, child_seed in zip(subspaces, seeds):
        start = time.perf_counter()
        results.append(optimize_mnb(X_search, y_search, classes=classes, feature_indices=indices,
                                   n_features_in=n_features_in, config=config, seed=child_seed))
        elapsed.append(time.perf_counter() - start)
    return GroupModelSearch(tuple(results), seeds, tuple(elapsed))


def run_pso_pipeline(X, y, *, pre_config: PrePSOConfig, pso_config, pso_seed: int,
                     sample_ids, source_feature_ids):
    """Reuse the verified pre-PSO stages, then optimize each retained subspace.

    Only y_search and X_search enter the optimizer. Dtest remains unscored.
    Historical one-round diagnostic. Use the revised runner for adaptive search and voting.
    """
    start = time.perf_counter()
    pre_result = run_pre_pso(X, y, config=pre_config, sample_ids=sample_ids,
                             source_feature_ids=source_feature_ids)
    plan, partitions, selection, report = pre_result
    split = plan.folds[pre_config.outer_fold_index]
    stage_start = time.perf_counter()
    search = optimize_selected_subspaces(partitions.X_search, np.asarray(y)[split.search_indices],
                                        classes=selection.classes,
                                        selected_subspaces=selection.selected_subspaces,
                                        n_features_in=partitions.X_search.shape[1],
                                        config=pso_config, seed=pso_seed)
    group_reports = []
    for index, result in enumerate(search.results):
        params = result.model.parameters_
        group_reports.append({"group_index": index, "seed": search.seeds[index],
                              "particle_dimensions": len(params.classes) * (1 + len(params.feature_indices)),
                              "initial_best_search_accuracy": result.history[0].best_fitness,
                              "best_search_accuracy": result.best_fitness,
                              "stop_reason": result.stop_reason,
                              "fitness_evaluation_count": result.fitness_evaluation_count,
                              "velocity_redraw_count": result.velocity_redraw_count,
                              "particle_reset_count": result.particle_reset_count,
                              "elapsed_seconds": search.elapsed_seconds[index],
                              "history": [asdict(entry) for entry in result.history],
                              "positive_prior": bool(np.all(params.class_prior > 0)),
                              "prior_sum": float(params.class_prior.sum()),
                              "positive_conditionals": bool(np.all(params.feature_prob > 0)),
                              "max_conditional_row_sum_error": float(np.abs(params.feature_prob.sum(axis=1) - 1).max()),
                              "selected_source_feature_ids_sha256": hashlib.sha256(
                                  partitions.source_feature_ids[params.feature_indices].astype('<i8').tobytes()).hexdigest()})
    report['purpose'] = 'development random-MNB/PSO model search; no ensemble or formal comparison'
    report['pso'] = {"config": asdict(pso_config), "fixed_inertia": .9, "root_seed": pso_seed,
                     "seed_policy": "independent SeedSequence children in selected group order",
                     "parameter_initialization": "independent open-uniform draws, block normalized; no empirical warm start",
                     "fitness": "Dsearch classification accuracy", "groups": group_reports,
                     "position_policy": "z+v; any raw coordinate outside (0,1) resets whole particle and velocity; otherwise block-normalize",
                     "scope": "one fresh PSO per selected subspace; no voting or outer weight iteration"}
    report['timings']['pso_stage_seconds'] = time.perf_counter() - stage_start
    report['timings']['pipeline_with_pso_seconds'] = time.perf_counter() - start
    report['peak_process_rss_bytes'] = peak_rss_bytes()
    return pre_result, search, report


def save_pso_pipeline(output: Path, result, *, sample_ids, sample_labels) -> None:
    pre_result, search, report = result
    save_pre_pso(output, pre_result, sample_ids=sample_ids, sample_labels=sample_labels)
    for index, pso in enumerate(search.results):
        params = pso.model.parameters_
        np.savez_compressed(output / f'group_{index:02d}_best_parameters.npz',
                            classes=params.classes, feature_indices=params.feature_indices,
                            class_prior=params.class_prior, feature_prob=params.feature_prob,
                            n_features_in=params.n_features_in)
    (output / 'run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
