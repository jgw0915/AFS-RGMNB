"""Reproducible pre-PSO pipeline; test data are partitioned, never scored."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..data.splitting import group_aware_splits, prepare_feature_partitions
from ..methods.random_subspace import select_subspaces
from ..utils.reproducibility import environment_record, peak_rss_bytes


@dataclass(frozen=True)
class PrePSOConfig:
    """Caller-specified development settings, not finalized thesis parameters."""
    n_groups: int
    candidates_per_group: int
    subspace_size: int
    alpha: float
    epsilon: float
    seed: int
    outer_folds: int
    search_fraction: float
    outer_fold_index: int

    def __post_init__(self):
        for name, value, minimum in [("n_groups", self.n_groups, 1),
                                     ("candidates_per_group", self.candidates_per_group, 1),
                                     ("subspace_size", self.subspace_size, 2),
                                     ("outer_folds", self.outer_folds, 2),
                                     ("seed", self.seed, 0), ("outer_fold_index", self.outer_fold_index, 0)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed >= 2**32 or self.outer_fold_index >= self.outer_folds:
            raise ValueError("seed must be < 2**32 and outer_fold_index < outer_folds")
        for name, value in [("alpha", self.alpha), ("epsilon", self.epsilon),
                            ("search_fraction", self.search_fraction)]:
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive scalar")
        if self.epsilon >= 1 or self.search_fraction >= 1:
            raise ValueError("epsilon and search_fraction must be in (0,1)")


def run_pre_pso(X, y, *, config: PrePSOConfig, sample_ids, source_feature_ids):
    """Return reusable result objects and a JSON-compatible development audit.

    All rows are retained. Hashing full count vectors is used only to protect
    partition independence. Vocabulary uses fit rows; JMH and fitted parameters
    use fit; selection uses search. There is no test accuracy computation.
    """
    start = time.perf_counter()
    environment = environment_record()
    if (isinstance(config.outer_fold_index, bool)
            or not isinstance(config.outer_fold_index, (int, np.integer))
            or not 0 <= config.outer_fold_index < config.outer_folds):
        raise ValueError("outer_fold_index must be an integer within the configured outer folds")
    y = np.asarray(y)
    plan = group_aware_splits(X, y, n_splits=config.outer_folds,
                             search_fraction=config.search_fraction, seed=config.seed,
                             sample_ids=sample_ids)
    split_seconds = time.perf_counter() - start
    split = plan.folds[config.outer_fold_index]
    feature_start = time.perf_counter()
    partitions = prepare_feature_partitions(X, split, source_feature_ids=source_feature_ids)
    feature_seconds = time.perf_counter() - feature_start
    selection = select_subspaces(partitions.X_fit, y[split.fit_indices],
                                partitions.X_search, y[split.search_indices],
                                n_groups=config.n_groups, candidates_per_group=config.candidates_per_group,
                                subspace_size=config.subspace_size, alpha=config.alpha,
                                epsilon=config.epsilon, seed=config.seed, class_order=plan.classes)
    groups = []
    for index, group in enumerate(selection.groups):
        candidates = [{"candidate_index": i, "accuracy": c.accuracy,
                       "correct_predictions": c.correct_predictions,
                       "search_samples": c.search_samples,
                       "local_feature_indices": c.feature_indices.tolist(),
                       "source_feature_ids": partitions.source_feature_ids[c.feature_indices].tolist()}
                      for i, c in enumerate(group.candidates)]
        groups.append({"group_index": index, "selected_index": group.selected_index,
                       "tied_best_indices": list(group.tied_best_indices), "candidates": candidates})
    report = {"schema_version": 1, "purpose": "development pre-PSO smoke test; not formal model comparison",
              "config": asdict(config), "environment": environment,
              "dataset_shape": list(X.shape), "dataset_nnz": int(X.nnz),
              "class_order": plan.classes.tolist(),
              "split_plan": plan.audit, "selected_fold": split.audit,
              "outer_fold_audits": [fold.audit for fold in plan.folds],
              "fit_observed_features": len(partitions.columns),
              "vocabulary_fit_scope": "Dfit only", "test_evaluated": False,
              "jmh": {"finite": bool(np.isfinite(selection.feature_scores).all()),
                      "minimum": float(selection.feature_scores.min()),
                      "maximum": float(selection.feature_scores.max()),
                      "sum": float(selection.feature_scores.sum())},
              "weights": {"finite": bool(np.isfinite(selection.sampling_weights).all()),
                          "minimum": float(selection.sampling_weights.min()),
                          "sum": float(selection.sampling_weights.sum()),
                          "sha256": hashlib.sha256(selection.sampling_weights.astype('<f8').tobytes()).hexdigest()},
              "groups": groups,
              "timings": {"grouping_all_splits_seconds": split_seconds,
                          "fit_only_feature_mapping_seconds": feature_seconds,
                          **selection.timings, "pipeline_seconds": time.perf_counter() - start},
              "peak_process_rss_bytes": peak_rss_bytes(),
              "memory_scope": "process lifetime peak incl imports, source matrix and retained split matrices",
              "limitations": ["Preserved historical preprocessing/dictionary cannot be reconstructed or purged of upstream leakage.",
                              "Exact count-vector grouping does not identify near duplicates, shared threads/authors or semantic duplicates.",
                              "Cross-label identical vectors remain together without repair or deletion.",
                              "Approximate stratification is audited; development settings are not finalized thesis parameters."]}
    return plan, partitions, selection, report


def save_pre_pso(output: Path, result, *, sample_ids, sample_labels) -> None:
    """Save private research outputs; callers must use an ignored output directory."""
    plan, partitions, selection, report = result
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fit_source_feature_ids.npy", partitions.source_feature_ids)
    np.save(output / "jmh_scores.npy", selection.feature_scores)
    np.save(output / "sampling_weights.npy", selection.sampling_weights)
    split_arrays = {}
    for fold_index, split in enumerate(plan.folds):
        for role in ("fit", "search", "test"):
            split_arrays[f"fold_{fold_index}_{role}"] = getattr(split, f"{role}_indices")
    np.savez_compressed(output / "split_indices.npz", **split_arrays)
    selected = plan.folds[report["config"]["outer_fold_index"]]
    with (output / "selected_fold_records.tsv").open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(["row_index", "archive_sample_id", "original_class_label", "count_vector_group", "role"])
        for role in ("fit", "search", "test"):
            for row in getattr(selected, f"{role}_indices"):
                writer.writerow([int(row), sample_ids[row], sample_labels[row], plan.groups[row], role])
    (output / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
