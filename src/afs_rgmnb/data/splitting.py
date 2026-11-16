"""Duplicate-aware, approximately stratified fit/search/test partitions.

Groups describe exact count vectors, not raw document or biological identity.
All rows, including conflicting-label vectors, remain in one partition each.
Group-aware splitting is an explicit evaluation protocol around Chapter 3's
fit/search/test roles; it does not change the classifier or selection objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class GroupSplit:
    fit_indices: np.ndarray
    search_indices: np.ndarray
    test_indices: np.ndarray
    audit: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SplitPlan:
    groups: np.ndarray
    classes: np.ndarray
    sample_ids: np.ndarray
    folds: list[GroupSplit]
    audit: dict


@dataclass(frozen=True)
class FeaturePartitions:
    X_fit: sparse.csr_matrix
    X_search: sparse.csr_matrix
    X_test: sparse.csr_matrix
    columns: np.ndarray
    source_feature_ids: np.ndarray


def _canonical_counts(X) -> sparse.csr_matrix:
    """Canonical int64 CSR without changing the caller's matrix.

    Explicit zeros and repeated/unsorted storage entries must not create false
    identities. Counts outside signed int64 cannot be hashed by this protocol.
    """
    if sparse.issparse(X) and X.format != "csr":
        # Construct CSR storage without arithmetic reduction. Ordinary COO->CSR
        # conversion can hide invalid negative entries by cancellation, or wrap
        # overflowing duplicates before our exact-integer validation sees them.
        coo = X.tocoo(copy=True)
        order = np.argsort(coo.row, kind="stable")
        indptr = np.concatenate(([0], np.cumsum(np.bincount(coo.row, minlength=coo.shape[0]))))
        raw = sparse.csr_matrix((coo.data[order], coo.col[order], indptr), shape=coo.shape)
    else:
        raw = sparse.csr_matrix(X, copy=True)
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("A nonempty sample/feature matrix is required")
    data = raw.data
    if (not np.isrealobj(data) or not np.all(np.isfinite(data))
            or np.any(data < 0) or np.any(data != np.floor(data))):
        raise ValueError("Count vectors must contain finite nonnegative integer values")
    if data.dtype.kind in "fc":
        invalid_range = np.any(data >= 2**63)
    else:
        invalid_range = np.any(data > np.iinfo(np.int64).max)
    if invalid_range:
        raise ValueError("Count values exceed the supported int64 range")
    raw = raw.astype(np.int64)
    if not raw.has_canonical_format:
        # Python-integer accumulation prevents duplicate-entry overflow before
        # canonicalization. This branch is unnecessary for normal archive CSR.
        indices, values, indptr = [], [], [0]
        for row in range(raw.shape[0]):
            counts = {}
            start, end = raw.indptr[row:row + 2]
            for index, value in zip(raw.indices[start:end], raw.data[start:end]):
                index = int(index)
                counts[index] = counts.get(index, 0) + int(value)
                if counts[index] > np.iinfo(np.int64).max:
                    raise ValueError("Combined count values exceed the supported int64 range")
            for index, value in sorted(counts.items()):
                if value:
                    indices.append(index)
                    values.append(value)
            indptr.append(len(indices))
        raw = sparse.csr_matrix((np.asarray(values, dtype=np.int64),
                                 np.asarray(indices, dtype=np.int64),
                                 np.asarray(indptr, dtype=np.int64)), shape=raw.shape)
    raw.eliminate_zeros()
    raw.sort_indices()
    return raw


def _hash_vectors(X: sparse.csr_matrix) -> np.ndarray:
    groups = []
    prefix = b"afs-count-vector-v1\0" + X.shape[1].to_bytes(8, "big")
    for row in range(X.shape[0]):
        start, end = X.indptr[row:row + 2]
        digest = hashlib.sha256(prefix)
        digest.update(X.indices[start:end].astype("<i8", copy=False).tobytes())
        digest.update(X.data[start:end].astype("<i8", copy=False).tobytes())
        groups.append(digest.hexdigest())
    return np.asarray(groups)


def group_count_vectors(X) -> np.ndarray:
    """Return stable per-row SHA-256 identity groups for canonical count vectors.

    Hashing includes the feature-space dimension and fixed-endian int64 indices
    and counts. Dtype, explicit-zero and CSR storage ordering differences do not
    affect the identity. Labels are deliberately absent from the hash.
    """
    return _hash_vectors(_canonical_counts(X))


def _class_labels(values, *, name):
    # Validate scalar types before NumPy could turn numeric labels into strings.
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not raw.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional label array")
    kinds = set()
    for label in raw:
        if isinstance(label, (str, np.str_)):
            kinds.add("string")
            if not label.strip():
                raise ValueError(f"{name} contains an empty class label")
        elif isinstance(label, (int, float, bool, np.integer, np.floating, np.bool_)):
            kinds.add("numeric")
            if not np.isfinite(label):
                raise ValueError(f"{name} contains a nonfinite class label")
        else:
            raise ValueError(f"{name} labels must be finite numbers or nonempty strings")
    if len(kinds) != 1:
        raise ValueError(f"{name} cannot mix numeric and string labels")
    return np.asarray(values)


def _labels(y, n_samples: int, class_order=None):
    labels = _class_labels(y, name="y")
    if len(labels) != n_samples:
        raise ValueError("Labels must be aligned with samples")
    try:
        classes = np.unique(labels)
    except TypeError as exc:
        raise ValueError("Class labels must use a consistent sortable type") from exc
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    if class_order is not None:
        order = _class_labels(class_order, name="class_order")
        if (order.ndim != 1 or len(order) != len(classes)
                or len(set(order.tolist())) != len(order)
                or set(order.tolist()) != set(classes.tolist())):
            raise ValueError("class_order must list every observed class exactly once")
        classes = order.copy()
    return labels, classes


def _hist(labels, classes) -> np.ndarray:
    return np.asarray([np.count_nonzero(labels == cls) for cls in classes], dtype=np.int64)


def _index_digest(indices) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _id_digest(ids) -> str:
    payload = json.dumps(np.asarray(ids).tolist(), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_partition_indices(n_samples, fit, search, test):
    roles = {"fit": fit, "search": search, "test": test}
    for role, indices in roles.items():
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= n_samples)
                or np.unique(indices).size != indices.size):
            raise ValueError(f"Invalid {role} sample indices")
    combined = np.concatenate(list(roles.values()))
    if len(combined) != n_samples or np.unique(combined).size != n_samples:
        raise ValueError("Partitions must cover all samples once, with no index overlap")


def group_aware_splits(X, y, *, n_splits: int = 5, search_fraction: float = .2,
                       seed: int = 42, sample_ids=None, class_order=None) -> SplitPlan:
    """Create outer SGKF folds and nested group-disjoint fit/search partitions.

    Nested SGKF uses k=round(1/min(f,1-f)), with k>=2. Among its feasible folds,
    select the first minimizing size error plus mean absolute fit/search class-
    proportion deviations from the outer training labels. This deterministic
    allocation objective uses no model scores or outer test measurements.
    Exact stratification is not promised; actual deviations are recorded.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer at least two")
    if not np.isfinite(search_fraction) or not 0 < search_fraction < 1:
        raise ValueError("search_fraction must be in (0,1)")
    if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**32):
        raise ValueError("seed must be an integer in [0,2**32)")
    matrix = _canonical_counts(X)
    labels, classes = _labels(y, matrix.shape[0], class_order)
    if sample_ids is None:
        identifiers = np.asarray([f"row_{i:09d}" for i in range(matrix.shape[0])])
    else:
        identifiers = np.asarray(sample_ids)
        if (identifiers.ndim != 1 or len(identifiers) != matrix.shape[0]
                or any(value is None or not str(value).strip() for value in identifiers)
                or len(set(identifiers.tolist())) != len(identifiers)):
            raise ValueError("Sample IDs must be unique, nonempty and aligned with samples")
        identifiers = identifiers.copy()
    groups = _hash_vectors(matrix)
    unique_groups, group_sizes = np.unique(groups, return_counts=True)
    group_label_counts = {key: set() for key in unique_groups}
    for group, label in zip(groups, labels):
        group_label_counts[group].add(label.item() if hasattr(label, "item") else label)
    conflict_groups = {key for key, values in group_label_counts.items() if len(values) > 1}
    if len(unique_groups) < n_splits:
        raise ValueError("Too few identity groups for requested outer folds; no records were removed")
    for cls in classes:
        if len(np.unique(groups[labels == cls])) < n_splits:
            raise ValueError(f"Class {cls!r} has fewer than {n_splits} distinct identity groups; "
                             "all-class outer coverage is infeasible; no classes were removed")
    inner_splits = max(2, int(np.floor(1 / min(search_fraction, 1 - search_fraction) + .5)))
    outer = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    outer_test_coverage = np.zeros(matrix.shape[0], dtype=np.int64)
    full_proportions = _hist(labels, classes) / len(labels)
    folds = []
    for fold_index, (train, test) in enumerate(outer.split(matrix, labels, groups)):
        if np.any(_hist(labels[test], classes) == 0):
            raise ValueError(f"Outer fold {fold_index} lacks required test classes; no classes were removed")
        if len(np.unique(groups[train])) < inner_splits:
            raise ValueError("Too few training groups for the requested nested search fraction")
        inner_seed = (int(seed) + fold_index + 1) % 2**32
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=inner_seed)
        training_proportions = _hist(labels[train], classes) / len(train)
        candidates = []
        feasible = []
        for inner_index, (remainder, heldout) in enumerate(inner.split(matrix[train], labels[train], groups[train])):
            fit_local, search_local = (remainder, heldout) if search_fraction <= .5 else (heldout, remainder)
            fit, search = train[fit_local], train[search_local]
            fit_counts, search_counts = _hist(labels[fit], classes), _hist(labels[search], classes)
            covered = bool(np.all(fit_counts > 0) and np.all(search_counts > 0))
            objective = (abs(len(search) / len(train) - search_fraction)
                         + np.abs(fit_counts / len(fit) - training_proportions).mean()
                         + np.abs(search_counts / len(search) - training_proportions).mean())
            candidates.append({"index": inner_index, "all_classes_present": covered,
                               "allocation_objective": float(objective),
                               "fit_samples": len(fit), "search_samples": len(search)})
            if covered:
                feasible.append((float(objective), inner_index, fit, search))
        if not feasible:
            raise ValueError(f"Outer fold {fold_index} has no nested group split covering all classes; "
                             "no class was removed. Revise the documented split settings.")
        _, chosen_inner, fit, search = min(feasible, key=lambda item: (item[0], item[1]))
        _validate_partition_indices(len(labels), fit, search, test)
        roles = {"fit": fit, "search": search, "test": test}
        sample_overlaps, group_overlaps = {}, {}
        for first, second in [("fit", "search"), ("fit", "test"), ("search", "test")]:
            key = first + "_" + second
            sample_overlaps[key] = int(np.intersect1d(roles[first], roles[second]).size)
            group_overlaps[key] = int(np.intersect1d(groups[roles[first]], groups[roles[second]]).size)
        if any(sample_overlaps.values()) or any(group_overlaps.values()):
            raise RuntimeError("Internal split error: sample or identity-group overlap")
        role_audits = {}
        for role, indices in roles.items():
            counts = _hist(labels[indices], classes)
            proportions = counts / len(indices)
            role_audits[role] = {
                "samples": len(indices), "class_count": int(np.count_nonzero(counts)),
                "group_count": len(np.unique(groups[indices])),
                "class_distribution": {str(cls): int(count) for cls, count in zip(classes, counts)},
                "class_proportions": {str(cls): float(value) for cls, value in zip(classes, proportions)},
                "class_proportion_deviation_from_full": {
                    str(cls): float(value) for cls, value in zip(classes, proportions - full_proportions)},
                "max_abs_class_proportion_deviation_from_full": float(np.abs(proportions - full_proportions).max()),
                "fraction_of_all_samples": len(indices) / len(labels),
                "sample_indices_sha256": _index_digest(indices),
                "sample_ids_sha256": _id_digest(identifiers[indices]),
                "conflicting_label_group_count": len(set(groups[indices]) & conflict_groups),
            }
        outer_test_coverage[test] += 1
        audit = {"outer_fold": fold_index, "outer_seed": int(seed), "inner_seed": inner_seed,
                 "inner_n_splits": inner_splits, "selected_inner_fold": chosen_inner,
                 "inner_allocation_candidates": candidates,
                 "requested_search_fraction_of_train": float(search_fraction),
                 "realized_search_fraction_of_train": len(search) / len(train),
                 "roles": role_audits, "sample_overlaps": sample_overlaps,
                 "group_overlaps": group_overlaps, "all_classes_present": True,
                 "all_records_preserved": True}
        folds.append(GroupSplit(fit.copy(), search.copy(), test.copy(), audit))
    if not np.all(outer_test_coverage == 1):
        raise RuntimeError("Internal split error: outer test partitions do not cover every row once")
    audit = {
        "protocol": "StratifiedGroupKFold outer and nested allocation-only SGKF selection",
        "stratification": "approximate; observed deviations recorded per partition",
        "identity_definition": "SHA-256 canonical sparse count vector v1; labels excluded",
        "seed": int(seed), "outer_n_splits": int(n_splits),
        "requested_search_fraction": float(search_fraction),
        "inner_selection_objective": "absolute search-size error + mean absolute fit/search class-proportion deviations from outer train; ties first fold",
        "samples": len(labels), "classes": classes.tolist(),
        "class_distribution": {str(cls): int(count) for cls, count in zip(classes, _hist(labels, classes))},
        "group_count": len(unique_groups),
        "duplicate_groups": int(np.count_nonzero(group_sizes > 1)),
        "duplicate_rows_including_first": int(group_sizes[group_sizes > 1].sum()),
        "duplicate_excess_rows": int((group_sizes - 1).sum()),
        "conflicting_label_groups": len(conflict_groups),
        "conflicting_label_rows": int(np.isin(groups, list(conflict_groups)).sum()),
        "sample_ids_sha256": _id_digest(identifiers),
        "group_ids_sha256": hashlib.sha256("\n".join(groups).encode("ascii")).hexdigest(),
        "all_records_preserved": True, "outer_test_coverage_once": True,
        "limitations": ["Exact count-vector groups do not establish raw document identity or near-duplicate independence.",
                        "Historical preprocessing/vocabulary construction may contain unrecoverable upstream information; fit-only masks cannot undo it."],
    }
    return SplitPlan(groups, classes, identifiers, folds, audit)


def prepare_feature_partitions(X, split: GroupSplit, *, source_feature_ids=None) -> FeaturePartitions:
    """Learn observed columns from fit rows only; transform all roles identically.

    ``columns`` maps transformed columns back to the input matrix; explicit
    ``source_feature_ids`` preserves one-based document IDs or other dictionaries.
    Search and test rows never add vocabulary columns.
    """
    matrix = _canonical_counts(X)
    _validate_partition_indices(matrix.shape[0], split.fit_indices, split.search_indices, split.test_indices)
    if source_feature_ids is None:
        source_ids = np.arange(matrix.shape[1], dtype=np.int64)
    else:
        source_ids = np.asarray(source_feature_ids)
        if (source_ids.ndim != 1 or len(source_ids) != matrix.shape[1]
                or len(set(source_ids.tolist())) != len(source_ids)):
            raise ValueError("source_feature_ids must uniquely identify every input feature column")
    fit = matrix[split.fit_indices]
    columns = np.unique(fit.indices)
    if not len(columns):
        raise ValueError("No observed features in fit; search/test cannot supply a vocabulary")
    return FeaturePartitions(fit[:, columns].tocsr(), matrix[split.search_indices][:, columns].tocsr(),
                             matrix[split.test_indices][:, columns].tocsr(), columns.copy(), source_ids[columns].copy())
