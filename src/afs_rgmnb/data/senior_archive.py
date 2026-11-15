"""Read the *preserved count representations* in the academic-year 107 archive.

These readers do not reconstruct absent raw DNA or document preprocessing. They
preserve multiclass labels, repeated occurrence counts, and source feature IDs.
Historical binary conversions, feature groups and class-level statistics are
deliberately not inputs. See docs/references/senior_feature_extraction_comparison.md.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class ArchiveCounts:
    X: sparse.csr_matrix
    y: np.ndarray
    identifiers: list[str]
    label_names: dict[int, str]
    source_fingerprint: str
    source_paths: list[str]
    historical_folds: np.ndarray | None = None


def _add_fingerprint(digest, relative_path: str, payload: bytes) -> None:
    """Unambiguous framed path + byte SHA-256; independent of parent location."""
    name = relative_path.encode("utf-8")
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(hashlib.sha256(payload).digest())


def _append_row(pairs, indices, values, indptr, n_features: int, location: str):
    seen = set()
    row = []
    for feature_id, count in pairs:
        if not 0 <= feature_id < n_features:
            raise ValueError(f"{location}: feature ID {feature_id} outside [0, {n_features})")
        if count <= 0 or count > np.iinfo(np.int64).max:
            raise ValueError(f"{location}: occurrence count must be a positive int64")
        if feature_id in seen:
            raise ValueError(f"{location}: repeated feature ID {feature_id}")
        seen.add(feature_id)
        row.append((feature_id, count))
    for feature_id, count in sorted(row):
        indices.append(feature_id)
        values.append(count)
    indptr.append(len(values))


def _matrix(indices, values, indptr, n_features: int):
    return sparse.csr_matrix(
        (np.frombuffer(values, dtype=np.int64),
         np.frombuffer(indices, dtype=np.int32),
         np.frombuffer(indptr, dtype=np.int64)),
        shape=(len(indptr) - 1, n_features), dtype=np.int64,
    )


def read_gene_instances(directory: str | Path, n_features: int = 65536) -> ArchiveCounts:
    """Read class folders of ASCII ``feature count`` rows, without relabeling.

    ClassName_*.txt is decoded strictly as CP950, as verified in this archive.
    Numeric directory/file names supply stable *archive* identifiers, not the
    absent biological accession numbers. Taxonomic ranks are not inferred.
    """
    directory = Path(directory)
    metadata_files = sorted(directory.glob("ClassName_*.txt"))
    if len(metadata_files) != 1:
        raise ValueError("Expected exactly one ClassName_*.txt in instance directory")
    digest = hashlib.sha256()
    metadata_path = metadata_files[0]
    payload = metadata_path.read_bytes()
    _add_fingerprint(digest, metadata_path.name, payload)
    names = {}
    for line_no, line in enumerate(payload.decode("cp950").splitlines()[1:], 2):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{metadata_path}:{line_no}: missing taxon name")
        class_id = int(fields[0])
        if class_id in names:
            raise ValueError(f"{metadata_path}:{line_no}: repeated class code")
        names[class_id] = fields[1]
    folders = [p for p in directory.iterdir() if p.is_dir()]
    if not folders or any(not p.name.isdecimal() for p in folders):
        raise ValueError("Expected nonempty numeric class folders only")
    folders.sort(key=lambda p: int(p.name))
    indices, values, indptr = array("i"), array("q"), array("q", [0])
    labels, identifiers, sources = [], [], [metadata_path.name]
    for folder in folders:
        class_id = int(folder.name)
        if class_id not in names:
            raise ValueError(f"Class {class_id} absent from class-name metadata")
        files = list(folder.glob("*.txt"))
        if not files or any(not p.stem.isdecimal() for p in files):
            raise ValueError(f"{folder}: expected numeric instance filenames")
        files.sort(key=lambda p: int(p.stem))
        for path in files:
            relative = path.relative_to(directory).as_posix()
            payload = path.read_bytes()
            _add_fingerprint(digest, relative, payload)
            pairs = []
            for line_no, line in enumerate(payload.decode("ascii").splitlines(), 1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_no}: blank feature row")
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"{path}:{line_no}: expected feature ID and count")
                pairs.append((int(fields[0]), int(fields[1])))
            _append_row(pairs, indices, values, indptr, n_features, str(path))
            labels.append(class_id)
            identifiers.append(relative)
            sources.append(relative)
    if set(labels) != set(names):
        raise ValueError("Class metadata and nonempty class folders do not match")
    return ArchiveCounts(_matrix(indices, values, indptr, n_features),
                         np.asarray(labels, dtype=np.int32), identifiers, names,
                         digest.hexdigest(), sources)


def read_document_counts(path: str | Path, n_features: int = 83635) -> ArchiveCounts:
    """Read ASCII ``class,fold,feature,count,...`` with one-based feature IDs.

    Historical fold numbers are retained as metadata but never automatically
    reused for an experiment. Row numbers are archive IDs, not news Message-IDs.
    """
    path = Path(path)
    indices, values, indptr = array("i"), array("q"), array("q", [0])
    labels, folds, identifiers = [], [], []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_no, payload in enumerate(stream, 1):
            digest.update(payload)
            fields = payload.decode("ascii").strip().split(",")
            if len(fields) < 2 or len(fields) % 2:
                raise ValueError(f"{path}:{line_no}: expected class, fold and feature/count pairs")
            try:
                row = [int(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-integer field") from exc
            if row[0] < 1 or row[1] < 1:
                raise ValueError(f"{path}:{line_no}: class and historical fold must be positive")
            _append_row(zip((x - 1 for x in row[2::2]), row[3::2]),
                        indices, values, indptr, n_features, f"{path}:{line_no}")
            labels.append(row[0])
            folds.append(row[1])
            identifiers.append(f"row_{line_no:06d}")
    if not labels:
        raise ValueError(f"{path}: empty dataset")
    return ArchiveCounts(_matrix(indices, values, indptr, n_features),
                         np.asarray(labels, dtype=np.int32), identifiers, {},
                         digest.hexdigest(), [path.name], np.asarray(folds, dtype=np.int32))


def count_vector_duplicates(dataset: ArchiveCounts) -> dict:
    """Audit identical count vectors, not identical underlying sequences/text."""
    groups = defaultdict(list)
    X = dataset.X
    for i in range(X.shape[0]):
        start, end = X.indptr[i:i + 2]
        digest = hashlib.sha256()
        digest.update(X.indices[start:end].astype("<i4", copy=False).tobytes())
        digest.update(X.data[start:end].astype("<i8", copy=False).tobytes())
        groups[digest.digest()].append(i)
    repeated = [g for g in groups.values() if len(g) > 1]
    conflicts = [g for g in repeated if len(set(dataset.y[g].tolist())) > 1]
    return {
        "definition": "identical sparse count vectors; raw sequence/text identity unavailable",
        "duplicate_groups": len(repeated),
        "duplicate_rows_including_first": sum(map(len, repeated)),
        "duplicate_excess_rows": sum(len(g) - 1 for g in repeated),
        "conflicting_label_groups": len(conflicts),
        "conflicting_label_rows": sum(map(len, conflicts)),
    }


def summarize_counts(dataset: ArchiveCounts) -> dict:
    X = dataset.X
    distribution = Counter(map(int, dataset.y))
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    return {
        "shape": list(X.shape), "samples": X.shape[0], "classes": len(distribution),
        "class_distribution": {str(k): v for k, v in sorted(distribution.items())},
        "class_size_min": min(distribution.values()), "class_size_max": max(distribution.values()),
        "class_size_median": float(np.median(list(distribution.values()))),
        "classes_with_at_least_5_samples": sum(v >= 5 for v in distribution.values()),
        "classes_with_at_least_10_samples": sum(v >= 10 for v in distribution.values()),
        "declared_feature_dimensions": X.shape[1],
        "observed_feature_dimensions": int(np.count_nonzero(X.getnnz(axis=0))),
        "nnz": int(X.nnz), "density": X.nnz / (X.shape[0] * X.shape[1]),
        "sparsity": 1 - X.nnz / (X.shape[0] * X.shape[1]),
        "csr_bytes": int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes),
        "dtype": str(X.dtype), "minimum_stored_count": int(X.data.min()) if X.nnz else None,
        "maximum_stored_count": int(X.data.max()) if X.nnz else None,
        "total_occurrences": int(row_sums.sum()),
        "row_occurrence_sum_min": int(row_sums.min()),
        "row_occurrence_sum_max": int(row_sums.max()),
        "zero_feature_rows": int(np.count_nonzero(np.diff(X.indptr) == 0)),
        "invalid_records": 0, "invalid_policy": "fail immediately; no record is silently discarded",
        "duplicate_identifiers": len(dataset.identifiers) - len(set(dataset.identifiers)),
        "duplicates": count_vector_duplicates(dataset),
        "source_fingerprint_sha256": dataset.source_fingerprint,
        "source_file_count": len(dataset.source_paths),
    }


def fit_observed_columns(X_fit: sparse.csr_matrix) -> np.ndarray:
    """Training-only observed vocabulary mask for preserved source-ID columns."""
    return np.flatnonzero(np.asarray(X_fit.getnnz(axis=0)).ravel()).astype(np.int32)


def discover_archive_sources(root: str | Path) -> dict[str, Path]:
    """Discover verified source names without assuming the archive's folder name."""
    root = Path(root)
    candidates = {
        "bacteria": list(root.glob("Program/*/GenePreprocess/GenePreprocess/bin/Debug/bac_instance")),
        "fungi": list(root.glob("Program/*/GenePreprocess/GenePreprocess/bin/Debug/fun_instance")),
        "20news": list(root.glob("Program/*/text/20Newsgroup/20newsOutput.txt")),
    }
    for key, paths in candidates.items():
        if len(paths) != 1:
            raise ValueError(f"{key}: expected exactly one preserved count source, found {len(paths)}")
    return {key: paths[0] for key, paths in candidates.items()}
