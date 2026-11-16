"""Audit all preserved multiclass count datasets without executing legacy code.

Outputs contain no raw records; full matrices can optionally be exported into
the output directory (which must be outside the read-only archive). This is a
conversion/vocabulary audit, not an accuracy experiment or a reproduction.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afs_rgmnb.data.senior_archive import (discover_archive_sources,
    fit_observed_columns, read_document_counts, read_gene_instances, summarize_counts)


def peak_rss_bytes() -> int | None:
    """Process lifetime peak working set, including imported/native libraries."""
    if sys.platform == "win32":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess
        handle.restype = ctypes.c_void_p
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
        if memory(handle(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except ImportError:
        return None


def vocabulary_audit(dataset, seed: int) -> dict:
    """No scoring, fitting, tuning, accuracy computation, or minority filtering."""
    X, y = dataset.X, dataset.y
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 3:
        # A genuine three-way class-preserving split is mathematically impossible.
        # This two-way operation tests sparse vocabulary transformation only.
        fit, test = next(StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y))
        roles = {"fit": fit, "heldout": test}
        note = ("Two-way stratified 50/50 conversion audit only. Full multiclass data contain "
                "classes of size 2; class-preserving Dfit/Dsearch/Dtest is impossible. "
                "No search split or thesis performance result is produced.")
        protocol = "extraction_audit_only_two_way"
    else:
        train, test = next(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        fit, search = train_test_split(train, test_size=0.2, random_state=seed,
                                       stratify=y[train])
        roles = {"fit": fit, "search": search, "test": test}
        note = "One 5-fold outer split and 20% inner search split; audit settings, not approved final settings."
        protocol = "three_way_conversion_audit"
    columns = fit_observed_columns(X[fit])
    role_results = {}
    for role, rows in roles.items():
        transformed = X[rows][:, columns].tocsr()
        role_results[role] = {
            "shape": list(transformed.shape), "nnz": int(transformed.nnz),
            "zero_feature_rows": int(np.count_nonzero(np.diff(transformed.indptr) == 0)),
            "class_count": int(np.unique(y[rows]).size),
            "row_indices_sha256": hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest(),
        }
    assert len(set(np.concatenate(list(roles.values())).tolist())) == X.shape[0]
    assert all(np.intersect1d(a, b).size == 0 for i, a in enumerate(roles.values())
               for j, b in enumerate(roles.values()) if i < j)
    assert len(columns) == int(np.count_nonzero(X[fit].getnnz(axis=0)))
    return {"protocol": protocol, "note": note, "seed": seed,
            "vocabulary_fit_scope": "fit rows only", "observed_fit_features": len(columns),
            "roles": role_results,
            "duplicate_policy": "retained for full-source audit; not safe performance splits"}


def audit_one(root: Path, key: str, output: Path, seed: int, export: bool) -> dict:
    start = time.perf_counter()
    sources = discover_archive_sources(root, include_optional=key == "mdr88")
    source = sources[key]
    is_document = key in {"20news", "mdr88"}
    dataset = read_document_counts(source, n_features=12511 if key == "mdr88" else 83635) if is_document else read_gene_instances(source)
    load_seconds = time.perf_counter() - start
    report = summarize_counts(dataset)
    report.update({"dataset": key, "source": source.relative_to(root).as_posix(),
                   "representation": "preserved precomputed occurrence counts",
                   "label_policy": "preserve all original multiclass labels; no binary remap or resampling",
                   "raw_sequence_or_text_available": (source.parent / "MDR88.txt").is_file() if key == "mdr88" else False,
                   "original_accession_or_message_ids_available": False,
                   "taxonomic_hierarchy_available": None if is_document else False,
                   "biological_rank": None if is_document else "unverified; metadata provide names only",
                   "sequence_lengths": None,
                   "invalid_or_ambiguous_nucleotides": None,
                   "near_duplicate_sequences_or_text": "not assessable without raw data",
                   "label_names": {str(k): v for k, v in dataset.label_names.items()},
                   "load_seconds": load_seconds})
    if dataset.historical_folds is not None:
        values, counts = np.unique(dataset.historical_folds, return_counts=True)
        report["historical_fold_distribution_not_reused"] = dict(zip(map(str, values), map(int, counts)))
    report["vocabulary_audit"] = vocabulary_audit(dataset, seed)
    report["total_audit_seconds"] = time.perf_counter() - start
    report["process_lifetime_peak_rss_bytes"] = peak_rss_bytes()
    report["memory_scope"] = "fresh dataset worker; process lifetime peak including native libraries"
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["numpy"] = np.__version__
    report["scipy"] = __import__("scipy").__version__
    report["scikit_learn"] = __import__("sklearn").__version__
    inspected_at = datetime.now(timezone.utc)
    report["inspection_date"] = inspected_at.date().isoformat()
    report["inspection_timestamp_utc"] = inspected_at.isoformat()
    repo = Path(__file__).resolve().parents[1]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                         capture_output=True, text=True, check=False)
    report["source_code_commit"] = git.stdout.strip() if git.returncode == 0 else None
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                                capture_output=True, text=True, check=False)
    report["source_code_worktree_dirty"] = bool(git_status.stdout.strip())
    report["validation_code_sha256"] = {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in ["scripts/validate_senior_datasets.py", "src/afs_rgmnb/data/senior_archive.py"]}
    if export:
        export_start = time.perf_counter()
        sparse.save_npz(output / f"{key}_counts.npz", dataset.X)
        np.save(output / f"{key}_labels.npy", dataset.y)
        with (output / f"{key}_identifiers.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["row", "archive_identifier", "original_class_code", "taxon_name_if_available"])
            writer.writerows((i, identifier, int(dataset.y[i]), dataset.label_names.get(int(dataset.y[i]), ""))
                             for i, identifier in enumerate(dataset.identifiers))
        report["matrix_artifacts"] = {
            "counts": f"{key}_counts.npz", "labels": f"{key}_labels.npy",
            "identifiers": f"{key}_identifiers.tsv",
            "orientation": "rows=samples; columns=preserved source feature IDs",
            "column_to_source_id": "column + 1" if is_document else "column",
            "dna_feature_id_to_sequence_mapping": None if is_document else "unavailable; do not assume ACGT lexical order",
            "vocabulary": "declared source-ID space, including all-zero columns; fit-only selection required for experiments",
        }
        report["export_seconds"] = time.perf_counter() - export_start
        report["process_lifetime_peak_rss_bytes"] = peak_rss_bytes()
    (output / f"{key}_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", choices=["all", "20news", "bacteria", "fungi", "mdr88"], default="all",
                        help="all audits the three priority datasets; mdr88 is optional and separate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-matrices", action="store_true")
    args = parser.parse_args()
    root, output = args.archive.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the read-only reference archive")
    output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "all":
        reports = []
        for key in ["20news", "bacteria", "fungi"]:
            command = [sys.executable, str(Path(__file__).resolve()), "--archive", str(root),
                       "--output", str(output), "--dataset", key, "--seed", str(args.seed)]
            if args.export_matrices:
                command.append("--export-matrices")
            subprocess.run(command, check=True)
            reports.append(json.loads((output / f"{key}_validation.json").read_text(encoding="utf-8")))
        (output / "senior_validation_summary.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        result = audit_one(root, args.dataset, output, args.seed, args.export_matrices)
        print(json.dumps({k: result[k] for k in ["dataset", "samples", "classes",
              "observed_feature_dimensions", "nnz", "total_audit_seconds",
              "process_lifetime_peak_rss_bytes"]}))


if __name__ == "__main__":
    main()
