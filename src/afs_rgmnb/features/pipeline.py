from __future__ import annotations

"""Sparse count extraction with explicit all-record or balanced sampling."""
import argparse
import csv
import gzip
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Sequence
import numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from ..data.readers import Record
from ..data.preprocessing import (count_labels, choose_labels, sample_records,
                                  sampling_settings, label_selection_report)
from .dna_kmer import sparse_dna_matrix, kmer_to_text
from .document import vectorizer_from_args, preprocessing_from_args
from dataclasses import asdict
LOG = logging.getLogger("afs-rgmnb")

def extract_fold(records: Sequence[Record], train_idx: np.ndarray, search_idx: np.ndarray,
                 test_idx: np.ndarray, args: argparse.Namespace, out: Path, labels: List[str]):
    out.mkdir(parents=True, exist_ok=True)
    fit = [records[int(i)] for i in train_idx]
    search = [records[int(i)] for i in search_idx]
    test = [records[int(i)] for i in test_idx]
    if args.kind == "dna":
        all_fit = sparse_dna_matrix([r.content for r in fit], args.k, None)
        observed = np.unique(all_fit.indices).astype(np.int32)
        features = [kmer_to_text(int(i), args.k) for i in observed]
        Xs = [all_fit[:, observed].tocsr(),
              sparse_dna_matrix([r.content for r in search], args.k, observed),
              sparse_dna_matrix([r.content for r in test], args.k, observed)]
    else:
        vec = vectorizer_from_args(args)
        Xs = [vec.fit_transform([r.content for r in fit]),
              vec.transform([r.content for r in search]),
              vec.transform([r.content for r in test])]
        features = list(vec.get_feature_names_out())
    observed_count = len(features)
    if observed_count < args.min_features:
        raise ValueError(f"{out.name}: only {observed_count} observed features in Dfit "
                         f"(required >= {args.min_features}); do not count all-zero 8-mers")
    label_id = {label: i for i, label in enumerate(labels)}
    roles = (("fit", fit), ("search", search), ("test", test))
    for (role, recs), X in zip(roles, Xs):
        if X.data.size and (np.any(X.data < 0) or np.any(X.data != X.data.astype(np.int64))):
            raise ValueError("Input is not a nonnegative integer matrix")
        sparse.save_npz(out / f"X_{role}.npz", X.tocsr())
        np.save(out / f"y_{role}.npy", np.asarray([label_id[r.label] for r in recs], dtype=np.int32))
        with (out / f"ids_{role}.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sample_id", "label_id", "taxon_or_class", "lineage"])
            for rec in recs:
                writer.writerow([rec.identifier, label_id[rec.label], rec.label, rec.lineage])
    with (out / "features.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["column", "feature"])
        writer.writerows(enumerate(features))
    summary = {"kind": args.kind, "k": args.k if args.kind == "dna" else None,
               "feature_definition": f"overlapping DNA {args.k}-mer counts" if args.kind == "dna" else "word occurrence counts",
               "vocabulary_fit_scope": "Dfit only", "n_features": observed_count,
               "n_fit": len(fit), "n_search": len(search), "n_test": len(test),
               "nonzero_elements": {role: int(X.nnz) for (role, _), X in zip(roles, Xs)},
               "zero_feature_rows": {role: int(np.count_nonzero(np.diff(X.indptr) == 0))
                                     for (role, _), X in zip(roles, Xs)},
               "class_distribution": {role: dict(Counter(r.label for r in recs)) for role, recs in roles}}
    if args.kind == "text":
        summary["document_preprocessing"] = getattr(args, "document_preprocessing", "legacy")
        if summary["document_preprocessing"] == "senior":
            policy = asdict(preprocessing_from_args(args))
            policy["stopword_count"] = len(policy.pop("stopwords"))
            summary["document_preprocessing_policy"] = policy
        summary["text_encoding"] = getattr(args, "encoding", "utf-8")
        summary["decoding_errors"] = getattr(args, "encoding_errors", "strict")
    if args.baseline:
        # Reference check only, not user's PSO/ensemble method.
        model = MultinomialNB(alpha=1.0).fit(Xs[0], [r.label for r in fit])
        scores = {}
        for (role, recs), X in zip(roles[1:], Xs[1:]):
            truth = [r.label for r in recs]
            pred = model.predict(X)
            scores[role] = {"accuracy": float(accuracy_score(truth, pred)),
                            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
                            "balanced_accuracy": float(balanced_accuracy_score(truth, pred))}
        summary["baseline_mnb_laplace_1"] = scores
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("%s: fit=%d search=%d test=%d observed_features=%d", out.name,
             len(fit), len(search), len(test), observed_count)
    return summary


def export_selected(records: Sequence[Record], out: Path, kind: str) -> None:
    with gzip.open(out / "selected_samples.tsv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "class", "lineage", "sequence" if kind == "dna" else "document"])
        for rec in records:
            writer.writerow([rec.identifier, rec.label, rec.lineage, rec.content])


def validated_splits(y: np.ndarray, labels: Sequence[str], args: argparse.Namespace):
    """Check every realized split before exporting matrices; never drop rare classes."""
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not 0 < args.search_fraction < 1:
        raise ValueError("--search-fraction must be between 0 and 1")
    counts = np.bincount(y, minlength=len(labels))
    rare = [f"{labels[i]} ({int(n)})" for i, n in enumerate(counts) if n < args.folds]
    if rare:
        raise ValueError(f"{args.folds}-fold stratification is infeasible: each retained class "
                         f"needs at least {args.folds} records after deduplication; found "
                         f"{rare[:8]!r}. No classes were automatically removed. Inspect the "
                         "selection audit, reduce --folds, or explicitly set --min-class-size "
                         "or --exclude-class and document the changed population.")
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = []
    expected = set(range(len(labels)))
    for index, (train_idx, test_idx) in enumerate(outer.split(np.arange(len(y)), y), 1):
        try:
            fit_idx, search_idx = train_test_split(
                train_idx, test_size=args.search_fraction, stratify=y[train_idx],
                random_state=args.seed + index)
        except ValueError as exc:
            raise ValueError(f"Fold {index} cannot produce stratified Dfit/Dsearch with "
                             "the configured --search-fraction. No classes were removed. "
                             "Inspect class counts and explicitly revise the split or class "
                             f"exclusion policy. Details: {exc}") from exc
        for role, indices in (("fit", fit_idx), ("search", search_idx), ("test", test_idx)):
            missing = expected - set(y[indices])
            if missing:
                raise ValueError(f"Fold {index} D{role} is missing classes "
                                 f"{[labels[i] for i in sorted(missing)]!r}; no classes were "
                                 "automatically removed. Revise --search-fraction or explicitly "
                                 "filter rare classes and record the exclusion.")
        splits.append((fit_idx, search_idx, test_idx))
    return splits


def run(args: argparse.Namespace) -> dict:
    if args.kind == "dna" and (args.k < 1 or args.k > 15):
        raise ValueError("k must be between 1 and 15 for two-bit 32-bit feature indexing")
    settings = sampling_settings(args)
    args.output.mkdir(parents=True, exist_ok=True)
    stats = count_labels(args)
    selection = label_selection_report(stats, args)
    (args.output / "class_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = choose_labels(stats, args)
    LOG.info("Selected %d classes; class counts: %s", len(selected),
             ", ".join(f"{s}:{stats[s]}" for s in selected[:10]))
    records, sampling_audit = sample_records(args, selected, return_report=True)
    (args.output / "sampling_audit.json").write_text(
        json.dumps(sampling_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(records) < args.min_total_samples:
        raise ValueError(f"Only {len(records)} samples; requirement is {args.min_total_samples}")
    labels = sorted(selected)
    label_id = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_id[r.label] for r in records], dtype=np.int32)
    split_indices = validated_splits(y, labels, args)
    with (args.output / "class_index.tsv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["label_id", "full_taxon_or_class", "original_count", "selected_count"])
        for label in labels:
            wr.writerow([label_id[label], label, stats[label], sum(r.label == label for r in records)])
    export_selected(records, args.output, args.kind)
    all_summary = []
    # Output folds with separate Dfit/Dsearch/Dtest and FIT-ONLY vocabulary.
    for i, (fit_idx, search_idx, te_idx) in enumerate(split_indices, 1):
        all_summary.append(extract_fold(records, fit_idx, search_idx, te_idx, args,
                                        args.output / f"fold_{i:02d}", labels))
    args_config = vars(args).copy()
    args_config = {k: str(v) if isinstance(v, Path) else v for k, v in args_config.items()}
    manifest = {"schema_version": 2, "parameters": args_config,
                "sampling_settings": settings, "class_selection": selection,
                "sampling_audit": sampling_audit,
                "source_file_size_bytes": args.input.stat().st_size if args.input.is_file() else None,
                "counts_eligible_labels": len(stats),
                "selected_class_count": len(labels), "n_selected": len(records),
                "selected_class_counts": dict(Counter(rec.label for rec in records)),
                "folds": all_summary,
                "limitations": ["Exact duplicate inputs removed before splitting; near-duplicate/homology leakage NOT excluded.",
                                "Actual feature dimension reported from Dfit, not theoretical 4**k.",
                                "This is count preprocessing, NOT a reproduction of genetic-code transformations in Lin (2016)."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Finished. Artifacts: %s", args.output)
    return manifest


def inspect(args: argparse.Namespace) -> dict:
    counts = count_labels(args)
    selection = label_selection_report(counts, args)
    eligible = choose_labels(counts, args)
    summary = {"n_labels": len(counts), "n_records": sum(counts.values()),
               "eligible": len(eligible), "selected_labels":
               [{"label": s, "count": counts[s]} for s in eligible],
               "class_size_threshold": selection["settings"]["class_size_threshold"],
               "sampling_settings": selection["settings"], "class_selection": selection}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "all_class_counts.tsv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["class", "samples", "eligible_for_experiment"])
        for label in sorted(counts, key=lambda lab: (-counts[lab], lab)):
            wr.writerow([label, counts[label], int(label in eligible)])
    summary["n_classes_at_least_100"] = sum(n >= 100 for n in counts.values())
    summary["n_classes_at_least_200"] = sum(n >= 200 for n in counts.values())
    summary["n_classes_at_least_500"] = sum(n >= 500 for n in counts.values())
    (args.output / "taxonomy_or_class_inspection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Inspection completed: %s", args.output)
    return summary
