#!/usr/bin/env python3
"""MNB feature-extraction pipeline for DNA (8-mer) and documents, Python >=3.10.

IMPORTANT: This program implements the current thesis's count-vector formulation,
not Lin (2016)'s genetic-code/amino-acid transformations. Taxonomy-aware inputs
are supported for the main family/genus DNA experiments.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

LOG = logging.getLogger("mnb-extract")
DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
RANKS6 = ("domain", "phylum", "class", "order", "family", "genus")
RANKS7 = ("domain", "kingdom", "phylum", "class", "order", "family", "genus")
BAD_TAXON = ("uncultured", "unclassified", "unknown", "unidentified", "metagenome",
             "incertae sedis", "environmental sample", "uncertain", "unassigned")


@dataclass(frozen=True)
class Record:
    identifier: str
    label: str
    content: str
    lineage: str = ""


def open_text(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" \
        else path.open("r", encoding="utf-8", errors="replace")


def fasta_iter(path: Path) -> Iterator[Tuple[str, str]]:
    """Streaming parser; joins multi-line sequences, no Biopython dependency."""
    header, pieces = None, []
    with open_text(path) as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(pieces)
                header, pieces = line[1:].strip(), []
            else:
                if header is None:
                    raise ValueError("FASTA sequence encountered before its header")
                pieces.append(line)
        if header is not None:
            yield header, "".join(pieces)


def parse_taxonomy(raw: str, rank_schema: str, domain: str) -> Optional[Dict[str, str]]:
    parts = [x.strip() for x in raw.strip().split(";") if x.strip()]
    # DADA2 may include a seventh species level; toGenus should have at most seven.
    if rank_schema == "six":
        if len(parts) < 6:
            return None
        parts, ranks = parts[:6], RANKS6
    elif rank_schema == "seven":
        if len(parts) < 7:
            return None
        parts, ranks = parts[:7], RANKS7
    else:
        # Auto accepts only unambiguous 6- or 7-level paths. For SILVA 144
        # official 7-rank taxonomy, use --rank-schema seven explicitly.
        if len(parts) == 6:
            ranks = RANKS6
        elif len(parts) == 7:
            # Distinguish seven prokaryotic ranks from six + species.
            # There is no reliable universal inference: require explicit choice.
            raise ValueError("Seven taxonomy tokens are ambiguous: specify --rank-schema six or seven")
        else:
            return None
    tax = dict(zip(ranks, parts))
    if domain and tax.get("domain", "").casefold() != domain.casefold():
        return None
    return tax


def label_from_taxonomy(tax: Dict[str, str], rank: str) -> Optional[str]:
    if rank not in tax or not tax[rank].strip():
        return None
    ranks = RANKS7 if "kingdom" in tax else RANKS6
    i = ranks.index(rank)
    path = [tax.get(k, "").strip() for k in ranks[:i + 1]]
    if any(not s or s.casefold() in {"na", "none", "null"} or
           any(bad in s.casefold() for bad in BAD_TAXON) for s in path):
        return None
    # Retain all ancestral ranks to distinguish same genus/family names in
    # different lineages, instead of silently merging distinct taxa.
    return ";".join(path)


def load_metadata(path: Path, id_col: str, rank: str, delimiter: str) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    with open_text(path) as fin:
        rdr = csv.DictReader(fin, delimiter=delimiter)
        required = {id_col, rank}
        if not required.issubset(rdr.fieldnames or []):
            raise ValueError(f"metadata requires columns: {sorted(required)}")
        for row in rdr:
            seqid = (row[id_col] or "").strip()
            family = (row.get("family") or "").strip()
            genus = (row.get("genus") or "").strip()
            value = (row.get(rank) or "").strip()
            if not seqid or not value:
                continue
            if any(x in value.casefold() for x in BAD_TAXON):
                continue
            # For genus, join parent family whenever present.
            label = f"{family};{genus}" if rank == "genus" and family else value
            if seqid in mapping and mapping[seqid][0] != label:
                raise ValueError(f"Conflicting metadata for id {seqid!r}")
            mapping[seqid] = (label, f"{family};{genus}" if family else value)
    return mapping


def dna_records(args: argparse.Namespace) -> Iterator[Record]:
    metadata = None
    if args.dna_format == "metadata":
        if not args.metadata:
            raise ValueError("--dna-format metadata requires --metadata")
        metadata = load_metadata(args.metadata, args.id_col, args.rank, args.meta_delimiter)
    for i, (header, seq) in enumerate(fasta_iter(args.input), start=1):
        if args.dna_format == "dada2":
            seqid, taxonomy = f"seq_{i:09d}", header
        elif args.dna_format == "silva":
            # SILVA export: >accession.start.stop taxonomy;taxonomy;...
            fields = header.split(maxsplit=1)
            if len(fields) != 2 or ";" not in fields[1]:
                continue
            seqid, taxonomy = fields
        else:
            seqid, taxonomy = header.split(maxsplit=1)[0], ""
        if metadata is None:
            tax = parse_taxonomy(taxonomy, args.rank_schema, args.domain)
            if tax is None:
                continue
            label = label_from_taxonomy(tax, args.rank)
            lineage = ";".join(tax.values())
        else:
            item = metadata.get(seqid)
            if item is None:
                continue
            label, lineage = item
        if label:
            # RNA uracil can be treated as DNA thymine for k-mer counting.
            yield Record(seqid, label, seq.upper().replace("U", "T"), lineage)


def text_records(args: argparse.Namespace) -> Iterator[Record]:
    if args.text_format == "dir":
        # input/class_name/file.txt: recursive file layout, lexicographic order
        for class_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
            for path in sorted(p for p in class_dir.rglob("*.txt") if p.is_file()):
                with open_text(path) as fin:
                    yield Record(str(path.relative_to(args.input)), class_dir.name, fin.read())
    elif args.text_format == "jsonl":
        with open_text(args.input) as fin:
            for i, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                label = str(row.get(args.label_col, "")).strip()
                body = str(row.get(args.text_col, ""))
                if label and body.strip():
                    seqid = str(row.get(args.id_col) or f"row_{i:09d}")
                    yield Record(seqid, label, body)
    else:
        with open_text(args.input) as fin:
            rdr = csv.DictReader(fin, delimiter=args.delimiter)
            required = {args.label_col, args.text_col}
            if not required.issubset(rdr.fieldnames or []):
                raise ValueError(f"CSV/TSV input requires columns: {sorted(required)}")
            for i, row in enumerate(rdr, start=1):
                label = (row[args.label_col] or "").strip()
                body = row[args.text_col] or ""
                if label and body.strip():
                    seqid = (row.get(args.id_col) or "").strip() or f"row_{i:09d}"
                    yield Record(seqid, label, body)


def stream(args: argparse.Namespace) -> Iterator[Record]:
    return dna_records(args) if args.kind == "dna" else text_records(args)


def valid_record(record: Record, args: argparse.Namespace) -> bool:
    if not record.content:
        return False
    if args.kind == "dna":
        run = 0
        for ch in record.content.upper():
            run = run + 1 if ch in "ACGTU" else 0
            if run >= args.k:
                return True
        return False
    return bool(record.content.strip())


def count_labels(args: argparse.Namespace) -> Counter:
    counts: Counter = Counter()
    total = 0
    for rec in stream(args):
        total += 1
        if valid_record(rec, args):
            counts[rec.label] += 1
    LOG.info("Scanned %d records, %d eligible labels, %d valid records", total,
             len(counts), sum(counts.values()))
    return counts


def choose_labels(counts: Counter, args: argparse.Namespace) -> List[str]:
    threshold = max(args.min_class_size, args.samples_per_class)
    candidates = sorted((lbl for lbl, n in counts.items() if n >= threshold),
                        key=lambda label: (-counts[label], label))
    if args.max_classes > 0:
        candidates = candidates[:args.max_classes]
    if len(candidates) < args.min_classes:
        raise ValueError(f"Only {len(candidates)} labels have >= {threshold} records; "
                         f"need >= {args.min_classes}. Try lower thresholds after checking the source.")
    return candidates


def sample_records(args: argparse.Namespace, labels: Sequence[str]) -> List[Record]:
    """Per-class reservoir sampling with bounded memory (two passes over input)."""
    rng = random.Random(args.seed)
    reservoirs: Dict[str, List[Record]] = {lab: [] for lab in labels}
    seen = Counter()
    wanted = args.samples_per_class + max(10, args.samples_per_class // 2)
    for rec in stream(args):
        if rec.label not in reservoirs or not valid_record(rec, args):
            continue
        seen[rec.label] += 1
        bucket = reservoirs[rec.label]
        if len(bucket) < wanted:
            bucket.append(rec)
        else:
            j = rng.randrange(seen[rec.label])
            if j < wanted:
                bucket[j] = rec
    # Globally deduplicate exact sequences/text; reject cross-class label collisions.
    all_samples = [rec for label in labels for rec in reservoirs[label]]
    # Hash collision extremely unlikely; using SHA-256 to identify exact duplicates.
    grouped: Dict[str, List[Record]] = defaultdict(list)
    for rec in all_samples:
        normalized = rec.content.upper() if args.kind == "dna" else rec.content
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        grouped[digest].append(rec)
    clean = {label: [] for label in labels}
    conflicts = 0
    for grp in grouped.values():
        if len({rec.label for rec in grp}) > 1:
            conflicts += len(grp)
            continue
        clean[grp[0].label].append(grp[0])
    LOG.info("Removed %d conflicting-label duplicates; %d unique sampled candidates",
             conflicts, sum(map(len, clean.values())))
    final = []
    for label in labels:
        bucket = clean[label]
        rng.shuffle(bucket)
        if len(bucket) < args.samples_per_class:
            raise ValueError(f"After exact deduplication, class {label!r} has only "
                             f"{len(bucket)} unique sampled sequences/documents, "
                             f"below {args.samples_per_class}; lower sample count or "
                             "select another class/source")
        final.extend(bucket[:args.samples_per_class])
    rng.shuffle(final)
    return final


def kmer_to_text(index: int, k: int) -> str:
    arr = ["A"] * k
    for i in range(k - 1, -1, -1):
        arr[i] = "ACGT"[index & 3]
        index >>= 2
    return "".join(arr)


def count_kmers(seq: str, k: int) -> Dict[int, int]:
    """Rolling two-bit index, overlapping windows, no ambiguous-base bridging."""
    mask, code, length = (1 << (2 * k)) - 1, 0, 0
    counts: Dict[int, int] = {}
    for ch in seq.upper():
        if ch == "U":  # RNA origin
            ch = "T"
        digit = DNA_MAP.get(ch)
        if digit is None:
            code, length = 0, 0
            continue
        code = ((code << 2) | digit) & mask
        length += 1
        if length >= k:
            counts[code] = counts.get(code, 0) + 1
    return counts


def sparse_dna_matrix(texts: Sequence[str], k: int, vocab: Optional[np.ndarray]) -> sparse.csr_matrix:
    if vocab is None:
        ncols = 4 ** k
        lookup = None
    else:
        ncols = len(vocab)
        lookup = {int(feature): i for i, feature in enumerate(vocab)}
    row_ind, col_ind, values = [], [], []
    for row, seq in enumerate(texts):
        for col, count in count_kmers(seq, k).items():
            mapped = col if lookup is None else lookup.get(col)
            if mapped is not None:
                row_ind.append(row)
                col_ind.append(mapped)
                values.append(count)
    return sparse.csr_matrix((np.asarray(values, dtype=np.int32),
                              (row_ind, col_ind)), shape=(len(texts), ncols), dtype=np.int32)


def extract_fold(records: Sequence[Record], train_idx: np.ndarray, search_idx: np.ndarray,
                 test_idx: np.ndarray, args: argparse.Namespace, out: Path, labels: List[str]):
    out.mkdir(parents=True, exist_ok=True)
    fit = [records[int(i)] for i in train_idx]
    search = [records[int(i)] for i in search_idx]
    test = [records[int(i)] for i in test_idx]
    if args.kind == "dna":
        all_fit = sparse_dna_matrix([r.content for r in fit], args.k, None)
        observed = np.flatnonzero(np.asarray(all_fit.getnnz(axis=0)).ravel()).astype(np.int32)
        features = [kmer_to_text(int(i), args.k) for i in observed]
        Xs = [all_fit[:, observed].tocsr(),
              sparse_dna_matrix([r.content for r in search], args.k, observed),
              sparse_dna_matrix([r.content for r in test], args.k, observed)]
    else:
        vec = CountVectorizer(lowercase=True, strip_accents="unicode",
                              stop_words="english" if args.stopwords == "english" else None,
                              min_df=args.min_df, max_features=args.max_features or None,
                              token_pattern=args.token_pattern, dtype=np.int32)
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


def run(args: argparse.Namespace) -> dict:
    if args.kind == "dna" and (args.k < 1 or args.k > 15):
        raise ValueError("k must be between 1 and 15 for two-bit 32-bit feature indexing")
    if args.samples_per_class < 2 * args.folds:
        raise ValueError("samples per class must be at least twice the number of folds")
    args.output.mkdir(parents=True, exist_ok=True)
    stats = count_labels(args)
    selected = choose_labels(stats, args)
    LOG.info("Selected top %d classes; class counts: %s", len(selected),
             ", ".join(f"{s}:{stats[s]}" for s in selected[:10]))
    records = sample_records(args, selected)
    if len(records) < args.min_total_samples:
        raise ValueError(f"Only {len(records)} samples; requirement is {args.min_total_samples}")
    labels = sorted(set(rec.label for rec in records))
    label_id = {label: i for i, label in enumerate(labels)}
    with (args.output / "class_index.tsv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["label_id", "full_taxon_or_class", "original_count", "selected_count"])
        for label in labels:
            wr.writerow([label_id[label], label, stats[label], sum(r.label == label for r in records)])
    export_selected(records, args.output, args.kind)
    y = np.asarray([label_id[r.label] for r in records], dtype=np.int32)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_summary = []
    # Output folds with separate Dfit/Dsearch/Dtest and FIT-ONLY vocabulary.
    for i, (tr_idx, te_idx) in enumerate(folds.split(np.arange(len(records)), y), 1):
        fit_idx, search_idx = train_test_split(tr_idx, test_size=args.search_fraction,
                                               stratify=y[tr_idx], random_state=args.seed + i)
        all_summary.append(extract_fold(records, fit_idx, search_idx, te_idx, args,
                                        args.output / f"fold_{i:02d}", labels))
    args_config = vars(args).copy()
    args_config = {k: str(v) if isinstance(v, Path) else v for k, v in args_config.items()}
    manifest = {"schema_version": 1, "parameters": args_config,
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
    eligible = choose_labels(counts, args)
    summary = {"n_labels": len(counts), "n_records": sum(counts.values()),
               "eligible": len(eligible), "selected_labels":
               [{"label": s, "count": counts[s]} for s in eligible],
               "class_size_threshold": max(args.min_class_size, args.samples_per_class)}
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


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=["inspect", "extract"], help="Inspect class sizes or export CV matrices")
    p.add_argument("--kind", choices=["dna", "text"], required=True)
    p.add_argument("--input", required=True, type=Path, help="FASTA(.gz), CSV(.gz), or text directory")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-classes", type=int, default=10, help="0 selects every eligible class")
    p.add_argument("--min-classes", type=int, default=5)
    p.add_argument("--min-class-size", type=int, default=100)
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--min-total-samples", type=int, default=500)
    p.add_argument("--min-features", type=int, default=5000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--search-fraction", type=float, default=0.2)
    p.add_argument("--baseline", action="store_true", help="Fit standard Laplace MNB as a sanity check")
    p.add_argument("--k", type=int, default=8, help="Overlapping DNA k-mer size, stride=1")
    p.add_argument("--dna-format", choices=["dada2", "silva", "metadata"], default="dada2")
    p.add_argument("--rank", choices=["family", "genus"], default="family")
    p.add_argument("--rank-schema", choices=["auto", "six", "seven"], default="six",
                   help="DADA2 usually six: Domain/Phylum/Class/Order/Family/Genus; SILVA144 may use seven")
    p.add_argument("--domain", default="Bacteria", help="Filter taxonomy path by first rank; empty string to disable")
    p.add_argument("--metadata", type=Path, help="For --dna-format metadata only")
    p.add_argument("--meta-delimiter", default="\t", help="Delimiter for metadata CSV or TSV")
    p.add_argument("--id-col", default="id")
    p.add_argument("--text-format", choices=["csv", "jsonl", "dir"], default="csv")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--delimiter", default=",", help="Document CSV/TSV delimiter")
    p.add_argument("--min-df", type=int, default=1)
    p.add_argument("--max-features", type=int, default=0, help="0 means no vocabulary cap")
    p.add_argument("--stopwords", choices=["none", "english"], default="none")
    p.add_argument("--token-pattern", default=r"(?u)\b\w\w+\b")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.search_fraction <= 0 or args.search_fraction >= 1:
        parser.error("--search-fraction must be between 0 and 1")
    if args.kind == "dna" and args.k != 8:
        LOG.warning("Thesis protocol fixes k=8; requested k=%d is for sensitivity tests only", args.k)
    try:
        result = inspect(args) if args.command == "inspect" else run(args)
    except (ValueError, FileNotFoundError) as e:
        parser.exit(2, f"ERROR: {e}\n")
    print(json.dumps({"output": str(args.output), "n_selected": result.get("n_selected"),
                      "eligible_labels": result.get("eligible"),
                      "completed": args.command}, ensure_ascii=False))


if __name__ == "__main__":
    main()