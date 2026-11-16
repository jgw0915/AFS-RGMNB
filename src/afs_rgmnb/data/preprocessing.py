from __future__ import annotations

import argparse
import hashlib
import logging
import random
from collections import Counter
from typing import Dict, List, Sequence
from .readers import Record, stream
LOG = logging.getLogger("afs-rgmnb")

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


def sampling_settings(args: argparse.Namespace) -> dict:
    """Resolve legacy balanced defaults without applying them to all-record mode."""
    mode = getattr(args, "sampling_mode", "balanced")
    if mode not in {"all", "balanced"}:
        raise ValueError("sampling_mode must be 'all' or 'balanced'")
    minimum = getattr(args, "min_class_size", None)
    minimum = (100 if mode == "balanced" else 1) if minimum is None else minimum
    maximum = getattr(args, "max_classes", None)
    maximum = (10 if mode == "balanced" else 0) if maximum is None else maximum
    if minimum < 1 or maximum < 0:
        raise ValueError("min_class_size must be positive and max_classes nonnegative")
    if mode == "balanced" and args.samples_per_class < 1:
        raise ValueError("samples_per_class must be positive in balanced mode")
    threshold = max(minimum, args.samples_per_class) if mode == "balanced" else minimum
    duplicate_policy = getattr(args, "duplicate_policy", None) or (
        "keep" if mode == "all" else "drop")
    if duplicate_policy not in {"keep", "drop"}:
        raise ValueError("duplicate_policy must be keep or drop")
    return {"mode": mode, "minimum_class_size": minimum,
            "class_size_threshold": threshold, "maximum_classes": maximum,
            "samples_per_class": args.samples_per_class if mode == "balanced" else None,
            "duplicate_policy": duplicate_policy}


def label_selection_report(counts: Counter, args: argparse.Namespace) -> dict:
    settings = sampling_settings(args)
    include = set(getattr(args, "include_class", None) or [])
    exclude = set(getattr(args, "exclude_class", None) or [])
    if include & exclude:
        raise ValueError("A class cannot be both included and excluded")
    unknown = (include | exclude) - set(counts)
    if unknown:
        raise ValueError(f"Explicit class filter contains unknown labels: {sorted(unknown)!r}")
    candidates, excluded = [], []
    for label in sorted(counts, key=lambda value: (-counts[value], value)):
        reason = None
        if include and label not in include:
            reason = "not_in_explicit_include_list"
        elif label in exclude:
            reason = "explicit_exclude_list"
        elif counts[label] < settings["class_size_threshold"]:
            reason = "below_class_size_threshold"
        elif settings["maximum_classes"] and len(candidates) >= settings["maximum_classes"]:
            reason = "maximum_classes_limit"
        if reason:
            excluded.append({"label": label, "count": int(counts[label]), "reason": reason})
        else:
            candidates.append(label)
    return {"settings": settings, "selected_labels": candidates,
            "original_valid_class_counts": dict(counts), "excluded_classes": excluded,
            "excluded_class_records": sum(row["count"] for row in excluded)}


def choose_labels(counts: Counter, args: argparse.Namespace) -> List[str]:
    report = label_selection_report(counts, args)
    candidates = report["selected_labels"]
    if len(candidates) < args.min_classes:
        raise ValueError(f"Only {len(candidates)} labels remain after the recorded class filters; "
                         f"need >= {args.min_classes}. Inspect the source or explicitly revise "
                         "--min-classes, --min-class-size, or class filters.")
    return candidates


def sample_records(args: argparse.Namespace, labels: Sequence[str], *, return_report: bool = False):
    """Keep all records or explicitly balance them, reporting every exclusion.

    Duplicate/conflicting content is detected across the entire selected input, even
    in balanced mode. Balanced content storage uses a reservoir; digest/ID indexes
    still grow with the number of input records. Original records are never edited.
    """
    settings = sampling_settings(args)
    if settings["mode"] == "balanced":
        LOG.warning("Balanced sampling changes the original class distribution. "
                    "Use --sampling-mode all to preserve multiclass proportions.")
    rng = random.Random(args.seed)
    reservoirs: Dict[str, List[Record]] = {lab: [] for lab in labels}
    seen, input_counts = Counter(), Counter()
    digests, identifiers = {}, set()
    report = {"mode": settings["mode"], "reader_emitted_records": 0,
              "duplicate_policy": settings["duplicate_policy"],
              "invalid_content_records": 0, "excluded_label_records": 0,
              "duplicate_content_records_removed": 0,
              "conflicting_label_records_removed": 0,
              "duplicate_audit_scope": "all valid records in selected classes",
              "reader_scope_note": "Reader-level rejected taxonomy/empty fields are not emitted here."}
    wanted = args.samples_per_class + max(10, args.samples_per_class // 2)
    for rec in stream(args):
        report["reader_emitted_records"] += 1
        if not valid_record(rec, args):
            report["invalid_content_records"] += 1
            continue
        if rec.label not in reservoirs:
            report["excluded_label_records"] += 1
            continue
        if not rec.identifier or rec.identifier in identifiers:
            raise ValueError(f"Missing or duplicate sample identifier {rec.identifier!r}; "
                             "repair the input IDs before splitting. IDs are never silently renamed.")
        identifiers.add(rec.identifier)
        input_counts[rec.label] += 1
        normalized = rec.content.upper().replace("U", "T") if args.kind == "dna" else rec.content
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in digests:
            entry = digests[digest]
            entry[0] += 1
            entry[1].add(rec.label)
            if settings["duplicate_policy"] == "drop":
                continue
        else:
            digests[digest] = [1, {rec.label}]
        seen[rec.label] += 1
        bucket = reservoirs[rec.label]
        if settings["mode"] == "all" or len(bucket) < wanted:
            bucket.append(rec)
        else:
            j = rng.randrange(seen[rec.label])
            if j < wanted:
                bucket[j] = rec
    conflicts = {digest for digest, (_, labs) in digests.items() if len(labs) > 1}
    report["conflicting_label_records_detected"] = sum(digests[key][0] for key in conflicts)
    report["same_label_duplicate_excess_detected"] = sum(
        count - 1 for count, labs in digests.values() if len(labs) == 1)
    if settings["duplicate_policy"] == "drop":
        report["conflicting_label_records_removed"] = report["conflicting_label_records_detected"]
        report["duplicate_content_records_removed"] = report["same_label_duplicate_excess_detected"]
    clean = {label: [] for label in labels}
    for label, bucket in reservoirs.items():
        for rec in bucket:
            normalized = rec.content.upper().replace("U", "T") if args.kind == "dna" else rec.content
            if settings["duplicate_policy"] == "keep" or (
                hashlib.sha256(normalized.encode("utf-8")).hexdigest() not in conflicts):
                clean[label].append(rec)
    final = []
    for label in labels:
        bucket = clean[label]
        if settings["mode"] == "balanced":
            rng.shuffle(bucket)
            if len(bucket) < args.samples_per_class:
                raise ValueError(f"After exact deduplication, class {label!r} has only "
                                 f"{len(bucket)} unique reservoir candidates, below "
                                 f"{args.samples_per_class}; inspect duplicates, explicitly lower "
                                 "--samples-per-class, or use --sampling-mode all.")
            final.extend(bucket[:args.samples_per_class])
        else:
            final.extend(bucket)
    if settings["mode"] == "balanced":
        rng.shuffle(final)
    report["input_selected_class_counts"] = dict(input_counts)
    report["retained_class_counts"] = dict(Counter(rec.label for rec in final))
    eligible = sum(input_counts.values()) - report["conflicting_label_records_removed"] - report["duplicate_content_records_removed"]
    report["downsampled_records"] = eligible - len(final)
    # Compatibility field applies only when duplicates were explicitly removed.
    report["downsampled_unique_records"] = report["downsampled_records"] if settings["duplicate_policy"] == "drop" else 0
    if settings["duplicate_policy"] == "keep" and (
        report["conflicting_label_records_detected"] or report["same_label_duplicate_excess_detected"]):
        LOG.warning("Duplicate content retained by explicit keep policy; use reviewed "
                    "group-aware splits before making classification performance claims.")
    report["retained_records"] = len(final)
    LOG.info("Selection audit: %s", report)
    return (final, report) if return_report else final
