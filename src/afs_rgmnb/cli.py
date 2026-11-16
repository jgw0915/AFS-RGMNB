from __future__ import annotations

"""Sparse count-extraction commands; the experiment runner is planned."""
import argparse
import json
import logging
from pathlib import Path
from .features.pipeline import inspect, run
LOG = logging.getLogger("afs-rgmnb")

def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=["inspect", "extract"], help="Inspect class sizes or export CV matrices")
    p.add_argument("--kind", choices=["dna", "text"], required=True)
    p.add_argument("--input", required=True, type=Path, help="FASTA(.gz), CSV(.gz), or text directory")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-mode", choices=["all", "balanced"], default="balanced",
                   help="all retains original class sizes after explicitly configured filtering; "
                        "balanced is the legacy equal-count subset and changes class proportions")
    p.add_argument("--duplicate-policy", choices=["keep", "drop"], default=None,
                   help="keep preserves records and reports duplicate content; drop removes "
                        "conflicts and deduplicates. Omitted: keep in all, drop in balanced")
    p.add_argument("--max-classes", type=int, default=None,
                   help="Class cap only; 0 means all eligible classes, NOT all records. "
                        "Omitted: 0 in all mode, 10 in balanced mode")
    p.add_argument("--min-classes", type=int, default=5)
    p.add_argument("--min-class-size", type=int, default=None,
                   help="Explicitly exclude smaller classes; omitted: 1 in all mode, "
                        "100 in balanced mode. Fold feasibility is checked separately")
    p.add_argument("--include-class", action="append", default=None,
                   help="Keep this exact class label; repeat for multiple labels")
    p.add_argument("--exclude-class", action="append", default=None,
                   help="Explicitly exclude this exact class label; repeat as needed")
    p.add_argument("--samples-per-class", type=int, default=200,
                   help="Target records per class in balanced mode; ignored in all mode")
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
    p.add_argument("--document-preprocessing", choices=["legacy", "senior"], default="legacy",
                   help="senior: hyphen-preserving tokenizer and explicit stoplist/Porter choices")
    p.add_argument("--stopword-file", type=Path, help="UTF-8 one-word-per-line explicit stoplist")
    p.add_argument("--porter-mode", choices=["none", "original", "martin", "nltk"], default="original")
    p.add_argument("--remove-headers", action="store_true",
                   help="Optional RFC822 header stripping; not verified as historical behavior")
    p.add_argument("--encoding", default="utf-8", help="Raw document decoding, e.g. latin-1")
    p.add_argument("--encoding-errors", choices=["strict", "replace"], default="strict")
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
    except (ValueError, FileNotFoundError, UnicodeError) as e:
        parser.exit(2, f"ERROR: {e}\n")
    print(json.dumps({"output": str(args.output), "n_selected": result.get("n_selected"),
                      "eligible_labels": result.get("eligible"),
                      "completed": args.command}, ensure_ascii=False))
