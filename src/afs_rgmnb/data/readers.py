from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple
from ..utils.io import open_text
from .taxonomy import parse_taxonomy, label_from_taxonomy, load_metadata

@dataclass(frozen=True)
class Record:
    identifier: str
    label: str
    content: str
    lineage: str = ""


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
    encoding = getattr(args, "encoding", "utf-8")
    errors = getattr(args, "encoding_errors", "strict")
    if args.text_format == "dir":
        # input/class_name/file.txt: recursive file layout, lexicographic order
        for class_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
            for path in sorted(p for p in class_dir.rglob("*.txt") if p.is_file()):
                with open_text(path, encoding=encoding, errors=errors) as fin:
                    yield Record(str(path.relative_to(args.input)), class_dir.name, fin.read())
    elif args.text_format == "jsonl":
        with open_text(args.input, encoding=encoding, errors=errors) as fin:
            for i, line in enumerate(fin, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                label = str(row.get(args.label_col) or "").strip()
                body = row.get(args.text_col) or ""
                if not isinstance(body, str):
                    raise ValueError(f"{args.input}:{i}: document body must be raw text")
                if label and body.strip():
                    seqid = str(row.get(args.id_col) or f"row_{i:09d}")
                    yield Record(seqid, label, body)
    else:
        with open_text(args.input, encoding=encoding, errors=errors) as fin:
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
