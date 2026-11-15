from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple
from ..utils.io import open_text
RANKS6 = ("domain", "phylum", "class", "order", "family", "genus")
RANKS7 = ("domain", "kingdom", "phylum", "class", "order", "family", "genus")
BAD_TAXON = ("uncultured", "unclassified", "unknown", "unidentified", "metagenome", "incertae sedis", "environmental sample", "uncertain", "unassigned")

def parse_taxonomy(raw: str, rank_schema: str, domain: str) -> Optional[Dict[str, str]]:
    parts = [x.strip() for x in raw.strip().split(";")]
    # Keep internal missing ranks; dropping them would silently shift Family/Genus.
    while parts and not parts[-1]:
        parts.pop()
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
