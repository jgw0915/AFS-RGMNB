#!/usr/bin/env python3
"""Backward-compatible imports and CLI for the original extraction pipeline."""
from pathlib import Path
import sys

# Permit historical direct-script usage before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from afs_rgmnb.data.readers import Record, fasta_iter, dna_records, text_records, stream
from afs_rgmnb.data.taxonomy import parse_taxonomy, label_from_taxonomy, load_metadata
from afs_rgmnb.data.preprocessing import valid_record, count_labels, choose_labels, sample_records
from afs_rgmnb.features.dna_kmer import count_kmers, kmer_to_text, sparse_dna_matrix
from afs_rgmnb.features.pipeline import extract_fold, export_selected, run, inspect
from afs_rgmnb.utils.io import open_text
from afs_rgmnb.cli import argument_parser, main

if __name__ == "__main__":
    main()
