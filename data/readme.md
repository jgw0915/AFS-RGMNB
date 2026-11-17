# Thesis data storage

This directory is the local workspace for datasets being considered or used in
the high-dimensional multiclass thesis experiments. Only this README and the
two empty-directory markers belong in Git. Dataset files, record-level metadata,
feature matrices and downloaded archives remain local and ignored.

```text
data/
|-- readme.md                 # Tracked storage conventions
|-- raw/                     # Original inputs before feature extraction
|   `-- <dataset>/<version>/
`-- extracted/               # Preserved or newly generated feature counts
    `-- <dataset>/<version>/<representation-or-run>/
```

Creating this layout does not itself approve or validate any dataset. The
read-only senior materials stay under `../reference/` relative to this directory;
they are separate research references and were not moved or copied here.
The November 17 inspection separately downloaded the small official UCI Splice
archive into `raw/uci_splice/retrieved-2026-11-17/` and generated an 8-mer audit in
`extracted/uci_splice/dna8_audit-2026-11-17/`. See the
[candidate review](../docs/experiments/dataset_candidate_review.md) for measured
results and limits; storage here does not make a dataset a final thesis choice.

## `raw/`: data before feature extraction

Store original DNA/RNA sequence files, source document text, download archives,
labels and source metadata here, organized by dataset and version. Here, *raw*
means before feature extraction, not necessarily an uncompressed file. An
upstream tokenized document collection must be described as tokenized rather than
original prose. A precomputed count matrix is an extracted representation even
when received directly from its publisher or a senior researcher.

Keep source files unchanged. Preserve the source URL or local reference path,
retrieval date, dataset/release version, license or access terms, checksums,
original identifiers, label definitions and any official train/test assignment
in a local provenance manifest. For biological data, retain sequence accessions
and taxonomy release/rank evidence when available; record missing metadata
explicitly. A filename alone does not establish biological identity or rank.

## `extracted/`: numerical feature representations

Store sparse nonnegative integer feature-count matrices, aligned labels and
sample identifiers, original feature identifiers, vocabulary mappings and
extraction/audit metadata here. Use separate subdirectories for:

- **Preserved counts:** representations supplied by a senior or publisher.
  Record their source fingerprint and known preprocessing limitations. Do not
  describe them as a reproduction of raw-text or raw-sequence extraction.
- **Newly generated counts:** representations produced from identified raw inputs.
  Record the source fingerprints, extractor configuration, code revision,
  dependency versions, random seeds and relevant normalization/tokenization or
  ambiguous-base policies.

The same dataset can have multiple versions and representations; do not overwrite
one with another. Preserve original sample/class/feature identifiers and record
explicit mappings to matrix rows, label order and feature columns. Report
declared dimensions separately from actually observed columns. Document every
excluded record/class and the reason; a storage location is not proof that the
dataset meets the thesis requirements.

## Experimental separation and eligibility

Preserve original multiclass labels and natural class counts. No automatic class
balancing, binary conversion, rare-class removal or taxonomy remapping follows
from placing data here. Verify sample counts, observed feature counts, labels,
count validity and the required class coverage before an experiment.

Learn vocabulary and supervised feature-scoring parameters from Dfit only; apply
the fixed feature mapping to Dsearch and Dtest. Candidate selection and PSO use
Dsearch. Dtest is reserved for final evaluation. Record all split seeds and
sample/group assignments. Identical vectors must remain within one partition;
sequence/document identity and other dependence groups require their own reviewed
evidence and grouping protocol. Insufficient classes/groups must trigger a clear
failure or a documented scientific decision, not silent resampling.

For newly learned vocabularies, keep fold-specific representations or their
reproducible transformation definitions distinct. A full-corpus inventory may be
stored as an audit artifact, but its vocabulary or supervised selections must not
be silently used as training-fitted information. Masking a historical matrix to
Dfit-observed columns cannot establish that its upstream preprocessing was free
of leakage.

Model-search outputs, particle histories, trained parameters, timing logs and
experiment-specific results belong under ignored `../outputs/`, rather than
being mixed with source datasets. Public derived summaries belong in `../docs/`
or `../reports/` after review; share no raw records or large arrays through Git.

## Verify Git protection

From the repository root, check any intended destination before adding files:

```powershell
git check-ignore -v data/raw/example/version/sequences.txt
git check-ignore -v data/extracted/example/version/counts.npz
git status --short
```

The ignore rules cover all data filenames and nested folders, not only known
sequence or matrix extensions. Only `data/readme.md`, `data/raw/.gitkeep` and
`data/extracted/.gitkeep` are permitted by this layout. Do not use `git add -f` for
datasets. Ignore rules do not remove files already tracked: review
`git ls-files -- data` and staged filenames before every commit.
