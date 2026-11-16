# Validation results - independently inspected local run

## Commands and retained evidence

Original outputs at `outputs/2026-11-16/` were preserved. This continuation read
their full JSON contents and ran a fresh export into a separate directory:

```powershell
python scripts/validate_senior_datasets.py --archive '.\107-???' --output '.\outputs\2026-11-16\verified_run' --dataset all --export-matrices
python scripts/validate_senior_datasets.py --archive '.\107-???' --output '.\outputs\2026-11-16\optional_mdr88' --dataset mdr88
python -m unittest discover -s tests -v
python tools/check_thesis_spec.py
```

Every exported matrix and label array was reloaded and checked against its JSON:
shape, nnz, integer dtype, positive stored counts and label length match. Each
source fingerprint matches the corresponding earlier local run. Complete raw
outputs remain ignored; [verified_metrics.json](verified_metrics.json) contains
only derived measurements. It records the exact code hashes and dirty-worktree
state, because the continuation began from committed code while other changes
were being developed. No raw data or historical programs were executed or uploaded.

## Full count-matrix audit and export

| Dataset | Nonzero entries | Sparsity | CSR memory (MiB) | Audit (seconds) | Export (seconds) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 20Newsgroup | 1,505,793 | 99.9044% | 17.30 | 1.30 | 0.51 | 184.13 |
| Bacteria | 8,792,221 | 97.8992% | 100.64 | 64.09 | 1.30 | 394.92 |
| Fungi | 7,138,523 | 98.5909% | 81.72 | 69.50 | 0.91 | 347.87 |

Audit time includes source loading, schema/count/duplicate inspection and a
vocabulary conversion check; export is timed separately. Peak RSS includes
Python, native libraries and export allocations in a fresh dataset worker.
CSR memory is matrix data/index/indptr storage, not peak process memory. Storage
cache and filesystem overhead make fresh timings differ from earlier runs
(1.32/46.85/12.45 seconds). Neither set is model-training time.

Priority matrices are nonnegative `int64`, with stored count ranges 1-301,
1-8 and 1-9 respectively. Zero-valued cells are implicit. No zero rows or invalid
records were found. Counts were not normalized, binarized or weighted by TF-IDF.

## Identical-vector audit

| Dataset | Repeated groups | Rows including first | Excess rows | Cross-label groups | Cross-label rows |
|---|---:|---:|---:|---:|---:|
| 20Newsgroup | 38 | 89 | 51 | 5 | 10 |
| Bacteria | 31 | 67 | 36 | 0 | 0 |
| Fungi | 225 | 671 | 446 | 16 | 46 |

These are **identical count vectors**, not proof of identical raw documents or
sequences. All rows remain in the descriptive audit. Some duplicate vectors cross
class labels; neither removal nor label repair is justified from counts alone.
The conversion splits below retain these vectors and are therefore **not clean
performance evaluation splits**. A reviewed duplicate/group policy is required
before reporting classifier results; near-duplicates cannot be assessed here.

## Training-only observed columns

| Dataset | Conversion protocol | Features observed in fit only | Rows by role |
|---|---|---:|---|
| 20Newsgroup | three_way_conversion_audit | 66,570 | fit=12,049, search=3,013, test=3,766 |
| Bacteria | extraction_audit_only_two_way | 63,682 | fit=3,193, heldout=3,193 |
| Fungi | extraction_audit_only_two_way | 57,411 | fit=3,865, heldout=3,865 |

Seed 42, disjoint row indices, fit-only feature masks. The 20News check uses one
outer fivefold partition, with 20% of its training portion reserved for search.
Genes use a two-way 50/50 conversion audit because size-two classes cannot support
three partitions. No candidate fitting, PSO fitness or accuracy was computed.
This is not completed fivefold cross-validation. The original preprocessing that
created these fixed source columns cannot be audited from the saved counts;
fit-only masks prevent adding held-out columns but cannot undo upstream leakage.

## Output paths

`outputs/2026-11-16/verified_run/` contains, for each priority dataset:
`*_counts.npz`, `*_labels.npy`, `*_identifiers.tsv`, `*_validation.json`, plus
`senior_validation_summary.json`. Identifiers are archive paths/row numbers, not
original accessions or Message-IDs. NPZ columns preserve declared source-ID space
(including zero columns); 20News source IDs equal column+1, gene IDs equal column.
The historical DNA feature-ID-to-sequence mapping is not established, so do not
assign these columns the new extractor's lexical ACGT order.

Optional MDR88 audit: **14,680 records, 3 classes, 12,331 observed features**,
583,473 nonzero entries. Source raw-text label totals agree with count-file class
sizes as a multiset, but row alignment and raw preprocessing reproduction remain
unverified. The missing seven-record class was absent before this audit.

## Test and source status

Baseline 11 tests and the expanded 32-test suite pass locally on Python 3.12.7.
Current runtime: NumPy 2.4.3, SciPy 1.17.1,
scikit-learn 1.9.0. These differ from the pinned Python 3.10
CI requirements; this local run does not claim a new Python 3.10 execution.
The thesis fingerprint checker reports all 12 reviewed sources unchanged at
`669245c9be4073e682ae76d6b0bdf80cd724d89b`; the thesis working tree is clean.

Remaining limits: priority raw-text/sequence extraction, confirmed biological
rank, original sequence lengths/ambiguity/IDs, raw near-duplicates, final class
eligibility and independent evaluation settings. No AFS-RGMNB, PSO, ensemble or
classification performance result is claimed.
