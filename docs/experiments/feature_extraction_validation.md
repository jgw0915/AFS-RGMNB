# Real-source representation validation

The independently inspected November 16 report is the current validation record:
[dataset status](../../reports/2026-11-16/dataset_status.md),
[full measured results](../../reports/2026-11-16/validation_results.md), and
[derived machine-readable metrics](../../reports/2026-11-16/verified_metrics.json).

All three priority preserved-count datasets were read in full, exported, and
reloaded with matching source fingerprints. Counts and original multiclass codes
were retained. This validates count conversion, not historical raw extraction.
MDR88 is separately available in an already reduced three-class form.

The report distinguishes full-matrix observed features, fit-only conversion
features, nonzero entries, class distributions, duplicate vectors, invalid rows,
runtime and matrix/process memory. Gene fit-only checks are two-way audits, not
the thesis evaluation protocol. No classifier accuracy experiment is complete.

Raw extractor examples are covered by deterministic synthetic tests (overlap,
ambiguity breaks, U mapping, hyphen tokens, Porter variants and repeated terms).
Their success does not establish biological rank or reproduce missing raw inputs.
