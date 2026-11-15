# Verification report — 2026-11-15

## Completed locally

- `python -m unittest discover -s tests -v`: **11/11 passed** on Python 3.13.5; source syntax targets Python 3.10, package pins are Python 3.10-compatible. **Server Python 3.10 execution remains unverified**; repository CI targets Python 3.10.
- Synthetic document test: 500 selected samples, 5 classes, 5 outer folds; observed word features per Dfit = **9,173 / 9,182 / 9,165 / 9,160 / 9,159**. Artificial data, not scientific evidence.
- Actual laboratory senior (陳泓宇) Fungi FASTA, Family target: **500 samples (20 families × 25)**; Dfit dimensions per fold = **17,570 / 18,221 / 18,419 / 17,835 / 17,644**.
- Actual laboratory senior (陳泓宇) Bacteria FASTA, Genus target: **500 samples (25 genera × 20)**; Dfit dimensions per fold = **33,721 / 34,162 / 34,012 / 34,006 / 33,882**.
- Fivefold outputs for both real FASTA subsets contained `X_fit`, `X_search`, `X_test`, `y` labels, IDs and feature vocabulary with the proper stratified sizes (`Dfit=320`, `Dsearch=80`, `Dtest=100`).

Optional Laplace MNB sanity-check test accuracy on the *same* old Bacteria and Fungi samples was high (Fungi around 0.90 and Bacteria around 1.00). **Do not interpret these as benchmark performance**: these datasets likely have closely related sequences, and stratified splits alone do not enforce evolutionary independence.

## Not completed / blocked

- The **three original datasets from 蔡幸辰's archive** (1 document + 2 gene sequences) **have not been inspected, parsed, or tested**; the provided ZIP of 1,170,781,101 bytes exceeds the Drive connector's 268,435,456-byte single-file download limit.
- Exact real-data names, file formats, counts, classes, text tokenization and preprocessing used by 蔡幸辰 cannot be honestly reconstructed from the archive yet.
- Python 3.10 on user's own lab server: not executed; local runtime only provides Python 3.13.5.
- Similarity/group-aware biological split and license/redistribution audit are outstanding.

## How to complete the original three-dataset acceptance test

1. Extract the original ZIP on lab server (`unzip`); identify exact three dataset files.
2. Run `python tools/preflight.py /path/to/extracted --output /tmp/tsai_inventory.json`.
3. For each gene FASTA, inspect the first header and taxonomy hierarchy. Use `--dna-format silva`/`dada2`, or create an explicit `id,family,genus` metadata TSV. For document files, use labeled CSV/TSV/JSONL or class subdirectories.
4. Run `feature_pipeline.py inspect` on each to record valid class distributions.
5. Run `feature_pipeline.py extract` using `--max-classes` and `--samples-per-class` satisfying `n >= 500`; verify observed `d >= 5000` in each Dfit, and check all split matrices are nonnegative integers.
6. Log the exact source file SHA256, preprocessing flags, sample sizes, label maps, feature dimensions and split seed. Confirm no cross-split exact or near duplicates.

The project deliberately refuses extraction on undersized datasets, rather than inventing or padding features to make them pass the high-dimensional threshold.