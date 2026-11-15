# Feature representation and extraction — updated 2026-11-16

The current thesis Chapter 3 requires **nonnegative integer occurrence-count features**. Differentiate **reading preserved count matrices** (verified here) from **re-extracting features from raw texts/FASTA** (not reproduced for these datasets).

## DNA

The reference method uses **overlapping DNA 8-mers, stride 1**, recording each occurrence (e.g. `AAAAAAAAA` has two copies of `AAAAAAAA`). A valid DNA 8-mer lies in a 65,536-dimensional possible feature universe. The new Python representation applies documented conventions for ambiguous bases and `U`→`T`, without reverse-complement merging; these conventions have not all been established as identical to the historical program.

The **complete precomputed** Bacteria and Fungi matrices have respectively **64,934** and **61,563** observed feature columns from the verified local audit; these are not fold-specific dimensions. The archived files do not contain original sequences, accession identifiers or full taxonomic lineages. **Family/Genus classification targets are not verified.**

## Documents

蔡幸辰 (2019) §3.1 describes English space/punctuation tokenization (hyphenated forms kept), lowercase normalization, stopword removal and Porter stemming. Current `src/afs_rgmnb/features/document.py` implements configurable variants, but the historical exact tokenizer, complete stoplist and stemming implementation cannot be validated from the saved counts.

The archived 20Newsgroup count source has **83,635 possible source IDs**, of which **83,415** appear in the full sparse matrix. Do not run the tokenizer on this already-processed matrix; treat its values as occurrence counts rather than TF-IDF.

## Data separation and scope

- Preserve the original multiclass labels; do not silently undersample, oversample or merge into minority-versus-rest labels.
- Fit any new vocabulary and supervised feature scoring on `Dfit` only. Full-matrix feature coverage is descriptive and should not be substituted for a training-fold vocabulary.
- A revised rare-class policy, biological rank validation and raw-sequence/text comparisons are prerequisites for future benchmark claims.
- **No PSO, JMH, ensemble or final classification performance is claimed by this audit.**

See [dataset_status.md](dataset_status.md) and [validation_results.md](validation_results.md).
