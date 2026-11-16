# Feature representation and extraction — updated 2026-11-16

The current thesis Chapter 3 requires **nonnegative integer occurrence-count features**. Differentiate **reading preserved count matrices** (verified here) from **re-extracting features from raw texts/FASTA** (not reproduced for these datasets).

## DNA

The reference method uses **overlapping DNA 8-mers, stride 1**, recording each occurrence (e.g. `AAAAAAAAA` has two copies of `AAAAAAAA`). A valid DNA 8-mer lies in a 65,536-dimensional possible feature universe. The new Python representation applies documented conventions for ambiguous bases and `U`→`T`, without reverse-complement merging; these conventions have not all been established as identical to the historical program.

The **complete precomputed** Bacteria and Fungi matrices have respectively **64,934** and **61,563** observed feature columns from the verified local audit; these are not fold-specific dimensions. The archived files do not contain original sequences, accession identifiers or full taxonomic lineages. **Family/Genus classification targets are not verified.**

## Documents

蔡幸辰 (2019) §3.1 describes English space/punctuation tokenization (hyphenated forms kept), lowercase normalization, stopword removal and Porter stemming. Current `src/afs_rgmnb/features/document.py` implements configurable variants, but the historical exact tokenizer, complete stoplist and stemming implementation cannot be validated from the saved counts.

Worked example before stemming: `The real-time systems, systems!` with explicit
stopword `the` becomes `real-time:1, systems:2`. The implemented original Porter
variant then yields `real-tim:1, system:2`; hyphenated words remain one token.
Porter is applied after stopword removal. Single-character words/digits are kept
unless the chosen stoplist removes them. This is a documented tokenizer choice,
not proof of exact historic behavior. [NLTK documents its three Porter modes](https://www.nltk.org/api/nltk.stem.porter.html).

Raw-text CLI choices are `--document-preprocessing senior`, an explicit UTF-8
`--stopword-file`, and `--porter-mode original|martin|nltk|none`. The alternative
`--stopwords english` names scikit-learn's list and is not the recovered senior
list. Stoplist SHA-256 and options are recorded. Decoding uses `--encoding` and
strict errors unless replacement is explicitly requested. `--remove-headers`
optionally strips an RFC822 header block; this is distinct from the senior's
removal of header-related words. Default legacy tokenization remains available
for backward compatibility. No raw 20News documents were available to execute
these choices against the original corpus.

The archived 20Newsgroup count source has **83,635 possible source IDs**, of which **83,415** appear in the full sparse matrix. Do not run the tokenizer on this already-processed matrix; treat its values as occurrence counts rather than TF-IDF.

## Data separation and scope

- Use `--sampling-mode all` to retain original multiclass records; duplicates are
  kept and reported by default. `--duplicate-policy drop` explicitly removes
  conflicting-label content and collapses same-label duplicates. Balanced mode
  retains its legacy deduplication/subsampling behavior and emits a warning.
- `--max-classes 0` controls the class cap; it does not turn balanced mode into
  full processing. `--include-class`, `--exclude-class`, and `--min-class-size`
  produce exclusion audits. Every actual fit/search/test split is checked before
  matrices are exported; infeasible class counts fail without automatic removal.
- Fit any new vocabulary and supervised feature scoring on `Dfit` only. Full-matrix feature coverage is descriptive and should not be substituted for a training-fold vocabulary.
- A revised rare-class policy, biological rank validation and raw-sequence/text comparisons are prerequisites for future benchmark claims.
- **No PSO, JMH, ensemble or final classification performance is claimed by this audit.**

See [dataset_status.md](dataset_status.md) and [validation_results.md](validation_results.md).
