# Feature extraction

The active master's Chapter 3 §3.2.1 defines nonnegative integer occurrence counts for documents and DNA. Count matrices are sparse, with consistent column order across fit/search/test partitions.

## DNA

Use overlapping 8-mers with stride one. `AAAAAAAAA` contains two occurrences of `AAAAAAAA`, so its count is **2**, not 1. An ambiguous nucleotide breaks a window: `AAAAAAAANAAAAAAAA` yields two valid `AAAAAAAA` windows and no window crossing `N`. The existing extractor maps RNA `U` to `T`; this is an explicit representation convention, not a verified historical preprocessing choice. Reverse complements are not merged.

Preserve source identifiers and labels. The archive currently supplies precomputed counts; without raw sequences, rerunning 8-mer extraction, checking sequence ambiguity/length, and validating Family/Genus labels cannot be claimed.

## Documents

The senior's description supplied by the user is punctuation/space tokenization retaining hyphenated words, lowercasing, stopword removal, and Porter stemming. A configurable implementation and source verification are being completed. For `Real-time systems, systems!`, hyphen-preserving lowercasing yields `real-time:1`, `systems:2`; with Porter stemming, `systems` becomes `system`.

An existing term-count matrix is read directly; it is not tokenized again. Raw-text availability, document-header handling, encoding, stoplist and stemmer version must be documented. No TF-IDF, binary conversion or length normalization is applied to the primary representation.

## Leakage and sampling

Learn vocabulary on D_fit and apply it unchanged to D_search and D_test. Full-dataset feature inventories are descriptive only. Preserve multiclass distributions by default for the new full-data workflow; any controlled balanced subset must be explicitly selected and labelled.
