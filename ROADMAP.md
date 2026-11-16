# Thesis implementation roadmap

## Completed and verified

- [x] Read active sibling thesis, resolve Chapter 3 equations and included notation.
- [x] Archive-wide Git protection; original reference projects remain read-only.
- [x] Equation-to-module mapping, scientific questions and read-only fingerprint check.
- [x] Recover modular count-extraction package and legacy compatibility wrapper.
- [x] Three original multiclass count sources read with strict schema validation.
- [x] November 16 Markdown report, Chinese professor summary and unsent email draft.
- [x] Configurable hyphen-preserving document tokenizer, stoplist and Porter variants.
- [x] Explicit all-record sampling, class filters, duplicate policies and split preflight.
- [x] Baseline 11 tests preserved; expanded suite covers counts, decoding and exclusions.
- [x] Verified Chapter 3 JMH, initial weights, inclusion updates, L1 change and
  sequential fixed-m sampling; 9 numerical tests (41 total local tests).

## Current session

- [x] Independently export/reload all three matrices; full distributions,
  runtime/memory, source fingerprints and two-page professor PDF visually verified.
- [x] Review and checkpoint report/data changes locally; no push performed.

## Scientific or source blocks

- Raw 20News text and gene sequences/accessions/lineages were not recovered.
- Gene class codes/names do not establish a consistent Family or Genus target.
- Classes with two records cannot be present in fit/search/test simultaneously;
  full gene fivefold experiments need a reviewed target/eligibility protocol.
- Identical count vectors (some with different labels) require a reviewed
  grouping/quality policy before performance claims.
- Exact historical stoplist/stemmer and DNA feature-ID-to-sequence order are unknown.
- Active Chapter 3 does not define all-zero JMH normalization; main baseline's
  multiclass chi-square aggregation and final hyperparameter settings need review.

## Subsequent implementation

1. Subspace-normalized MNB estimation/prediction and search-based candidate selection,
   using the verified JMH/weight/sampling helpers.
2. Consecutive weight-stability controller and full experiment configuration contracts.
3. Normalized independent-uniform model generation, exact Chapter 3 PSO, final-iteration
   fixed-G ensemble and stopping rules, after report/data validation is complete.
4. Configurable nested training/search and outer CV/repeats; primary adapted
   Chi-square + Bagging-MNB comparator, optional standard classifiers and ablations.
5. Versioned experiment records, timing/memory, per-class results, aggregate Chapter 4
   tables and figures. Final settings require scientific decisions, not guessed defaults.

Feature extraction is not the final research objective. No model accuracy or
complete AFS-RGMNB implementation is claimed by the current data audit.
