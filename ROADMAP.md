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
- [x] Review and checkpoint report/data changes locally (historical checkpoint).

## Pre-PSO milestone (continued November 17)

- [x] Stable canonical sparse-vector groups, including conflicting-label groups.
- [x] Group-disjoint outer/nested SGKF; all classes/rows retained; realized class
  deviations, source IDs, group/sample overlaps and all five fold plans audited.
- [x] Dfit-only observed columns and consistent original-feature-ID mapping.
- [x] Subspace-normalized MNB (3.7)-(3.8), empirical priors and reusable scoring.
- [x] G×L candidates, once-per-fit JMH, Dsearch accuracy and deterministic first ties.
- [x] Synthetic end-to-end coverage; **68 tests pass**, including all previous41.
- [x] Full20News smoke and repeat with pinned dependencies:12 models,3 selected
  subspaces; test unscored; full logs/artifacts local and ignored.
- [x] Four requested derived reports in `reports/2026-11-16/`; professor report unchanged.
- [x] User-authorized branch push; Python 3.10 GitHub Actions passed all68tests
  on d3a44f8 ([run44007208985](https://github.com/jgw0915/AFS-RGMNB/actions/runs/44007208985)).

## Scientific or source blocks

- Raw 20News text and gene sequences/accessions/lineages were not recovered.
- Gene class codes/names do not establish a consistent Family or Genus target.
- Classes with two records cannot be present in fit/search/test simultaneously;
  full gene fivefold experiments need a reviewed target/eligibility protocol.
- Exact identical vectors now remain group-disjoint without label repair;
  near duplicates and historical upstream preprocessing still limit benchmark claims.
- Exact historical stoplist/stemmer and DNA feature-ID-to-sequence order are unknown.
- Active Chapter 3 does not define all-zero JMH normalization; main baseline's
  multiclass chi-square aggregation and final hyperparameter settings need review.

## Subsequent implementation

1. Separately scope normalized independent-uniform model generation and numerical
   constraints against the unchanged active Chapter3. No initialization is implemented yet.
2. Exact Chapter3 PSO and its constraints/stopping, in a subsequent milestone.
3. Fixed-G voting, consecutive weight stability and final-iteration AFS loop.
4. Formal outer CV/repeats; primary adapted
   Chi-square + Bagging-MNB comparator, optional standard classifiers and ablations.
5. Versioned experiment records, timing/memory, per-class results, aggregate Chapter 4
   tables and figures. Final settings require scientific decisions, not guessed defaults.

Feature extraction is not the final research objective. Development Dsearch
candidate accuracies are recorded; formal Dtest accuracy, full AFS-RGMNB and
Chapter4 comparative results remain future work.
