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

- Raw 20News text was not recovered in the 107 archive. New 100/104 nucleotide
  sources and archived lineages are now inspected; current rank reconciliation,
  biological grouping and the bridge to 107 count rows/features remain unresolved.
- The 107 gene class codes/names alone do not establish a consistent Family or
  Genus target. New headers require explicit schema and unresolved-label handling.
- Classes with two records cannot be present in fit/search/test simultaneously;
  full gene fivefold experiments need a reviewed target/eligibility protocol.
- Exact identical vectors now remain group-disjoint without label repair;
  near duplicates and historical upstream preprocessing still limit benchmark claims.
- Exact historical stoplist/stemmer and DNA feature-ID-to-sequence order are unknown.
- Active Chapter 3 does not define all-zero JMH normalization; main baseline's
  multiclass chi-square aggregation and final hyperparameter settings need review.

## Random MNB and PSO milestone (November 17)

- [x] Independent Uniform(0,1) initialization and block normalization, not empirical
  warm starts or Dirichlet substitution; stable M(1+m) serialization.
- [x] Exact synchronous PSO with fixed .9 inertia, scalar per-particle rhos,
  boundary redraw/reset, historical-best retention and exact-position convergence.
- [x] Q07 addition/strict pre-normalization (0,1) interpretation explicitly recorded.
- [x] One fresh PSO per selected group; sparse search-only fitness; no test scoring.
- [x] 25 new numerical/integration tests; all 93 local tests pass with pinned libraries.
- [x] Three-swarm 20News smoke and identical repeat, saved-model Dsearch checks,
  full histories, boundary counters, runtime/memory and source fingerprints.
- [x] Gene taxonomy evidence and small-class protocol requirements documented;
  full genes have 419/581 classes with fewer than five count-vector groups.
- [ ] New-code remote Python 3.10 CI for this local milestone, when publishing is authorized.

## November 25 revised workflow

- [x] Review revised Chapter 3, notation and Figure 3.1 against the compiled thesis.
- [x] Finish adaptive search before random initialization, run G independent swarms
  on the stopping round’s subspaces, construct one hard-vote model and score Dtest.
- [x] Test stage order, fit/search/test isolation, ties, convergence/reset,
  same-seed replay, serialization and full-fold aggregation.
- [x] Run and repeat one preserved 20News development fold. Record the measured
  workload, predictions and timings in the [validation report](reports/2026-11-25/revised_method_validation.md).
- [ ] Compare the separately named timestamped PDF if supplied.
- [ ] Review Q08 with the advisor before implementing the primary comparator.
  Use common fold indices, count inputs and G members.
- [ ] Agree on tuning, repeat seeds and rare-class eligibility before formal
  cross-dataset accuracy/runtime comparisons and Chapter 4 tables.

## November 26 performance diagnostics

- [x] Replay the verified fold and compare Dsearch predictions of full-feature,
  fitted-subspace, initial-particle and PSO models under the same mapping.
- [x] Complete 49 controlled executions and record all configurations, swarm
  behavior and timings. Reproduce the first phase with
  `python scripts/run_search_diagnostics.py --archive './reference/107-林幸辰' --output outputs/diagnosis/ofat`.
  The [run guide](README.md#diagnose-20news-search-accuracy) includes follow-up and reporting commands.
- [x] Prepare sparse class/term document frequencies, reproducible bootstrap
  indices and the [Q08 proposal](reports/2026-11-26/baseline_protocol_proposal.md).
- [ ] Agree on the independent validation design and comparator policies before
  using further outer folds. Compare the two [provisional candidates](reports/2026-11-26/hyperparameter_sensitivity.md)
  against fitted and equal-budget independent-random controls.

## Reference and data-storage inspection (November 17)

- [x] Protect the moved `reference/` tree and inspect both added senior folders
  read-only; recover nucleotide sources, archived ranks and historical filtering.
- [x] Audit unique FASTA/text populations, IDs, ambiguity, duplicates, full-source
  8-mer inventories and limits of current grouped allocation.
- [x] Read/render all four Week 16 PDF pages; list all 30 dataset table entries,
  distinguish binary/TF-IDF variants and check primary sources for the shortlist.
- [x] Create protected `data/raw/`, `data/extracted/` and tracked storage README.
- [x] Download official small UCI Splice source and audit integer DNA 8-mers:
  3,190 rows / 3 classes / 47,888 full-source observed features; all five current
  exact-count-vector-group-disjoint split plans pass without dropping rows/classes;
  source-ID/biological grouping still needs review; no model run.
- [ ] Review raw gene labels/accession joins and a defensible grouping/allocation
  protocol; sequence recovery does not by itself make rare-class splits feasible.
- [ ] Acquire and audit selected raw document candidates (20News, BBC, AG News)
  with Dfit-only vocabulary; source-page verification is not local validation.
