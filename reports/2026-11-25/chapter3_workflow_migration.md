# Chapter 3 workflow migration

The November 25 revision moves feature-weight updating and its stopping decision
ahead of random MNB generation. The retained object at iteration T is a set of G
feature subspaces. One ensemble H is constructed after the G searches finish.

| Stage | Earlier reviewed specification | November 25 revision and implementation |
|---|---|---|
| Fit statistics, JMH, initial q | Once per fold | Retained, equations 3.1–3.5. Fit-only statistics are reused across candidate rounds. |
| Sample, fit and select | G×L fitted models per round | Retained, equations 3.6–3.9. A persistent generator draws from the current q. The first search-accuracy winner is retained per group. |
| Update q, test stability | Equations 3.17–3.20, after per-round model search/voting | Equations 3.10–3.13, at the end of each feature-only round. The strict inclusion threshold, normalization and L1 distance are unchanged. |
| Random initialization and PSO | Equations 3.10–3.15, G searches per round | Equations 3.14–3.19, G searches after stopping. The candidate fit estimates have no role in initialization. |
| Ensemble and evaluation | Equation 3.16, per-round ensemble and final H_T | Equation 3.20, one final hard-vote H. Section 3.5 evaluates it on Dtest. |

For T completed rounds, both designs specify T×G×L fitted-candidate evaluations.
The theoretical number of PSO searches drops from T×G to G. For T=4 and G=3
this changes 12 searches to 3. The earlier executable was a one-round diagnostic.
There is no paired measured runtime comparison between the two full designs.

### Source review

Earlier thesis commit `669245c9be4073e682ae76d6b0bdf80cd724d89b` and
revised commit `059f00b643fd4e94ad0c8eb49b05ac2d9dd6be44` have the same
main inclusion graph. The initial fingerprint check identified changes in
`Chapter/Third Chapter.tex`, `Chapter/chapter3-notation-table.tex` and
`figures/chapter3-method.{drawio,pdf}`. The revised figure sends the continuation
branch to sampling and the stopping branch to PSO. Search data enter both
candidate selection and swarm fitness. Test data enter the evaluation box.

The compiled sibling `build/main.pdf` contains the revised flow on printed page 17
and equations 3.1–3.20 on pages 19–26. The requested timestamped export
`main(20261125-163149).pdf` was unavailable for an independent comparison.
The manifest now also checks the compiled PDF. Exact paths, hashes and the
equation-by-equation source labels are in the [specification status](../../docs/methodology/specification_status.md)
and [implementation mapping](../../docs/methodology/chapter3_implementation_mapping.md).

### Code and scientific decisions

| Module | Responsibility after migration |
|---|---|
| `methods/random_subspace.py` | Shared single-round selection from specified q, fit statistics and generator. The old one-round wrapper remains callable. |
| `methods/adaptive_sampling.py`, `methods/adaptive_search.py` | Compute inclusion fractions, threshold and normalize, update/reset the stability counter, return stopping-round indices. |
| `methods/random_generation.py`, `methods/pso.py`, `experiments/pso_search.py` | Reuse existing probability generation and synchronous swarm search. Equation references and diagnostic descriptions were updated. |
| `methods/ensemble.py` | Equal label votes in fixed class order. Saves and reloads the subspace and probability blocks of every member. |
| `experiments/runner.py`, `scripts/run_revised_cv.py` | Build H from fit/search, predict on held-out rows, aggregate a complete fold set and persist artifacts. |

Q01 still rejects an undefined all-zero JMH normalization. Q07 still applies
position addition, strict open-interval checks before normalization and whole-particle
reset. The revised text gives no new explicit position-map equation. The current
split protocol groups exact count vectors, requires full class coverage and learns
the column map on Dfit. The primary comparator, gene-task eligibility and final
study settings await the decisions recorded in [open questions](../../docs/methodology/open_questions.md).
