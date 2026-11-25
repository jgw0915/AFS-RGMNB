# Implementation checkpoint for the November 25 revision

Branch `refactor/revised-chapter3-workflow-20261125` starts from the verified
main/origin HEAD `e76d445590f014d3102ac1dbd32b8680472269f4`.
The executable and initial numerical checks are in
`126feb8e5eb0457a74107e33463973625de05c37`. Additional random-draw and
held-out isolation checks are in `047dd23cfd847833618ddca0689374932c5e01c4`.
The two preserved-count runs used the executable commit.

| Deliverable | Evidence |
|---|---|
| Revised method | Equations 3.1–3.20 and Fig. 3.1 mapped to the fold runner and numerical components |
| Numerical and integration checks | 101 baseline tests retained, seven new tests, 108 passed on both Python 3.10.22 and 3.12.7 |
| Real-data check | One repeated fold, all 20 classes, T=4, 48 candidates, 3 independent swarms, 1 ensemble, 318/3,759 correct |
| Specification synchronization | Four changed inputs reviewed. The checker passes on 13 sources at thesis revision `059f00b6`, including the compiled PDF. The timestamped export is outstanding. |
| Formal experiments | Revised K-fold runner available. Full real-data CV, approved comparison policy, multi-dataset results and Chapter 4 tables remain pending. |

### Changed files

| Directory | Files |
|---|---|
| Repository root | `AGENTS.md`, `README.md`, `ROADMAP.md` |
| `src/afs_rgmnb/methods/` | `adaptive_sampling.py`, `adaptive_search.py`, `ensemble.py`, `pso.py`, `random_generation.py`, `random_subspace.py` |
| `src/afs_rgmnb/experiments/` | `pso_search.py`, `runner.py` |
| `configs/experiments/` | `20news_revised_dev.json` |
| `scripts/` | `run_revised_cv.py` |
| `tests/` | `test_random_generation.py`, `test_revised_workflow.py` |
| `docs/methodology/` | `chapter3_implementation_mapping.md`, `open_questions.md`, `specification_status.md`, `thesis_source_fingerprints.json` |
| `docs/experiments/` | `revised_workflow.md` |
| `reports/2026-11-25/` | `chapter3_workflow_migration.md`, `revised_method_validation.md`, `implementation_status.md` |

### Continue from this checkpoint

1. Compare the separately named November 25 export to the checked-in sibling build when available.
2. Resolve Q08 for the primary Chi-square + Bagging-MNB comparator, including score aggregation,
   zero margins, fixed-feature cutoff/ties and bootstrap missing-class behavior. Reuse the same
   split indices, count representation and ensemble size.
3. Agree on final settings, repetitions and the biological grouping/eligibility protocol.
   Keep tuning inside training partitions. This development fold has already been inspected.

The [validation report](revised_method_validation.md) gives measurements and artifact
locations. The [experiment guide](../../docs/experiments/revised_workflow.md) gives the
full-CV command and output schema. The earlier advisor reports and raw-text demo remain
available at their original paths.
