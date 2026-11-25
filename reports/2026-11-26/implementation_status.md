# November 26 diagnostics checkpoint

The branch `experiment/revised-method-diagnostics` started at verified origin/main
`b111a1d568f30719bf2afe25bae495e24f15f494`. The implementation and tests are in
`c4c93f1331b4f74465435161d6c1186051d849d9`. All 49 scheduled Dsearch executions
finished successfully. After committing the code, an additional seed-42 verification
reproduced every deterministic reference diagnostic, all archived final parameters
and the same 247 correct search predictions. This replay is separate from the
sensitivity inventory.

### Implemented

| Files | Responsibility |
|---|---|
| `methods/adaptive_search.py` | Optional capture of candidate draw indices in each completed round |
| `methods/pso.py`, `experiments/pso_search.py` | Optional initial position capture, velocity-repair counts and actual coordinate displacement, with accepted moves and resets distinguished |
| `experiments/diagnostics.py` | Fit/search-only controls, per-round coverage, sklearn comparison, hard-vote outcomes, equal-budget independent random search and fixed-member prefix controls |
| `experiments/baselines.py` | Fit-only sparse document frequencies by class and reproducible ordinary bootstrap row draws |
| `scripts/run_search_diagnostics.py` | Fingerprint and split verification, replay checks, matched seeds, feature reuse, trial budget, incremental per-attempt ledger and ignored outputs |
| `configs/experiments/20news_diagnostics{,_followup,_confirmation}_v1.json` | Three versioned schedules referencing the original revised development configuration |
| `scripts/summarize_search_diagnostics.py` | Generate aggregate measurement tables and the accuracy/cost and swarm-behavior figures |
| `tests/test_search_diagnostics.py` | Seven numerical and integration tests for the diagnostics, controls and baseline building blocks |

Section 3.3 still finishes before random initialization and G independent PSO
searches. The production workflow constructs one hard-vote ensemble. Additional
votes in the diagnostics compare explicit controls. Every proposed model uses the
original independent-uniform initialization, normalization, accuracy objective,
boundary handling and fixed class-order tie rule.

### Validation and artifacts

- The unmodified baseline passed 108 tests. The final suite passed **115/115 on Python
  3.10.22** (6.792 s) and **115/115 on Python 3.12.7** (6.660 s) with the pinned
  dependencies. The new-code validation was local.
- The numerical experiments used Python 3.12.7, NumPy 1.26.4, SciPy 1.11.4,
  scikit-learn 1.4.2 and one OpenBLAS thread. The three ledgers record environment,
  source commit and worktree code hashes at execution; the executions preceded the
  implementation commit. The reusable plotting script uses Matplotlib.
- Across all 49 executions, each saved confusion matrix had 20 rows and columns
  and 3,020 observations. Its trace matched the reported correct count and accuracy.
  Every full-feature and fitted-subspace prediction check agreed with scikit-learn.
  Each measured build time reconciled to its three component intervals.
- The read-only specification checker matched all 13 sources at thesis revision
  `059f00b643fd4e94ad0c8eb49b05ac2d9dd6be44`. The separately timestamped
  `main(20261125-163149).pdf` remains unavailable locally. The reviewed November
  25 compiled thesis is `../Master-Thesis/build/main.pdf`.
- Local full records are in `outputs/2026-11-26/{ofat,followup,confirmation}/`.
  Each directory has `inventory.json`, per-seed result JSON and per-selection
  feature JSON. Research outputs and read-only sources were checked against Git
  ignore rules. Historical reports through November 25 were preserved.

The original seed-42 models and partition files remain in
`outputs/2026-11-25/run-a/`. To reproduce the three schedules and figures, use
the [README commands](../../README.md#diagnose-20news-search-accuracy).
Use the [measurement inventory](measurement_tables.md) for a lookup by seed,
the [diagnosis](performance_diagnosis.md) for the controlled comparison, and the
[sensitivity report](hyperparameter_sensitivity.md) for runtime and reset behavior.

### Remaining research decisions

The results support taking the 50-particle short run and the ten-member larger
run forward as provisional settings. The first costs less and the second faces
a strong independent-random control. Freeze a training-only validation design
before drawing performance conclusions. The primary comparison still requires
advisor agreement on the multiclass chi-square formula and zero margins, feature
budget/ties, handling missing bootstrap classes, tuning and timing. The [Q08
proposal](baseline_protocol_proposal.md) records these choices. Full comparative
Chapter 4 evaluation and validation on unseen outer folds remain future work.
