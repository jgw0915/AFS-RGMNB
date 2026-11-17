# Random-MNB/PSO implementation checkpoint

Baseline recovered on branch `refactor/full-thesis-framework` at 4a56b0a, clean.
New source commit: **`9d1ae30ffa3b37588cc8fd7ee7993ec35fd67cd9`**. Original research data and
thesis references were not modified or uploaded. The original professor report
remains unchanged.

| Component | Verified state |
|---|---|
| Random MNB initialization | Complete: positive independent-uniform block-normalized priors/theta, fixed serialization |
| PSO on selected subspace | Complete: exact synchronous fixed-inertia update, boundary redraw/reset, historical bests, exact convergence and search-accuracy fitness |
| Stage integration | One fresh PSO per preselected group; no empirical warm start; no test data in optimizer |
| Tests | 93 local tests pass with pinned dependencies, including all previous 68 |
| Real-data development execution | 20News full multiclass source; three swarms, 90 fitness evaluations; two identical seeded runs and independent saved-model fitness checks |
| Full AFS-RGMNB | Voting, full outer iteration/stopping and formal comparator/CV/Chapter4 experiments remain pending |
| New remote CI | Not run for this local source commit; no push is performed without explicit authorization |

The unchanged thesis at 669245c9 defines the method. Q07 position-map interpretation
remains explicit rather than presented as a newly discovered numbered equation.
The parameter values in the smoke config are development settings, not advisor-
approved final experiment settings. See [the numerical/real-run report](random_mnb_pso_validation.md)
and [gene requirements](../../docs/experiments/gene_experiment_requirements.md).

Continuation after interruption: inspect Git and this checkpoint first; do not
recreate completed modules. Reuse `outputs/2026-11-17/pso_smoke/run.json` and the
repeat directory. Run the 93 tests and read-only thesis checker before changes.
Research arrays and record-level outputs stay ignored. Next proposed milestone:
review and implement fixed-G voting plus the exact outer weight/stopping loop;
finish scientific dataset/protocol decisions before formal comparisons.

Final recovery verification: editable installation succeeded; all 93 tests passed
again in 3.721 seconds. The read-only checker confirmed all 12 reviewed sources
unchanged and an empty thesis working tree. Python 3.10 syntax parsing passed for
46 Python files; this is a syntax check, not a substitute for Python 3.10 CI.
`git diff --check` passed. Archive/output ignore rules were verified and neither
directory has tracked files. Independent code/report review found no material
blocker. No push or merge was performed in this continuation.
