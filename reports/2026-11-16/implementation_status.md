# Pre-PSO milestone implementation status

Continuation executed 2026-11-17 Asia/Taipei in the requested November 16 report
directory. Branch `refactor/full-thesis-framework`; recovered clean remote/local
`7a93149f16d2df4c630ef966fa3b60d56dbfb9ea` without discarding newer work.
Source implementation commit: **`fcb68df8979f2fb5eece6cbc7130c84524dc990a`**.

| Evidence level | Status |
|---|---|
| Mathematical functions | Prior JMH/weights/sampler preserved; eq(3.7)-(3.8) subspace MNB and eq(3.9) group selection complete |
| Synthetic numerical/integration tests | **68/68 pass**: prior 41 + splitting 9 + MNB 11 + selection 4 + full pipeline 3 |
| Real dataset execution | Complete 20News preserved-count load, all five group split audits, fit-only mapping and twelve modeled candidates in outer fold 0; two identical seeded runs |
| Full AFS-RGMNB | Pending and outside this milestone: random parameter generation, PSO, voting and full outer iteration |
| Formal classification performance | Not completed; reported candidate accuracy is Dsearch debugging evidence, no Dtest metric |
| New-code Python 3.10 GitHub Actions | **Passed: 68 tests**, workflow run44007208985 on d3a44f84a9b3e63c087e2a76bed111a3d2f882c7 |

## Implemented APIs

- `data.splitting`: `group_count_vectors`, `group_aware_splits`,
  `prepare_feature_partitions`; `SplitPlan`, `GroupSplit`, `FeaturePartitions`.
- `methods.mnb`: `class_statistics`, `estimate_parameters`, `MNBParameters`,
  `class_log_scores`, `SubspaceMNB.fit/from_statistics/from_parameters/predict`.
- `methods.random_subspace`: existing sampler plus `select_subspaces` returning
  `CandidateSelection`, `SelectedGroup`, `CandidateResult`.
- `experiments.pre_pso`: `PrePSOConfig`, `run_pre_pso`, `save_pre_pso`.
- Runner/config: `scripts/run_pre_pso.py`,
  `configs/experiments/20news_pre_pso_smoke.json`.

## Validation records

Required editable installation and full unittest commands passed. Local Python 3.12
current-library suite and isolated pinned requirements suite both pass all 68 tests;
latest pinned run took 2.952 s. Pinned versions are NumPy 1.26.4, SciPy 1.11.4,
sklearn 1.4.2, NLTK 3.9.2. The local run is distinct from remote Python 3.10
verification. Following explicit user approval on November 17, both milestone
commits were pushed. The [new Python 3.10 workflow](https://github.com/jgw0915/AFS-RGMNB/actions/runs/44007208985)
completed successfully for `d3a44f84a9b3e63c087e2a76bed111a3d2f882c7`.
Job133785317546 verified checkout, Python setup, pinned requirements, editable
installation and unittest execution. Its log records **68 tests in2.125s, OK**.
This completes the outstanding Python 3.10 milestone check; the real-data smoke
was executed locally under the separately recorded pinned Python 3.12 environment.

`tools/check_thesis_spec.py` confirms all 12 reviewed files match thesis
`669245c9be4073e682ae76d6b0bdf80cd724d89b`; thesis Git status is clean. Actual
entry is `../Master-Thesis/main.tex` -> `Chapter/Third Chapter.tex` plus active
notation. No historical note changed the algorithm. `git diff --check` passed.
Archive, matrices, IDs and heavy artifacts are ignored; professor PDF/summary
were preserved unchanged.

Reports: [split audit](20news_split_audit.md),
[MNB numerical validation](subspace_mnb_validation.md),
[real integration](pre_pso_integration.md). Source fingerprints are verified and
recorded; the real run used a clean source commit and explicit file hashes.

## Handoff and next milestone

No data or scientific blocker prevents the requested pre-PSO flow. Remaining
limitations are historical upstream preprocessing, unobserved raw/near duplicates
and non-final development hyperparameters. The requested pre-PSO milestone is
complete, including remote CI. This push was explicitly authorized by the user;
future unrelated pushes remain subject to AGENTS.md.

To continue after interruption: inspect Git first, retain commits/outputs, read
this file, run the 68 tests and thesis checker, then inspect existing
`outputs/2026-11-17/pre_pso_pinned/run.json` before rerunning. Use the pinned
libraries for reproducing its split/candidate figures. The next proposed task is
reviewing normalized-uniform MNB parameter initialization and its constraints
against the current thesis, under a separately scoped milestone; it has not begun.
