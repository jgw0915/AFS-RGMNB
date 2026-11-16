# Real 20Newsgroup pre-PSO integration

Completed 2026-11-17 Asia/Taipei; development smoke test only. Source code
`fcb68df8979f2fb5eece6cbc7130c84524dc990a` was clean when executed. All raw counts and generated
artifacts remain ignored. The professor's existing PDF/summary were not modified.

## Reproduction

Install the pinned requirements in an isolated environment, then the editable
package. In this session the environment was `outputs/ci_deps312/` (Python 3.12):

```powershell
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/run_pre_pso.py --archive '.\107-林幸辰' --output outputs/2026-11-17/pre_pso_pinned
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/run_pre_pso.py --archive '.\107-林幸辰' --output outputs/2026-11-17/pre_pso_pinned_repeat
```

Config: `configs/experiments/20news_pre_pso_smoke.json`, SHA-256
`b79046cf19b923f9a841517fb1b095d453a31ad7633fbe9f39d5ab149677f3a0`. Development settings: G=3,L=4,m=128,alpha=1,
epsilon=.05,seed=42,outer folds=5,requested inner search fraction=.2,modeled outer
fold 0. These are not finalized thesis hyperparameters.

The source fingerprint is checked before modeling:
`2f18b7b11c294dda1d716988f38a4c1184c0344c1eec73ad0f510d0a887b1528`. A mismatch fails instead of using an unreviewed
dataset. The source contains all 18,828 multiclass rows and 20 classes. Source
sample IDs, original class codes and feature IDs are preserved. Details of the
approximately stratified protocol and conflicts are in [the split audit](20news_split_audit.md).

## Pipeline and invariants

Group-aware splitting -> Dfit-only 65,540-column map -> JMH once
on fit -> initial q -> independent G x L fixed-m sequential draws -> subspace-normalized
MNB estimated on fit -> **accuracy on search** -> first best candidate per group.

Fit/search/test contain **12,049/3,020/3,759 rows**,
with all 20 classes and zero sample/group overlap. Every candidate has 128 distinct
valid local indices; duplicate sets across independent draws are allowed.

JMH finite=True, min=1.81467175597e-06, max=0.0409366180713,
sum=7.85714759492. q finite=True, min=9.82303069849e-07,
sum=1; exploration floor=.05/65540.
q SHA-256: `716cf78fc0f1e3716ccd34cb7e5f405c9a3cdd42f39a03f6f8bb5fad0ccfa2aa`.

## Candidate results (zero-based group/candidate indices)

These are development **Dsearch accuracies**, not final test accuracy or Chapter4
comparative results. No full-feature diagnostic classifier was needed.

| Group | Candidate | Correct/search rows | Accuracy | Selected |
|---|---|---:|---:|---|
| 0 | 0 | 1216/3020 | 0.402649006623 | Yes |
| 0 | 1 | 1099/3020 | 0.363907284768 |  |
| 0 | 2 | 1152/3020 | 0.381456953642 |  |
| 0 | 3 | 1036/3020 | 0.343046357616 |  |
| 1 | 0 | 1168/3020 | 0.386754966887 | Yes |
| 1 | 1 | 1165/3020 | 0.385761589404 |  |
| 1 | 2 | 1153/3020 | 0.381788079470 |  |
| 1 | 3 | 974/3020 | 0.322516556291 |  |
| 2 | 0 | 1151/3020 | 0.381125827815 | Yes |
| 2 | 1 | 1065/3020 | 0.352649006623 |  |
| 2 | 2 | 1121/3020 | 0.371192052980 |  |
| 2 | 3 | 1126/3020 | 0.372847682119 |  |

| Group | Selected candidate | All tied-best indices | Selected feature count | SHA-256 of compact JSON original source-ID list |
|---|---|---|---:|---|
| 0 | 0 | [0] | 128 | `611fd417a6d06a124f774ca218336e8a3c8f5d3d2d8826e02a2301bf829f84da` |
| 1 | 0 | [0] | 128 | `0087972338457a6f624c7c062e144ec79eada68de6b9e8da6a11f11a3bed1982` |
| 2 | 0 | [0] | 128 | `87abff15fc10a99a30bcdab6deb1518c0790f359b074ef0aaf258d56a2febbde` |

All selected maxima are unique in this run. Exact selected **local indices and
original one-based source IDs**, plus all other candidate draws, are recorded in
private `run.json` under `groups[].candidates[]`; nothing is inferred from word
names or re-tokenized. This report keeps generated feature lists out of Git.

## Measurements and repeatability

- Source loading: 1.278909 s.
- Grouping plus all five outer/nested split audits: 11.950383 s.
- Fit-only column mapping: 0.070347 s.
- Reusable fit statistics: 0.030106 s.
- JMH and initial q: 0.135492 s.
- Twelve candidate draws, fits, search evaluations and choices: 0.904211 s.
- Pipeline total excluding source loading/export: 13.096308 s.
- Process-lifetime peak RSS: 199,229,440 bytes (190.00 MiB),
  including imports, source and transformed matrices; not isolated model memory.

Runtime: Python 3.12.7, NumPy 1.26.4, SciPy 1.11.4,
sklearn 1.4.2. A second fresh process reproduced all five split
arrays/audits, fit mapping, scores, weights, candidate indices/accuracies and winners
exactly. Source/config/code hashes match, both worktrees were clean. Timing and
memory/timestamps are measurements rather than deterministic outputs. Full code
SHA-256 snapshots and package versions are in each private `run.json`.

## Scope and outputs

Private output contains `run.json`, `split_indices.npz`, `selected_fold_records.tsv`,
`fit_source_feature_ids.npy`, `jmh_scores.npy`, `sampling_weights.npy`. The API
returns reusable split, feature-partition, candidate-selection and audit objects.

**Dtest was never scored or used for candidate/parameter selection.** All five
outer split plans were audited, but candidate modeling ran only on outer fold 0;
this is not a completed fivefold performance experiment. Grouping protects only
exact vectors and cannot undo historical preprocessing leakage. No randomly
generated parameters, PSO particles, ensemble voting, outer AFS loop or final
comparative accuracy experiment was implemented.
