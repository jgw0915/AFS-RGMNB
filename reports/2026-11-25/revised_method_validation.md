# Revised method validation

Two fresh processes ran the same small-budget development configuration on fold 0 of
the preserved 20News count matrix. Both scored **318/3,759 (8.4597%)** on Dtest.
Only this fold was evaluated on real data. A K-fold mean, K-fold SD and benchmark
comparison are therefore unavailable. The seed and settings were fixed before evaluation.

### Source, split and configuration

Source SHA-256 `2f18b7b11c294dda1d716988f38a4c1184c0344c1eec73ad0f510d0a887b1528`.
The source has 18,828 documents, 20 original classes, 83,635 declared columns
and 83,415 observed columns across all documents. The group-disjoint outer and
nested split retained all records. All 38 duplicate-vector groups (89 rows),
including five with conflicting labels, stayed intact. All pairwise index/group
overlap counts were zero. Exact-vector grouping cannot detect near duplicates or
undo upstream vocabulary decisions. The November 23 raw-text demo is separate.

| Partition | Rows | Classes | Role |
|---|---:|---:|---|
| Dfit | 12,049 | 20 | Define 65,540 observed columns, score features and fit candidate MNBs |
| Dsearch | 3,020 | 20 | Select one candidate per group and score particles |
| Dtest | 3,759 | 20 | Predict after construction of H |

The pinned environment used Python 3.12.7, NumPy 1.26.4, SciPy 1.11.4 and
scikit-learn 1.4.2. Both runs recorded clean code commit
`126feb8e5eb0457a74107e33463973625de05c37`. K=5, search fraction=0.2 and
split seed=42. Fold 0 derived sampling seed 2684470948 and PSO root seed
4091952314. G=3, L=4, m=128, α=1, ε=0.05, C₀=1, δq=0.01, Rq=2 and Tmax=4.
Each swarm used five particles, at most five updates, w=0.9, c₁=c₂=1.4 and
vmax=0.00001. Each position contained 20×129=2,580 coordinates.

### Measured sequence

| t | L1 weight change | Consecutive qualifying rounds | Candidate evaluations |
|---:|---:|---:|---:|
| 1 | 1.6780530895 | 0 | 12 |
| 2 | 0.5478149186 | 0 | 12 |
| 3 | 0.4651246925 | 0 | 12 |
| 4 | 0.3514265849 | 0 | 12 |

T=4 reached Tmax. The three fourth-round winners went to three fresh swarms.
There were **48** fitted candidates, **90** particle fitness evaluations
(including initialization) and **3** members in the single final ensemble.

| Final group | First four source feature IDs in draw order | SHA-256 of all 128 IDs (little-endian int64) | Swarm seed | Best Dsearch accuracy | Evaluations |
|---:|---|---|---:|---:|---:|
| 1 | 4048, 14387, 66602, 2262 | `c89e40151620af1c7ac20a8c616161011d9a29b176cf8cbcfcfc0dfa53ba6a49` | 2498773772 | 0.058278 | 30 |
| 2 | 12283, 8769, 997, 26103 | `8bced8cdfa07ebb2df08c42593563e887f0df9bb0fa4f9dc3017ca680e29e1d1` | 81010545 | 0.065563 | 30 |
| 3 | 11952, 57655, 27434, 50591 | `4d99ed6ac3a84800692a2553fd8d49b4f2ad6ff98f272d37c4dfe5e2406492f8` | 2568455293 | 0.084106 | 30 |

Each swarm stopped at its update limit. All selected source IDs in order are in
`run.json → folds[0].final_source_feature_ids`. The fitted-candidate winners,
ties and accuracies are under `weight_history`. Full outputs are private.

| Wall time in seconds | Run A | Run B |
|---|---:|---:|
| Source loading and CLI setup | 1.23567 | 1.23948 |
| Construct group-disjoint split plan for all 5 folds | 12.17630 | 12.05879 |
| Adaptive feature search including JMH | 3.58158 | 3.67700 |
| Random initialization and PSO | 0.08506 | 0.09180 |
| Construct final ensemble | 0.000017 | 0.000013 |
| Total model building | 3.66666 | 3.76881 |
| Ensemble test prediction | 0.01068 | 0.01414 |
| Fold runtime including mapping and scoring | 3.74801 | 3.86131 |
| Total through result assembly | 17.31532 | 17.31076 |

Peak process working sets were 195,842,048 and 196,509,696 bytes.
The rows include nested time intervals. No earlier full workflow was timed for comparison.

### Reproduction and checks

From the repository root with the pinned environment active:

```powershell
python scripts/run_revised_cv.py --archive './reference/107-林幸辰' --output outputs/2026-11-25/run-a
python scripts/run_revised_cv.py --archive './reference/107-林幸辰' --output outputs/2026-11-25/run-b
```

Each command requires a fresh directory. The saved JSON histories and split audits
matched across runs, including the q hashes, candidates, selections and swarm fitness
at every step. Every array in both pairs of `ensemble.npz` and `fold.npz` matched exactly.
Reloading the saved model and applying the persisted input-column map reproduced
every prediction and the original 318 correct classifications. The source IDs for
each reloaded member also matched the recorded final subspace.

The original 101 tests passed before modification. The final 108 passed with the
pinned dependencies on Python 3.10.22 (6.236 s) and Python 3.12.7 (5.831 s).
The tests include controlled convergence and reset, strict threshold equality,
real call-order and count assertions, independent random draws, hard-vote ties,
fixed-partition test-data isolation, serialization and synthetic three-fold
aggregation. The unchanged fingerprint checker passed for all 13 reviewed files.
The branch was validated locally.
