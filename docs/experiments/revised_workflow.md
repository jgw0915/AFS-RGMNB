# Revised Chapter 3 experiment

The driver reads the senior’s preserved 20News integer-count matrix. It verifies
the source fingerprint before splitting. Raw 20 Newsgroups messages and the
November 23 extraction demo use a separate reader and protocol.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/run_revised_cv.py --config configs/experiments/20news_revised_dev.json --archive './reference/107-林幸辰' --output outputs/revised-fold0
```

Choose another empty directory to repeat. Add `--all-folds` to evaluate all the
outer folds in the same plan. The reusable `run_cv` API also accepts other
in-memory sparse count datasets, aligned labels, sample IDs and source feature IDs.

| Configuration | Interpretation |
|---|---|
| `cv.n_splits, search_fraction, seed, folds` | Group-disjoint stratified allocation, requested search share of outer training data, root seed and zero-based fold indices |
| `adaptive.n_groups, candidates_per_group, subspace_size` | G independent groups of L draws per round, each containing m distinct fit-observed columns |
| `alpha, epsilon, c0` | Positive MNB smoothing, uniform mixture fraction in (0,1), positive inclusion threshold constant |
| `delta_q, rq, tmax` | Inclusive L1 tolerance in (0,2), consecutive qualifying rounds and iteration limit ≥ rq |
| `pso.n_particles, max_iterations, c1, c2, vmax` | Swarm size, maximum updates after initialization, two positive learning coefficients and speed bound. Inertia is fixed at 0.9. |

Each fold derives separate sampling and PSO seeds from the root seed and fold
index. A single sampling generator advances across all rounds. The PSO stage
spawns one independent stream per final group. Fold selection or execution order
therefore preserves the result for a given fold under the same library versions.

### Artifacts

- `run.json` records the source and configuration hashes, library versions, code revision,
  realized class and group counts, overlap audit, seeds, per-round weights digest and
  candidate accuracies, final source IDs, per-swarm fitness histories, accuracy and elapsed seconds.
- `fold_NN/fold.npz` holds partition row indices into the fingerprinted source, fit columns,
  source feature IDs, initial and terminal updated weights, and held-out predictions.
- `fold_NN/ensemble.npz` stores class order and the historical-best parameters and
  fit-column indices of each member. Reload with `VotingEnsemble.load(path)` and
  predict on `X_new[:, fit_columns]` using the original source dictionary and column order.

Model-building time spans JMH initialization through ensemble construction.
Prediction is timed separately. Fold runtime starts before fit-column mapping and ends after
accuracy calculation. CV runtime includes the common split plan, fold processing and artifact writes.
The CLI also records elapsed time from source loading through result assembly, before writing the final JSON.
Mean and population SD (ddof=0) summarize the K equally weighted fold accuracies only
after all K folds have run. Each development fold retains its own accuracy.

### Primary comparison

Section 3.5 specifies one fixed chi-square-selected subspace, bootstrap-fitted MNB
members, majority voting, the same occurrence counts, splits and G. Saved fold indices
and fit-column maps can be reused for this comparison. The multiclass contingency-score
aggregation, zero margins, feature cutoff/ties and missing classes in a bootstrap sample
require a reviewed policy before benchmarking. See [Q08](../methodology/open_questions.md).
