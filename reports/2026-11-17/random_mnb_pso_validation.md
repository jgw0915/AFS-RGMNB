# Thesis random MNB initialization and PSO validation

Executed 2026-11-17. Clean implementation source commit:
`9d1ae30ffa3b37588cc8fd7ee7993ec35fd67cd9`. This extends the verified pre-PSO stages; it does not
implement voting, the full adaptive outer loop or Chapter 4 comparisons.
The existing professor-facing reports and PDF remain unchanged.

## Active thesis correspondence

Read-only authority: `../Master-Thesis/main.tex` -> `Chapter/Third Chapter.tex`,
revision `669245c9be4073e682ae76d6b0bdf80cd724d89b`.

| Source | Implemented behavior |
|---|---|
| (3.10)-(3.11) | Independent U_y,V_yj Uniform(0,1); normalize priors across classes and conditionals within each class's selected features |
| (3.12) | Fixed prior-first, then row-major theta representation; M(1+m) coordinates |
| (3.13) | Personal best changes only on strictly greater search accuracy; ties keep old positions |
| (3.14) | Global best chosen from all personal bests; first particle resolves score ties |
| (3.15) | Synchronous .9*v+c1*rho1*(pbest-z)+c2*rho2*(gbest-z), with two scalar rho draws per particle/update |
| Section 3.4.2 prose | Out-of-bound velocity coordinates redrawn; invalid raw position resets the whole particle and velocity; valid positions block-normalized; histories retained |
| Section 3.4.2 prose | Exact equality of all positions or Imax updates stops search; fitness equality alone does not stop it |

The active prose does not supply a separate explicit position-addition equation
or numeric pre-normalization interval. Implementation follows the previously
recorded Q07 interpretation: `z_next=z+v`, raw coordinates strictly in (0,1),
otherwise whole-particle reset. This interpretation is explicit in module docs
and mapping; no older note supplies a replacement algorithm or thesis edit.
Convergence is checked at initialization and after updates, so Np=1 stops at
initialization; meaningful swarm settings use multiple particles.

Zero uniform draws are redrawn, not clipped to a minimum probability. Initial
random priors do not use empirical class counts. Normalized independent uniforms
are not replaced by a Dirichlet simplex law. Speed endpoints +/-vmax are valid.
Inertia is fixed at .9 and not configurable. No clipping, reflection, softmax,
coordinate-wise rho, asynchronous history update or fitted-candidate warm start
is substituted for the specified procedure.

## APIs and tests

- `methods/random_generation.py`: `generate_parameters`, `normalize_position`,
  `parameters_to_position`, `position_to_parameters`.
- `methods/pso.py`: `PSOConfig`, `PSOIteration`, `PSOResult`, `optimize_mnb`.
- `experiments/pso_search.py`: `optimize_selected_subspaces`, `run_pso_pipeline`,
  `save_pso_pipeline`, `GroupModelSearch`.
- `scripts/run_pso.py` and `configs/experiments/20news_pso_smoke.json` provide
  the reproducible real-count development command.

**93 tests pass locally** under pinned requirements on Python 3.12.7:
68 earlier tests + 9 initialization tests + 13 PSO numerical tests + 3 integrated
pipeline tests. They cover hand uniform draws, independent block normalization,
serialization, scalar-rho arithmetic, synchronous old histories, coordinate speed
redraw, whole reset/history retention, strict ties, exact versus approximate
convergence, evaluation budgets, sparse operation and fixed-seed repeatability.
Integrated spies verify that only X_search/y_search reach one fresh swarm per
group; model files round-trip with no pickle. New-code remote Python 3.10 CI has
not been run for this local commit; the prior 68-test workflow is separate.

After interruption recovery, editable installation and the full 93-test suite
passed again (3.721s). All 12 reviewed thesis files still match their fingerprints,
and the thesis working tree is clean. All 46 Python source/script/tool/test files
also parse with Python 3.10 syntax; actual new-code Python 3.10 execution remains
an outstanding CI check.

## Real 20News development run

Source SHA-256: `2f18b7b11c294dda1d716988f38a4c1184c0344c1eec73ad0f510d0a887b1528`. Original 18,828 rows and
20 classes retained, with the existing group-disjoint fivefold plan. Only outer
fold 0 was modeled: fit/search/test=12,049/3,020/3,759; fit-observed d=65,540.
Zero sample/group overlap; Dfit sets columns/JMH and screens G x L subspaces;
Dsearch selects subspaces and supplies PSO accuracy. Dtest remains unscored.

After the existing G=3,L=4,m=128,alpha=1,epsilon=.05,seed=42 screening, each
selected subspace gets a fresh Np=5 particle swarm, Imax=5 updates,
c1=c2=1.4,vmax=1e-5,w=.9. These are small development values, not finalized
thesis hyperparameters. Root PSO seed 42 creates independent SeedSequence children
in group order. Each particle has 20*(1+128)=2,580 coordinates.

| Group | Child seed | Initial best search accuracy | Historical best search accuracy | Fitness evaluations | Speed coordinates redrawn | Whole particle resets | Stop reason |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 2684470948 | 0.072847682119 | 0.072847682119 | 30 | 51,400 | 4 | max_iterations |
| 1 | 4091952314 | 0.063576158940 | 0.078145695364 | 30 | 59,220 | 8 | max_iterations |
| 2 | 233227757 | 0.065894039735 | 0.065894039735 | 30 | 58,785 | 8 | max_iterations |

All three swarms reached the five-update limit; each evaluated 5*(1+5)=30 models,
90 total. Best-so-far accuracy never decreases. The random-search results are
small-budget diagnostics, not final classification performance. The fitted MNB
models used to screen feature sets were not injected into the swarms; after
selection the parameters start from the thesis random law. These smoke results
do not establish an advantage over the fitted screening models. Later budget
and hyperparameter studies must use training/search validation, not Dtest.

The speed-redraw and particle-reset counters are recorded to expose how the
strict boundary policy behaves in 2,580 dimensions. They must not be hidden or
replaced with a different boundary map to improve a reported score. All saved
priors/conditionals are finite and positive, with block sums within float64
roundoff (maximum conditional-row sum error 2.22e-16).

## Reproducibility, runtime and artifacts

Two fresh pinned-environment runs reproduce the group splits, fit mapping,
JMH/q, screened candidates, group seeds, all particle-score histories, boundary
counters and saved best parameter arrays exactly. Each saved model was also
restored independently and its Dsearch accuracy matched the reported best.
No test score was computed in that verification. Both runs used clean source
`9d1ae30ffa3b37588cc8fd7ee7993ec35fd67cd9`; current source-file hashes match their captured SHA-256
snapshot. Timing/memory values are measurements, not deterministic outputs.

Environment: Python 3.12.7, NumPy 1.26.4, SciPy 1.11.4,
scikit-learn 1.4.2.

- Source load: 1.250499s.
- Existing split/JMH/candidate pipeline: 11.733169s.
- Three PSO searches including model/report extraction: 0.094216s.
- Pipeline including PSO, excluding source load/export: 11.827716s.
- Peak process RSS including imports/data: 198,352,896 bytes
  (189.16 MiB).

Reproduce using an environment installed from `requirements.txt`:

```powershell
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/run_pso.py --archive '.\107-林幸辰' --output outputs/2026-11-17/pso_smoke
& '.\outputs\ci_deps312\Scripts\python.exe' scripts/run_pso.py --archive '.\107-林幸辰' --output outputs/2026-11-17/pso_smoke_repeat
```

The ignored directories contain run.json, split arrays, source feature mapping,
JMH/q arrays, source-row IDs/labels, and group_00/01/02_best_parameters.npz.
The CLI verifies the source fingerprint and protects the read-only archive and
thesis; in-repository research outputs must be ignored. No raw data, model arrays
or historical source programs are committed with this derived report.

## Remaining thesis work

Fixed-G voting, the complete adaptive weight/stopping outer loop, the finalized
baseline/tuning protocols, repeated full-CV/test evaluation and Chapter 4 result
aggregation remain pending. The gene dataset prerequisites are recorded in
[gene_experiment_requirements.md](../../docs/experiments/gene_experiment_requirements.md).
