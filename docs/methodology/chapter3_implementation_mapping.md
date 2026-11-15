# Chapter 3 implementation mapping

Reviewed 2026-11-16 against thesis Git revision `669245c9be4073e682ae76d6b0bdf80cd724d89b`.
Authoritative entry point: `../Master-Thesis/main.tex`. Its active order is
`Chapter/First Chapter.tex`, `Chapter/Second Chapter.tex`, `Chapter/Third Chapter.tex`.
Chapter 3 includes `Chapter/chapter3-notation-table.tex` and
`figures/chapter3-method.pdf` (editable source `figures/chapter3-method.drawio`).
The standalone `notes/chapter3-notation.tex` is **not** included by `main.tex` and
contains superseded designs. Labels below are stable references; rendered equation
numbers (3.1–3.20) are derived from the active source order and should be rechecked
after revisions. Line numbers refer to the reviewed source, not an older PDF.

## Scope and data contracts

`X_fit`, `X_search`, `X_test` are sparse CSR matrices with identical feature-column
order and finite nonnegative integer occurrence counts. `y_*` are one-dimensional
class labels aligned with rows. A persisted class order maps thesis classes
`1,...,M` to Python positions `0,...,M-1`; a persisted vocabulary maps thesis
features `1,...,d` to Python columns `0,...,d-1`. The feature list is fitted only
on `D_fit` as a conservative, explicit implementation policy. The active body
states the split purposes but omits vocabulary-learning scope; earlier supporting
notes alternate between `D_fit` and outer `D_train`. No test vocabulary learning
is permitted. See `open_questions.md` for this distinction.

The full flow is: split → learn vocabulary on fit → count features → calculate
JMH once → initialize weights → generate `G × L` subspaces → estimate and compare
fit-based MNB models on search → keep one subspace per group → run one fresh PSO
per selected subspace → construct `G`-member hard-vote ensemble → update feature
weights from selected subspaces → check consecutive weight stability/iteration
limit → retain the **last completed ensemble** → evaluate test once. PSO never
directly optimizes feature-sampling weights. There is no `D_select`, candidate
acceptance threshold, ensemble accumulation, or best-outer-iteration selection.

## Symbols and Python representations

| Thesis notation | Python meaning/shape |
|---|---|
| `x_ij`, `y_i` | `X[i,j]`, `y[i]`; counts and class labels |
| `d`, `M`, `K` | `n_features`, `n_classes`, `n_splits`; do not use the older Ch2 meanings of M/K |
| `N_yj`, `n_y`, `n_fit` | `feature_counts[M,d]`, `class_counts[M]`, number of fit rows |
| `p_hat_yj`, `pi_hat_y` | `feature_prob[M,d]`, `class_prior[M]` estimated on fit |
| `p_bar_yj`, `s_j` | `complement_prob[M,d]`, `scores[d]` |
| `q^(t)` | `sampling_weights[d]`, strictly positive and normalized |
| `G`, `L`, `m` | groups/ensemble members, candidates per group, fixed subspace size |
| `S_tgℓ`, `S*_tg` | unique integer column arrays, shape `[m]`; `G × L` candidates, `G` selected |
| `pi^(b)`, `theta^(b)` | one model's positive `class_prior[M]`, `feature_prob[M,m]` |
| `N_p`, `z_a`, `v_a` | particle count; positions/velocities `[N_p, M(1+m)]` |
| `p_a`, `p_*` | copied personal/global best positions; scores stored separately |
| `rho_1,a`, `rho_2,a` | two independent scalar draws per particle per inner iteration |
| `r_j`, `r_tilde_j` | inclusion fractions and thresholded values, arrays `[d]` |
| `Delta`, `delta_q`, `R_q`, `T_max` | L1 weight change, tolerance, consecutive-stability target, outer limit |
| `I_max` | maximum number of PSO position updates, excluding initialization |

## Equation and stage traceability

Status at mapping creation: **specified / implementation planned**. The module
paths are required interfaces, not assertions that tests have already passed.
The implementation status section below is updated only after code verification.

| Section / source line | Identifier (rendered number) | Mathematical meaning and I/O | Python component | Hyperparameters | Testable properties | Status / ambiguity |
|---|---|---|---|---|---|---|
| 3.1, 7–24 | `sec:pso-framework`, `fig:chapter3-main-flow`, `tab:method-notation` | Full ordered flow; fit/search/test to final fixed-size ensemble and test metrics | `methods/ensemble.py`, `experiments/runner.py` | all method and split settings | Each outer iteration rebuilds G members; final ensemble is H_T; test is absent from fit API | Specified; figure simplifies the detailed prose (see below) |
| 3.2.1, 33–42 | `sec:method-extraction` | Documents → word occurrence counts; DNA → overlapping length-8 windows with stride 1 → count matrix | `features/dna_kmer.py`, `features/document.py`, `features/vectorization.py` | DNA k=8 fixed canonically; document tokenizer/stopwords explicit | Repeated windows retained, no normalization/TF-IDF, nonnegative integers, stable columns | Count semantics verified; ambiguous-base, U/T, stopword/stemming details not specified in body |
| 3.2.2, 44–49 | `sec:method-data-split` | Stratified K-fold test; each outer train is split into disjoint fit/search; only search scores candidates and PSO | `data/splitting.py`, `experiments/runner.py` | K≥2; fit/search ratio; seed | Disjoint indices; outer test covers each row once per repetition; all classes in fit/search; fit transformations do not see test | Specified split structure; no final K or ratio in active body |
| 3.3.1, 61–68 | `eq:method-md-estimates` (3.1) | `p_hat_yj=(N_yj+alpha)/(sum_k N_yk+d*alpha)`, `pi_hat_y=n_y/n_fit`; fit counts → class probabilities | `methods/feature_scoring.py`, `methods/mnb.py` | alpha>0 | Row sums=1; positive probabilities; empirical priors; occurrence totals (not presence counts) | Verified equation |
| 3.3.1, 69–75 | `eq:method-md-complement` (3.2) | `p_bar_yj=sum_(y'≠y)[pi_hat_y'/(1-pi_hat_y)]p_hat_y'j` | `methods/feature_scoring.py` | none | Mixture excludes target class; weights sum=1; ≥2 represented classes | Verified; reject one-class training |
| 3.3.1, 76–89 | `eq:method-jmh-score` (3.3) | `s_j=sum_y[p_hat*log(p_hat/p_bar)+(1-p_hat)*log((1-p_hat)/(1-p_bar))]`; probabilities → length-d scores | `methods/feature_scoring.py` | same alpha | s≥0 (floating roundoff considered); equal class feature probabilities give zero; two classes reduce to Jeffreys divergence | Verified; outer sum is NOT weighted again by priors; Bernoulli event split is not binarized input |
| 3.3.1, 91–98 | `eq:method-initial-sampling` (3.4) | `q_j=(1-epsilon)*s_j/sum_k s_k+epsilon/d` | `methods/adaptive_sampling.py` | 0<epsilon<1 | sums=1; q≥epsilon/d; larger score gives larger q | Verified when sum(s)>0; all-zero score rule missing in active equation |
| 3.3.1, 99–103 | `eq:method-sampling-weights` (3.5) | invariant `q_j>0`, `sum_j q_j=1` | `methods/adaptive_sampling.py`, `methods/random_subspace.py` | none | Reject negative, nonfinite, zero-mass weights; all produced distributions valid | Verified invariant |
| 3.3.2, 105–116 | `eq:method-sequential-sampling` (3.6) | From remaining R, draw j with `q_j/sum_(k∈R)q_k`, remove j; repeat m times independently for each candidate | `methods/random_subspace.py` | fixed 2≤m≤d, G≥1, L≥1, RNG seed | Exactly m unique valid columns; sequential conditional probabilities; separate candidates may duplicate | Verified; generic Bernoulli selection or variable m is not equivalent |
| 3.3.3, 118–132 | `eq:method-subspace-mnb-estimates` (3.7) | `theta_tilde_yj=(N_yj+alpha)/(sum_(k∈S)N_yk+m*alpha)`; empirical priors reused | `methods/mnb.py` | alpha>0, selected S | Per-class sum over selected m features=1; renormalize within subspace; sparse fit counts | Verified |
| 3.3.3, 135–143 | `eq:method-subspace-mnb-prediction` (3.8) | `argmax_y[log(pi_hat_y)+sum_(j∈S)x_j log(theta_tilde_yj)]` | `methods/mnb.py` | fixed class order | Sparse/dense agreement; all-zero subspace predicts maximal prior; no multinomial coefficient needed for argmax | Verified; candidate prediction ties use documented stable class order |
| 3.3.4, 145–157 | `eq:method-subspace-choice` (3.9) | For each group, choose candidate maximizing classification accuracy on search; emit G selected subspaces | `methods/random_subspace.py`, `methods/ensemble.py` | G,L,m | Exactly one per group; accuracy only on search; deterministic first-candidate tie policy | Verified; source does not specify candidate tie order |
| 3.4.1, 164–176 | `eq:method-mnb-constraints` (3.10) | Positive priors across M classes sum=1; each conditional row across m selected features sums=1 | `methods/mnb.py`, `methods/random_generation.py` | M≥2,m≥2 | Strict positivity and block sums, finite logs | Verified |
| 3.4.1, 177–184 | `eq:method-mnb-initialization` (3.11) | Independent U,V∼Uniform(0,1); normalize U across classes and V within each class | `methods/random_generation.py` | RNG seed, M,m,N_p | Deterministic fixed seed; valid blocks; separate particle draws | Verified; NOT Dirichlet(1), not uniform on simplex; no fit-derived prior in random candidates |
| 3.4.2, 196–206 | `eq:method-pso-particle-position` (3.12) | One position encodes pi and theta, M(1+m) coordinates; velocities initialize Uniform[-vmax,vmax] | `methods/random_generation.py`, `methods/pso.py` | N_p, vmax>0 | Pack/unpack round trip; class/feature ordering stable | Verified representation dimension; exact flat layout is routine implementation choice |
| 3.4.2, 208 | unnumbered `F_S(z)=A(h_(S,z);D_search)` | Decode particle and compute search classification accuracy | `methods/pso.py`, `methods/mnb.py` | none | Fitness∈[0,1]; no fit/test substitution; same class order | Verified objective; macro-F1 must not replace it |
| 3.4.2, 210–220 | `eq:method-pso-personal-best` (3.13) | Initialize pbest=z0; update pbest only on strict fitness improvement; ties retain prior | `methods/pso.py` | none | pbest score nondecreasing; equal score retains copied old position | Verified |
| 3.4.2, 221–227 | `eq:method-pso-global-best` (3.14) | Maximize over personal best positions, retaining a historically best model | `methods/pso.py` | none | gbest score never decreases; copies survive particle reset | Verified; equal-global-score tie resolved by stable particle order |
| 3.4.2, 229–244 | `eq:method-pso-velocity` (3.15) | `v_tilde=.9*v+c1*rho1*(pbest-z)+c2*rho2*(gbest-z)`; rho values are scalars per particle/iteration | `methods/pso.py` | fixed w=.9; c1,c2>0; vmax>0 | Hand-calculated update; synchronous use of previous whole-swarm histories; no decreasing inertia | Verified |
| 3.4.2, 245–249 | unnumbered bounds, normalization and stopping prose | Redraw each out-of-range speed coordinate; add speed to position; invalid position resets whole particle and speed; otherwise block-normalize; evaluate all then update histories; stop identical positions or Imax updates | `methods/pso.py` | Imax≥1,vmax,N_p | No clipping/softmax; all coordinates valid after repair; histories preserved; score equality is insufficient to stop | Prose specified; `(0,1)` boundary/addition interpretation documented in open questions |
| 3.4.2, 251–265 | `sec:method-ensemble`, `eq:method-vote` (3.16) | Fresh PSO per selected group; G members; equal hard vote; fixed class-order tie | `methods/ensemble.py` | G,class order | Fixed G; no member carryover, probability averaging or model weighting | Verified; label is within 3.4.2, not a separate subsection |
| 3.5, 270–278 | `eq:method-inclusion` (3.17) | `r_j=(1/G)sum_g 1[j∈S*_tg]` | `methods/adaptive_sampling.py` | G,m | r∈[0,1]; sum(r)=m for fixed-m subspaces; duplicates across groups counted | Verified; count selected subspaces, not PSO accuracy or parameter magnitudes |
| 3.5, 279–290 | `eq:method-rase-threshold` (3.18) | retain r if `r>C0/log(d)`; else replace with `C0/d` | `methods/adaptive_sampling.py` | C0>0 | Equality uses replacement branch; all values positive; C0≥log(d) gives uniform after normalization | Verified; no old-weight/JMH mixture |
| 3.5, 291–297 | `eq:method-rase-weight-update` (3.19) | Normalize thresholded inclusion values to q_next | `methods/adaptive_sampling.py` | C0 | q_next positive and sums=1 | Verified |
| 3.5, 299–305 | `eq:method-weight-change` (3.20), unnumbered stop rule | Delta=L1(q_next−q); stop after Rq consecutive Delta≤delta_q or Tmax; reset counter after exceedance | `methods/adaptive_sampling.py`, `methods/ensemble.py` | 0<delta_q<2, Rq positive integer, Tmax≥Rq | Delta∈[0,2]; consecutive counter resets; limit and convergence logged distinctly; return last H_t | Verified; weight stability is not proof of model/parameter convergence |
| 3.6, 307–314 | `sec:method-evaluation` | K-fold test accuracy mean/std + runtime; main comparator fixed Chi-square features + bootstrap MNB + majority vote; common splits/G/count representation | `experiments/baselines.py`, `metrics.py`, `runner.py`, `reporting.py` | K,repeats,seeds,G, feature-count/smoothing grids | Same fold records across methods; fit-only scoring; search-only tuning; test-only final metrics | Framework specified; Chapter 4 final datasets/settings absent; baseline exact scoring details need declared policy |

## Diagram and supporting-source audit

The included flowchart's editable XML contains the chain `preprocess → split →
initial → subsets → mnb-groups → subspace-choice → pso → update → decision → eval`.
Its no branch loops back to subspace sampling. A test arrow bypasses model building.
The diagram shows search data entering subspace choice but does not separately
draw its PSO-fitness dependency; the prose explicitly requires that dependency.
Its diamond mentions convergence but omits the iteration-limit branch described
in §3.5. These diagram simplifications must not remove either dependency in code.
Only this one Chapter 3 figure is included by the current main source. Other PDFs
(`chapter3-training-split`, `chapter3-rase-three-level`, `chapter3-overview`, flow
options) and `_v1`/`_v2` draw.io files are historical supporting artifacts.

Chapter 2 distinguishes JMH ranking from the proposed conversion to q; RaSE's
selection/update ideas from this fixed-m MNB/search-accuracy design; random
generation from bootstrap model fitting; and the older binary-NB PSO acceptance
threshold mechanism from the present one-PSO-per-subspace rule. It lists MNB,
Complement NB, logistic regression and SVM among relevant classifiers. Chapter 1
asks about accuracy and full model-building cost in text/DNA high dimensions; it
does not establish superior performance. Bibliography keys used directly in Ch3:
`McCallumNigam1998`, `LiuWong2013`, `Vinje2015`, `Tang2016`, `Huang2023`,
`Ho2024PSO`, `TianFeng2021`, `TianFeng2023`, `MadasuElango2020`.

## Implementation status and review discipline

Planned concrete APIs (not implemented by this source audit):

| Module | Proposed callable contract |
|---|---|
| `methods/mnb.py` | `class_statistics`, `estimate_parameters`, `SubspaceMNB.predict` |
| `methods/feature_scoring.py` | `jmh_scores`, `jmh_scores_from_statistics` |
| `methods/adaptive_sampling.py` | `initial_probabilities`, `update_probabilities`, `weight_change` |
| `methods/random_subspace.py` | `sample_subspace`, `select_subspaces` |
| `methods/random_generation.py` | `generate_parameters`, `parameters_to_position`, `position_to_parameters` |
| `methods/pso.py` | `PSOConfig`, `optimize_mnb` returning model plus stop/boundary/evaluation diagnostics |
| `methods/ensemble.py` | `AFSConfig`, `AFSRGMNB.fit(X_fit,y_fit,X_search,y_search)`, `predict(X)`; fit has no test argument |

The mapping was established before implementing the proposed algorithm. Update
this paragraph and the per-component status after the implementation and tests
are reviewed. Feature extraction already had 11 passing baseline tests before the
refactor (Python 3.12.7); that alone is not a claim of method correctness or real
dataset validation. Method correctness, data-validation evidence and experiment
results must remain separate. `specification_status.md` records the reviewed
source fingerprints; `open_questions.md` distinguishes missing scientific choices
from explicit engineering conventions.
