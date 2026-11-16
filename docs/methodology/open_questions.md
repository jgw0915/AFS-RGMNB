# Scientific questions and declared implementation choices

Reviewed 2026-11-16. Authority is the active `../Master-Thesis/main.tex` inclusion
graph at revision `669245c9be4073e682ae76d6b0bdf80cd724d89b`, especially
`Chapter/Third Chapter.tex`. The LaTeX project is read-only. This document records
questions; it does not amend the thesis.

## Scientific decisions that remain open

| ID | Question / source evidence | Consequence and present policy |
|---|---|---|
| Q01 | §3.3.1 equation `eq:method-initial-sampling` divides by sum(s); active body gives no all-zero case. `notes/chapter3-notation.tex` and research-context 2026-09-29 explicitly mention uniform fallback, but are standalone/historical sources. | A strict active-equation implementation must reject zero total score; any optional uniform fallback must be explicit and attributed to the supporting note. Never quietly return NaN or use an invented entropy score. Equal positive scores already yield uniform mathematically. |
| Q02 | §3.2.1 does not fix text tokenization, encoding, language-specific segmentation, stopword list, stemming or malformed-text policy. | Dataset configurations must declare these choices and distinguish original senior preprocessing from new raw-count extraction. Document provenance is required before claiming raw-corpus reproduction. |
| Q03 | §3.2.1 specifies overlapping DNA 8-mer counts but not U→T, ambiguity/invalid-symbol treatment, reverse-complement merging, duplicate/similarity threshold, or taxonomic schema. | Preserve legacy choices as explicit conventions (U→T, reset windows at ambiguity, no reverse-complement merging); validate real source headers. Exact duplicates and near-duplicates require separate reporting. Do not treat count validity as independence. |
| Q04 | §3.2.2 gives stratified K-fold plus stratified fit/search without final K, ratio, class-size eligibility, or repeat seeds; no active Chapter 4 exists. | Parameters must be configurable. Smoke-test values are not final thesis settings. Do not silently discard rare classes or claim a balanced subset represents the original multiclass population. |
| Q05 | Active extraction text says establish a common feature list but does not locate vocabulary fitting within the split. Older notes mention both fit and outer train. | Use D_fit-only vocabulary as an explicit conservative implementation restriction; record observed dimensions per fold. Test and search do not create columns. Changes require re-review. |
| Q06 | Alpha, epsilon, m, G, L, C0, Np, c1, c2, vmax, Imax, delta_q, Rq and Tmax have domains but no final values. Only k=8 and w=.9 are fixed in the current body. | Require recorded configurations; examples demonstrate mechanics only. Any performance-based configuration choice must use search or nested training validation, never outer test. |
| Q07 | §3.4.2 gives velocity equation and prose position update, but no explicit z_next=z+v equation or numerical definition of valid pre-normalized range. Supporting research-context 2026-10-12 and meeting note specify coordinates outside (0,1) reset the whole particle. | Implement addition followed by strict (0,1) validation and whole-particle redraw, explicitly documenting this inference. Keep historical bests. If thesis revises this to a different map, review PSO and bounds tests. Do not replace with clipping, reflection, softmax or generic projection. |
| Q08 | Ch3's main baseline is Chi-square + Bagging–MNB, but its body has no multiclass aggregation formula, zero-margin rule or bootstrap missing-class policy. Ch2 describes presence/absence contingency; `notes/baseline-screening-current.md` says class-prior-weighted scoring, G bootstrap draws of size n_fit, counts for MNB. | Standard single MNB/CNB/LR/SVM interfaces can be implemented. The adapted comparator must disclose its exact Chi-square and bootstrap policies; it cannot be claimed as a complete reproduction of Madasu–Elango. A generic raw-count sklearn chi2 call is not automatically the described contingency score. |
| Q09 | No Chapter 4 source, final dataset tasks, grouping threshold, tuning grids, repeated-CV counts or inferential comparison plan is present. | Build configurable experiment/report infrastructure and descriptive extraction reports; do not manufacture final experimental settings or claims. Report class distributions, macro-F1/balanced accuracy/per-class diagnostics as requested by the current user, retaining accuracy as the method's fitness and primary thesis metric. |
| Q10 | Standalone notes retain older variable-size subspaces, best-iteration selection, old M/K meanings, D_select and per-coordinate rho notation. | These are not active algorithms. Current body fixes m, G members, no D_select, final H_T, M classes, K folds, scalar rho per particle. Do not import stale note rules except separately identified clarifications. |

## Routine engineering conventions (not missing scientific equations)

- Python uses zero-based indices, with persisted feature and class ordering.
- Particle serialization can store all M priors followed by row-major M×m
  conditional probabilities; its inverse must be tested. The thesis specifies a
  fixed order but not a byte layout.
- For candidate and global-best argmax ties, choose the first index in a fixed
  deterministic ordering. Personal-best ties explicitly retain the earlier
  position. Prediction/ensemble ties use fixed class order (ensemble ties are
  explicit in the thesis). Log these conventions for reproducibility.
- Compute natural logarithms and use float64 for probabilities and scores.
  Validate nonfinite/negative inputs and preserve sparse sample matrices. Tiny
  negative KL values from floating-point cancellation may be clipped at zero
  only within a documented numerical tolerance; material negative values error.
- Open-uniform initialization must exclude machine-representable zero; sampling
  zero can be redrawn. Normalized uniforms are not a Dirichlet-uniform simplex
  sample. No empirical prior is substituted into random candidates.
- Check exact position equality for PSO convergence, never equality of fitness
  or an invented tolerance. Record whether initialization or an update triggered
  the test, particularly for a one-particle configuration; use multi-particle
  settings for meaningful swarm experiments.
- Inspect the original thesis figure and source, but do not execute historical
  binaries, modify reference sources, or commit the extracted archive.

## Changes requiring scientific review

Changing the JMH definition, mixing old weights into the RaSE update, optimizing
macro-F1 in place of accuracy, using random subspace sizes, replacing normalized
uniforms with a Dirichlet law, allowing PSO to optimize q, changing boundary
repair, retaining a best-scoring earlier H_t, or tuning from outer-test results
changes the method. Such changes require a separately approved methodology
revision and a new source fingerprint; they are not refactoring decisions.

## November 16 scope and implemented subset

The current user instruction prioritizes high-dimensional multiclass classification.
Natural class sizes and labels are preserved; no imbalance-specific learning is
implemented. Full gene evaluation remains unresolved because some classes have
two samples and biological ranks are not confirmed. JMH/initial weights, inclusion
updates, L1 change and sequential sampling are now implemented and numerically
tested; Q01 remains an explicit error rather than an invented fallback.
