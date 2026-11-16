# Subspace MNB numerical validation

Implemented `src/afs_rgmnb/methods/mnb.py` against active thesis Chapter 3
section 3.3.3 equations (3.7) and (3.8), reviewed source `669245c9` unchanged.

## APIs and mathematical behavior

- `class_statistics(X_fit,y_fit,class_order=...)` returns sparse float64 class
  occurrence totals, empirical sample counts, fixed classes and input dimension.
- `estimate_parameters(stats, feature_indices, alpha=...)` adds positive alpha
  and normalizes **within the selected m columns**, not the full input universe.
- `SubspaceMNB(alpha=...,class_order=...).fit(...)` and `.from_statistics(...)`
  produce the same model. The candidate selector reuses one fit aggregation.
- `MNBParameters`, `class_log_scores`, `.from_parameters(...)` and
  `.predict_log_scores(...)` expose reusable validated scoring for later work.
  Supplying explicit probabilities does not implement random initialization.
- `.predict(X_full)` returns the first maximum in sorted class order by default;
  explicit class order is preserved. Selected columns stay in draw order.

Prediction is `log(pi_y)+sum_(j in S) x_j log(theta_yj)`. These are log scores,
not normalized posterior probabilities. All-zero selected rows predict the
largest empirical prior, with deterministic first-class ties. A class with no
counts in its selected features receives a uniform conditional row after smoothing.
Alpha, labels, class order, counts, selected indices and explicit probability
parameters are validated; fractional/negative/nonfinite counts, missing or mixed
type labels, duplicate/out-of-range columns and invalid alpha are rejected.

## Hand calculation

For fit counts `[[2,0,100],[0,2,0],[1,1,3]]`, labels `[a,b,a]`, selected
columns `[0,1]`, alpha=1:

- Class counts are `[2,1]`, so priors are `[2/3,1/3]`.
- Selected occurrence totals are `[[3,1],[0,2]]`.
- Smoothed conditional rows are `[[4/6,2/6],[1/4,3/4]]`, each summing to one.
- The unselected third column contributes neither to the denominator nor scores.

Eleven targeted tests pass: hand-computed smoothing, empirical priors, occurrence
versus presence counts, sparse/dense equivalence, reuse of statistics, zero rows,
zero class-feature counts, reversed feature order, ties and invalid domains.

Comparison against [sklearn MultinomialNB](https://scikit-learn.org/1.4/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
is tested only after selecting exactly the same columns, with identical scalar
positive alpha, empirical unsmoothed priors and the same sorted class order.
The comparison directly checks conditional probabilities, empirical priors and
predictions. The score formula is separately checked against hand calculations.
This does not assert equivalence for different smoothing, priors, event models
or feature sets, and is not a reproduction of a separate thesis comparator.

## Candidate and integrated tests

Four selector tests verify exactly G groups and L candidates, fixed unique m,
first-candidate ties, supplied-seed reproducibility, one JMH call, and search
labels changing evaluation without changing fitted parameters. A hand-calculated
case gives candidate accuracies `[.5,1,1]`, selecting index1 (the first maximum).
Three integrated tests cover group split -> fit-only mapping -> JMH -> q -> models ->
search selection, heldout-only columns, API boundaries and saved private artifacts.

The full suite is **68 passing tests**, including all prior 41 tests. It passes
on Python 3.12 with both current local libraries and the pinned CI dependencies.
Python 3.10 GitHub Actions for the new commit is recorded separately in
`implementation_status.md`; no prior workflow result is treated as verification
of new code. No PSO, random MNB generation or ensemble is implemented here.
