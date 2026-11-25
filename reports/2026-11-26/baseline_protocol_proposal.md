# Primary comparator for advisor review

Chapter 3 §3.5 fixes a common count representation, split and G for the two methods.
The comparator selects one fixed chi-square feature set and fits G MNBs on bootstrap
training samples. Each member contributes one class vote. The full-vocabulary MNB
and fitted-subspace ensemble in this investigation are separate diagnostic controls.

Madasu and Elango report a class-prevalence-weighted chi-square score in §3.3.2,
binary term-presence inputs to their MNB in §3.4.4, and bagging with replacement in
§3.5.1. Their application was sentiment classification. Our occurrence-count MNB
is a deliberate adaptation. The sibling October 14 baseline note recommends screening
and estimation on Dfit and choosing the feature count and smoothing on Dsearch.
[Author manuscript](https://arxiv.org/pdf/1911.00288) · [Published article](https://doi.org/10.1007/s11042-019-08409-z)

### Proposed operational choices

| Decision | Proposed rule | Reason and consequence |
|---|---|---|
| Multiclass score | For each term and class c compute an uncorrected Pearson chi-square on its one-vs-rest 2×2 document-presence table. Sum the class scores with weights n_c/n. | Implements the prevalence-weighted formulation. All labels remain multiclass in training and evaluation. |
| Contingency counts | A = present and class c, B = present outside c, C = absent in c, D = absent outside c. Compute n(AD−BC)² / [(A+B)(C+D)(A+C)(B+D)] using float64. | Repetition within one document contributes once to screening. The subsequent MNB uses every occurrence. |
| Zero margins | Assign zero contribution to a constant feature. Require all target classes in Dfit before ranking. | Constant features cannot discriminate. This avoids arbitrary continuity corrections. |
| Cutoff and ties | Prefer K_features=m for the primary per-member budget. Rank descending and break ties by ascending original source feature ID. If a grid is approved, predeclare its bounds, smoothing grid and Dsearch rule. For equal accuracy prefer the smaller budget, then first declared smoothing. | Every proposed member sees m coordinates. Its feature union can exceed m; also report union size. |
| Bootstrap | Select the fixed columns on the full Dfit once. Independently sample n_fit row indices with replacement for each of the G members. Keep multiplicities and the original counts. | Gives ordinary bootstrap variation on a common feature dictionary. No class balancing or draws from Dsearch. |
| Missing class in a bootstrap | Proposed policy: use its empirical zero prior (exclude that class from this member’s possible predictions); map member output back to global class order. A one-class bag always predicts that class. Count every affected bag. | Retains every draw, avoids prior smoothing, and gives every member one vote. Redrawing until coverage would condition the bootstrap law. |
| Voting and timing | Use the common class order on vote ties. Reuse split indices and fit vocabulary. Time screening, bootstrap fitting and ensemble construction, then time prediction separately. Report shared preprocessing/split time once and any tuning cost separately. | Compare equal G and per-member feature budget on common partitions and hardware. Also report unique features, candidate fits and fitness calls. |

**Approval needed before benchmarking:** score and zero-margin conventions, per-member
versus union-budget matching, tie order and candidate grids, treatment of absent
classes, tuning allowance and timing boundaries. Record agreed settings before opening
further outer folds. `experiments/baselines.py` currently provides sparse fit-only
document frequencies and reproducible ordinary bootstrap indices. It does not choose
any of these unresolved policies or report a comparator accuracy.
