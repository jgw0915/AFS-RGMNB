# Duplicate-aware 20Newsgroup split audit

Executed 2026-11-17 Asia/Taipei; retained in the requested November 16 report directory.
Source code: `fcb68df8979f2fb5eece6cbc7130c84524dc990a`; clean worktree at execution.
Source SHA-256: `2f18b7b11c294dda1d716988f38a4c1184c0344c1eec73ad0f510d0a887b1528`.

## Source and identity definition

Preserved ASCII counts: `107-林幸辰/Program/00_前置處理/text/20Newsgroup/20newsOutput.txt`.
All **18,828 rows, 20 original classes, 1,505,793 nonzeros** retained. No labels
changed, no rows removed or balanced. Source feature IDs are one-based within
83,635 declared columns; 83,415 appear in the full source inventory.

Canonicalize sparse storage (validate integer nonnegative counts, sum duplicate
entries safely, eliminate explicit zeros, sort indices), then hash the full count
vector using a versioned prefix, dimension, fixed-endian int64 indices and values.
Labels are excluded. Equal vectors with different labels therefore share a group.
Storage format/dtype/order do not create artificial differences. Group identity
is stable under row reordering; SHA-256 collision resistance is assumed.

Verified **18,777 groups**, including **38 repeated groups / 89 participating rows /
51 excess rows** and **5 cross-label groups / 10 participating rows**. All conflicts
remain. The original class-distribution table reports class 19 as 755; this source
has 775, which is retained and documented rather than corrected to match the paper.

## Protocol

Outer **5-fold StratifiedGroupKFold**, shuffle=True, seed=42. Each outer training
portion is split using nested SGKF, seed=42+outer_fold+1. Requested search fraction
f=.2 gives k=round(1/min(f,1-f))=5 (round-half-up, minimum two). For f<=.5 the nested
heldout side is search; above .5 the roles reverse. Among feasible inner folds,
choose the first minimizing:

`abs(n_search/n_outer_train - f) + mean(abs(p_fit-p_outer_train)) + mean(abs(p_search-p_outer_train))`.

Only outer-training label proportions and target size enter this allocation
objective. No model score, test accuracy or test-based hyperparameter choice is
used. Exact proportions are not promised: [SGKF attempts stratification subject
to non-overlapping groups](https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
Every required class must occur in fit/search/test; infeasible cases raise an
error instead of removing classes. Every row appears in outer test exactly once.

| Outer fold (zero-based) | Fit rows | Search rows | Test rows | Maximum pairwise sample overlap | Maximum pairwise group overlap | All classes present |
|---|---:|---:|---:|---:|---:|---|
| 0 | 12049 | 3020 | 3759 | 0 | 0 | True |
| 1 | 12047 | 3008 | 3773 | 0 | 0 | True |
| 2 | 12046 | 3013 | 3769 | 0 | 0 | True |
| 3 | 12049 | 3011 | 3768 | 0 | 0 | True |
| 4 | 12054 | 3015 | 3759 | 0 | 0 | True |

## Modeled development fold: outer fold 0

Inner seed 43; selected inner fold 4. Requested
search share .2; realized **0.200411440706** of outer train.
Deviations below are absolute percentage-point differences from full-source class
proportions, not relative percent errors.

| Role | Samples | Classes | Groups | Maximum class-proportion deviation (percentage points) | Cross-label groups |
|---|---:|---:|---:|---:|---:|
| fit | 12,049 | 20 | 12,018 | 0.221502 | 4 |
| search | 3,020 | 20 | 3,005 | 0.519397 | 1 |
| test | 3,759 | 20 | 3,754 | 0.732866 | 0 |

| Original class code | Full count | Fit | Search | Test |
|---|---:|---:|---:|---:|
| 1 | 999 | 666 | 152 | 181 |
| 2 | 997 | 658 | 149 | 190 |
| 3 | 994 | 622 | 146 | 226 |
| 4 | 994 | 626 | 167 | 201 |
| 5 | 991 | 635 | 157 | 199 |
| 6 | 990 | 617 | 152 | 221 |
| 7 | 990 | 623 | 153 | 214 |
| 8 | 987 | 621 | 174 | 192 |
| 9 | 985 | 631 | 161 | 193 |
| 10 | 982 | 639 | 166 | 177 |
| 11 | 981 | 641 | 147 | 193 |
| 12 | 980 | 614 | 160 | 206 |
| 13 | 973 | 623 | 167 | 183 |
| 14 | 972 | 636 | 161 | 175 |
| 15 | 961 | 622 | 151 | 188 |
| 16 | 940 | 585 | 160 | 195 |
| 17 | 910 | 571 | 142 | 197 |
| 18 | 799 | 505 | 125 | 169 |
| 19 | 775 | 517 | 120 | 138 |
| 20 | 628 | 397 | 110 | 121 |

Fit/search/test sample and group overlaps are all zero. Columns are learned from
**fit only: 65,540 observed columns**; the same mapping is
applied to search/test, preserving original source IDs. Local zero-based model
columns and original one-based source IDs are stored separately.

## Reproducibility and remaining leakage limitations

Canonical environment: Python 3.12.7, NumPy 1.26.4, SciPy
1.11.4, sklearn 1.4.2. Two fresh runs reproduce every
outer/nested split array, audit, feature map, JMH score, sampling weight, candidate
and winner exactly. A preliminary sklearn 1.9.0 run produced different splits;
that run is not substituted into these measurements. Pin library versions:
identical seeds do not guarantee identical splitting across sklearn releases.

Private artifacts: `outputs/2026-11-17/pre_pso_pinned/` and `_repeat/`, including
`split_indices.npz`, `selected_fold_records.tsv`, `fit_source_feature_ids.npy`
and `run.json` (all five fold audits, per-class deviations, seeds and digests).
The TSV retains source archive row IDs and original class codes. No raw IDs or
matrices are committed with this report.

Grouping uses exact **count-vector** identity and does not establish raw document
identity. Near duplicates, shared threads/authors and semantic similarities are
not detected. Historical tokenization, header/stopword processing, stemming and
the precomputed dictionary cannot be reconstructed. Fit-only masks cannot undo
upstream vocabulary/feature-selection leakage. These remain development data
with explicit limitations, not a clean final raw-corpus benchmark. Test values
participate only in identity protection, stratification and the fixed transform;
no test labels/features enter JMH, parameter estimation or candidate selection.

Groups are defined before vocabulary reduction, in the fingerprinted full source
feature space. Later vocabulary masks or model subspaces can map distinct original
vectors to identical reduced representations; those projection collisions do not
establish duplicated source observations and are not relabeled as identity groups.
