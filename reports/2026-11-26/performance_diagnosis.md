# Why fold 0 performed poorly

The strongest evidence points to the random-MNB parameter search. Fitting parameters from counts on the three selected subspaces gave **46.85%** Dsearch accuracy. Using the same subspaces with fresh random parameters and the development PSO budget gave **8.18%**. Feature restriction also cost substantial predictive information. A fitted MNB using all 65,540 fit-observed columns achieved **86.52%**. These differences describe this fold and budget. They measure separate interventions and cannot be treated as an additive decomposition of test error.

### Controlled comparison on 3,020 search documents

| Model or control | Correct | Accuracy | How it was obtained |
|---|---:|---:|---|
| Majority class | 152 | 5.03% | Most frequent class in Dfit, fixed class order for ties |
| Full-feature MNB | 2,613 | 86.52% | Dfit counts, empirical priors, α=1 |
| Three fitted subspace MNBs | 1,376 / 1,242 / 1,319 | 45.56 / 41.13 / 43.68% | Same Dfit rows, selected columns and subspace denominators |
| Fitted-subspace vote | 1,415 | 46.85% | Equal votes from the three fitted controls |
| Three best initial particles | 143 / 198 / 167 | 4.74 / 6.56 / 5.53% | Highest Dsearch accuracy among five fresh draws per subspace |
| Best-initial-particle vote | 227 | 7.52% | Equal votes from those three initial models |
| Three historical-best PSO models | 176 / 198 / 254 | 5.83 / 6.56 / 8.41% | Five updates per group |
| Final proposed ensemble | 247 | 8.18% | Equal votes, original class order |

The conventional MNB controls agreed with scikit-learn on every search prediction for the full matrix and all three selected subspaces. The historical Dtest result remains 318/3,759 (8.4597%). No new Dtest predictions were made.

### What the subspaces capture

The four rounds each evaluated 12 candidates. Their mean search accuracies rose from 37.97% to 40.97%. The union of the three winners shrank from 362 to 203 distinct columns. Mean pairwise Jaccard overlap rose from 0.034 to 0.430. The initial distribution had an effective dimension of 1,789, measured by 1/Σq². This fell to 177 after the fourth update. The top 1% of coordinates held 41.05% of initial mass and 99.23% after the first update.

In the three final subspaces, 472, 558 and 461 search documents had all-zero counts (15.63%, 18.48%, 15.26%). The median document had only two nonzero selected features in each group. Those zero rows force a prior-only decision. They explain some lost coverage, while the fitted results demonstrate that many of the remaining rows carry class information. Increasing m to 1,024 raised the fitted vote to 74.37%, yet the proposed vote reached only 6.85% with the common small PSO budget. Per-round winners, overlaps, coverage and distributions are in the [measurement tables](measurement_tables.md).

### What happened inside PSO

| Group | Initial particle accuracies (%) in draw order | Historical best (%) at initialization and updates 1–5 | Resets / 25 particle updates | Coordinate redraws |
|---:|---|---|---:|---:|
| 1 | 3.81, 3.97, 4.17, 3.94, 4.74 | 4.74, 5.30, 5.30, 5.83, 5.83, 5.83 | 8 | 56,609 |
| 2 | 3.84, 4.67, 5.26, 6.56, 3.81 | 6.56, 6.56, 6.56, 6.56, 6.56, 6.56 | 6 | 59,218 |
| 3 | 5.40, 5.53, 4.93, 5.03, 4.17 | 5.53, 6.29, 6.32, 8.41, 8.41, 8.41 | 7 | 50,291 |

Each swarm used 30 fitness evaluations and reached its update limit. Across groups, 12 of 75 particle updates improved personal bests, and 5 of 15 swarm updates improved global bests. A velocity vector needed at least one coordinate redraw in 65/75 updates (86.67%). In total 166,118/193,500 coordinates were redrawn (85.85%) and 21/75 particles reset (28%). The ensemble gained 20 correct predictions, or 0.66 percentage points, over its initial vote.

On accepted moves, mean absolute coordinate changes after normalization were 5.00×10⁻⁶, 4.96×10⁻⁶ and 4.60×10⁻⁶. Including resets raised these means to 0.00175, 0.00131 and 0.00152. The distinction matters: a large jump caused by redrawing an entire model says little about movement toward the historical best. The [velocity sweep and equal-budget random controls](hyperparameter_sensitivity.md) quantify this effect.

### Did the ensemble collapse?

All three retained members predicted all 20 labels individually. Their pairwise disagreement averaged 95.36%. They frequently disagreed in three different directions: 2,656 of 3,020 votes (87.95%) had tied winners, so the prescribed fixed-order tie rule decided those rows. The final vote predicted 18 labels, with 62.52% of predictions concentrated on classes 1, 2 and 7. Classes 17 and 19 received zero predictions. The concentration is partial, and diversity by itself was insufficient. At G=10 with the larger swarm, the seed-42 prefix control had 2,716 ties using the first three members and 1,300 using all ten. Fixed-order prefixes use one trained ensemble and preserve its member order. [Class counts, recall and confusion matrix](measurement_tables.md)

### Replay and interpretation

The source SHA-256 was `2f18b7b11c294dda1d716988f38a4c1184c0344c1eec73ad0f510d0a887b1528`. Archived fit, search and test *index* digests matched the versioned configuration. The loader recomputed the column mapping from 12,049 fit rows, checked the source feature IDs and sliced only fit/search rows. The replay matched every round’s weight hash, candidate accuracy and selected columns, every swarm history, and the saved final parameter arrays. Regenerating all 15 starting positions from the original `b111a1d` implementation with the archived seeds also matched the diagnostic hashes exactly.

These checks found no implementation or numerical discrepancy explaining the low score. The fitted controls quantify the subspace cost. The same-subspace comparison, small accepted moves and frequent resets identify a further loss during random parameter search. Dsearch labels enter both subspace selection and fitness, so even a reproducible Dsearch gain still needs independent confirmation. Historical preprocessing of the preserved matrix and possible near-duplicate documents remain additional limits.
