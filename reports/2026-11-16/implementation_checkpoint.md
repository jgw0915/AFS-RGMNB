# Implementation continuation after the professor report

The professor-facing PDF and data report were completed and visually verified
before this additional mathematical milestone. Its reference to 32 tests describes
the data/preprocessing suite at report creation. Nine subsequent numerical tests
bring the full local suite to **41 passing tests** on Python 3.12.7.

The installed-package CLI was also checked on a synthetic 30-document CSV using
the named Porter/stoplist pipeline and all-record mode: all five fold exports
preserved class counts 10:20, removed the stopword, and produced `system:2` and
`real-tim:1`. This is synthetic integration evidence, not raw-corpus reproduction.
All 30 source/test/tool files parsed under Python's 3.10 grammar. Python 3.10
runtime execution was not available locally; CI installs the package explicitly.

Implemented against the unchanged active thesis Chapter 3:

| Source equations | Code | Verified behavior |
|---|---|---|
| (3.1)-(3.3) | `methods/feature_scoring.py:jmh_scores` | Smoothed occurrence-count probabilities; empirical-prior complementary mixtures; unweighted sum of class KLs; sparse class aggregates |
| (3.4)-(3.5) | `methods/adaptive_sampling.py:initial_probabilities` | Uniform exploration floor and valid distribution; all-zero scores fail explicitly |
| (3.6) | `methods/random_subspace.py:sample_subspace` | Fixed-size sequential weighted sampling without replacement; remaining-weight renormalization; reproducible supplied RNG |
| (3.17)-(3.20) | `inclusion_proportions`, `update_probabilities`, `weight_change` | Exact inclusion proportions, strict threshold branch, positive normalized update, L1 distance |

Tests include hand-calculated binary Jeffreys and nonuniform multiclass mixture
examples, counts versus binary presence, equality at the inclusion threshold,
domain rejection, and the actual conditional probability sequence. Float64 is
used; only negative KL cancellation within 1e-12 is clipped to zero. Probability
underflow/invalid domains produce errors instead of a changed model.
Features whose estimated probabilities are exactly identical across classes
receive exactly zero divergence, avoiding spurious positive complement-rounding
scores. No tolerance discards small genuinely discriminative differences.

The complete consecutive stopping loop, subspace MNB selection, random model
generation, PSO, ensemble and benchmark runner remain unimplemented. No complete
method, performance advantage or final experiment setting is claimed. The
equation mapping and roadmap record this distinction. Matrix artifacts remain
local and ignored; no research source files or datasets have been committed.

Local milestones: `971c35d` sampling/decoding fixes; `34876c4` professor report,
matrix validation and research documentation. Later commits may add this
mathematical subset; consult Git for the exact current revision. No push was
performed during this continuation.
