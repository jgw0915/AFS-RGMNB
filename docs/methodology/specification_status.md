# Reviewed thesis specification

Inspection date: **2026-11-16** (Asia/Taipei). The thesis was readable locally;
no permission exception or external upload was needed. The sibling project was
clean on branch `main` at inspection and remained read-only.

## Exact source location

```text
C:\Users\kiwi\NCKU_Master\Thesis\Random Multinomial Model on High Dimension Data\Master-Thesis
```

Implementation-relative path: `../Master-Thesis`.
Thesis Git revision: `669245c9be4073e682ae76d6b0bdf80cd724d89b`.
Entry point: `main.tex` (lines 116–118 include Chapters 1, 2 and 3).

| Role | Source path relative to thesis root | Inspection result |
|---|---|---|
| Main assembly and macros | `main.tex` | Report class; ordinary amsmath equations; no algorithm environment or custom methodology macros; active chapter sequence resolved |
| Motivation | `Chapter/First Chapter.tex` | High-dimensional count-based text/DNA classification; accuracy versus full model-building/search cost |
| Literature | `Chapter/Second Chapter.tex` | MNB and other classifiers; random generation, PSO, JMH, RaSE, PRS and Chi-square/Bagging distinctions |
| Authoritative method | `Chapter/Third Chapter.tex` | Entire source read in order; six sections; 20 numbered equations |
| Active notation table | `Chapter/chapter3-notation-table.tex` | Included by §3.1; M classes, K CV folds, G members/groups, fixed m |
| Active diagram | `figures/chapter3-method.pdf` | Included as `fig:chapter3-main-flow`; rendered with Poppler and visually inspected against editable source |
| Diagram source | `figures/chapter3-method.drawio` | Current vertical flow, one outer loop and held-out test bypass; not the older `_v1`/`_v2` files |
| Bibliography | `style/chapter2_references.bib` | Actual citation records inspected; no change to bibliography or citations |
| Reference instructions | `AGENTS.md`, `notes/research-context.md` | Dated research history read; older decisions do not override current main source |
| Supporting note, not active body | `notes/chapter3-notation.tex` | Contains all-zero JMH fallback but also superseded variable-m and best-ensemble rules; not silently adopted |
| Supporting baseline note | `notes/baseline-screening-current.md` | Distinguishes presence contingency scoring from count-input MNB and from complete reproduction |

There is **no Chapter 4 source or inclusion** in the current LaTeX assembly.
The project mentions Chapter 4/5 in the intended thesis organization, which does
not constitute experimental settings or completed results. Older build/output
PDFs, Word exports, alternative diagrams and meeting notes are not substituted
for the active LaTeX method. In particular, notes dated October 21 describe an
older best-iteration and variable-size design, while the current source uses
fixed m and the final completed ensemble.

## Specification and module status

The full equation-level mapping is in
[`chapter3_implementation_mapping.md`](chapter3_implementation_mapping.md).
Scientific omissions and explicit engineering choices are in
[`open_questions.md`](open_questions.md).

| Active section / stable labels | Affected Python components | Status at this audit |
|---|---|---|
| 3.2, `sec:method-extraction`, `sec:method-data-split` | `data/{readers,preprocessing,taxonomy,splitting}.py`, `features/{dna_kmer,document,vectorization}.py` | Modular count readers/extractors and split preflight verified; three real-source exports and report complete; see dated validation |
| 3.3.1, `eq:method-md-estimates` through `eq:method-sampling-weights` | `methods/feature_scoring.py`, `methods/adaptive_sampling.py` | JMH and initial probabilities implemented with numerical tests; zero-score inputs fail explicitly (Q01) |
| 3.3.2–3.3.4, `eq:method-sequential-sampling` through `eq:method-subspace-choice` | `methods/random_subspace.py`, `methods/mnb.py` | Sequential sampler, subspace MNB and first-best search-accuracy candidate selection implemented and tested |
| 3.4.1, `eq:method-mnb-constraints`, `eq:method-mnb-initialization` | `methods/random_generation.py` | Independent normalized uniforms mapped; implementation planned |
| 3.4.2, `eq:method-pso-particle-position` through `eq:method-vote` | `methods/pso.py`, `methods/ensemble.py` | Exact representation, fitness, scalar-rho update and fixed-G hard vote mapped; implementation planned |
| 3.5, `eq:method-inclusion` through `eq:method-weight-change` | `methods/adaptive_sampling.py`, `methods/ensemble.py` | Inclusion/threshold/normalization and L1 change implemented; consecutive stopping/full loop remain planned |
| 3.6, `sec:method-evaluation` | `experiments/{baselines,metrics,runner,reporting}.py` | Infrastructure planned; final experiment settings absent |

Planned module paths are interface targets and do not claim files or completed
scientific tests exist. This audit does not claim full-method reproduction,
improved classification results, or finalized Chapter 4 experiments.

## Known differences and unresolved details

1. Current extractor learns vocabulary only on D_fit; the active body does not
   state the fitting scope explicitly. This stricter policy prevents search/test
   vocabulary leakage and is recorded as an implementation convention.
2. Legacy DNA normalization (U→T; break windows on ambiguous bases; no
   reverse-complement merge) and document tokenizer choices are not all specified
   in the current body. Dataset configurations must expose them.
3. The active initial-probability equation is undefined when every JMH score is
   zero. The standalone note supplies a fallback but is not an active equation.
4. Position addition and the open-(0,1) pre-normalization bound are supported by
   dated notes; body prose is less explicit. No generic PSO clipping substitution
   is justified.
5. No final split ratio, numeric search settings, baseline tuning grid, similarity
   grouping protocol or statistical comparison design is specified. The latest user scope is multiclass
   classification with natural class distributions; no imbalance-specific learning
   or intervention is authorized. Additional metrics do not change search fitness.
6. The flowchart does not draw the search-to-PSO connection or limit-exit branch
   separately; body text requires both. The source contains a harmless reference
   typo: initial PSO positions refer to the constraint equation (3.10), while the
   actual generation rule is (3.11). Implementation should cite both.

## Non-destructive revision check

```powershell
python tools/check_thesis_spec.py
# If the sibling project is moved:
python tools/check_thesis_spec.py --thesis-root 'D:\research\Master-Thesis'
```

[`thesis_source_fingerprints.json`](thesis_source_fingerprints.json) stores SHA-256
digests, paths and affected-module hints, **not thesis content**. The command
only reads files and Git metadata; it never edits either repository, copies the
thesis, recompiles it or changes the reviewed baseline. Exit 0 means all reviewed
file bytes match; exit 1 means a source is changed/missing; exit 2 means an input
or access error. A different Git revision is reported separately because an
unrelated thesis commit need not change the reviewed methodology.

When drift occurs, inspect changes in the active inclusion graph first, then
review affected equations, diagrams, modules and mathematical tests. Record the
scientific decision and update this mapping/fingerprint manifest manually only
after review. Never automatically change thesis text to agree with code.

Validation in this session: the checker returned `sources_match: true` for all
12 recorded files and an empty thesis Git status. Synthetic unchanged, changed,
missing-file and path-traversal cases also passed. The rendered diagram preview
is under ignored `inspection_methodology/`; no reference PDF or preview is staged.

Implementation continuation: 41 local tests pass (32 data/preprocessing tests and
9 numerical Chapter 3 sampling tests). No PSO, ensemble or final accuracy experiment
is complete. Source fingerprints remain unchanged; the thesis remains read-only.

Pre-PSO continuation (2026-11-17): duplicate-aware group-disjoint evaluation,
fit-only mapping, subspace MNB and GxL selection are complete;68tests pass.
The group-aware split is a documented evaluation extension requested by the user;
no thesis text/equation changed. Two clean-source real20News runs match under
pinned dependencies. Formal CV accuracy comparisons, random initialization and
PSO remain unimplemented. New-commit Python 3.10 CI is reported separately and
must not be inferred from the prior7a93149 workflow.
