# AFS-RGMNB

A Python 3.10-compatible research framework for the NCKU master's thesis on
randomly generated MNB ensembles for high-dimensional **multiclass** classification.
The executable method follows the November 25 Chapter 3. Each fold finishes adaptive
subspace selection, runs one independent PSO search on each final subspace, constructs
one equal-vote ensemble and scores its held-out test partition.

## Run the revised workflow

After installing the package and pinned dependencies, run the fingerprinted
20News development fold from the repository root.

```powershell
python scripts/run_revised_cv.py --archive './reference/107-林幸辰' --output outputs/revised-fold0
```

Use `--config PATH` for a recorded parameter set and `--all-folds` for complete
K-fold evaluation. Each run needs its own output directory. The two-run development
validation retained all 20 classes and scored 318 of 3,759 held-out records correctly.
The primary Chi-square + Bagging-MNB comparison and formal Chapter 4 runs remain open.
See the [configuration and output guide](docs/experiments/revised_workflow.md) and
[validation report](reports/2026-11-25/revised_method_validation.md).

## Diagnose 20News search accuracy

The November 26 investigation reuses the archived fold-0 partition and scores
Dsearch. The fitted-subspace control reached 46.85% and the reference PSO vote
reached 8.18% on the same 3,020 documents. See the [diagnosis](reports/2026-11-26/performance_diagnosis.md)
and [accuracy–cost study](reports/2026-11-26/hyperparameter_sensitivity.md).

Use the pinned experiment environment and a fresh output directory for each
command. The default reference is `outputs/2026-11-25/run-a`; use
`--reference PATH` for an equivalent archived run containing `run.json`,
`fold_00/fold.npz` and `fold_00/ensemble.npz`. The driver verifies the source,
partition indices and fit-column mapping.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/run_search_diagnostics.py --archive './reference/107-林幸辰' --config configs/experiments/20news_diagnostics_v1.json --output outputs/diagnosis/ofat
python scripts/run_search_diagnostics.py --archive './reference/107-林幸辰' --config configs/experiments/20news_diagnostics_followup_v1.json --output outputs/diagnosis/followup
python scripts/run_search_diagnostics.py --archive './reference/107-林幸辰' --config configs/experiments/20news_diagnostics_confirmation_v1.json --output outputs/diagnosis/confirmation
```

To regenerate the aggregate Markdown and figures, run the following with NumPy
and Matplotlib installed. The measured report used Matplotlib 3.10.8.

```powershell
python scripts/summarize_search_diagnostics.py --ofat outputs/diagnosis/ofat --followup outputs/diagnosis/followup --confirmation outputs/diagnosis/confirmation --output outputs/diagnosis/report
```

## November 16 progress report

Start with [the Chinese professor summary](reports/2026-11-16/professor_summary_zh-TW.md),
[the PDF](reports/2026-11-16/professor_report_zh-TW.pdf),
[dataset status](reports/2026-11-16/dataset_status.md), and
[measured validation](reports/2026-11-16/validation_results.md).
The [email draft](reports/2026-11-16/email_draft_zh-TW.md) has not been sent.

| Preserved multiclass count source | Samples | Classes | Observed full-matrix features |
|---|---:|---:|---:|
| 20Newsgroup | 18,828 | 20 | 83,415 |
| Bacteria | 6,386 | 741 | 64,934 |
| Fungi | 7,730 | 926 | 61,563 |

These are verified **precomputed counts**, not newly extracted raw documents or
DNA. All records/classes remain. Numeric gene labels have unverified biological
rank; no Family/Genus task is established. Small classes and duplicate vectors
require reviewed evaluation rules. Optional local MDR88 has 14,680 records and
3 classes, excluding the seven-record fourth class upstream. No classifier
accuracy is claimed by the count audit.

## Install and check

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m afs_rgmnb --help
python tools/check_thesis_spec.py
```

On Linux, use `source .venv/bin/activate`. The console command `afs-rgmnb` is
installed with the package; `python -m afs_rgmnb` avoids PATH issues. Legacy
`python feature_pipeline.py ...` and historical imports remain supported.
The pinned requirements target Python 3.10 CI; reports record actual local
Python/library versions separately.

## Audit preserved senior counts

The read-only archive is now `reference/107-林幸辰/` (author 蔡幸辰). It is
ignored as an entire directory. Do not execute legacy binaries or concatenate
historical binary experiments. The audit discovers its actual internal paths:

```powershell
python scripts/validate_senior_datasets.py --archive '.\reference\107-林幸辰' --output '.\outputs\2026-11-16\verified_run' --dataset all --export-matrices
# Optional reduced MDR88 source, audited separately:
python scripts/validate_senior_datasets.py --archive '.\reference\107-林幸辰' --output '.\outputs\2026-11-16\optional_mdr88' --dataset mdr88
```

Outputs are sparse NPZ counts, NPY labels, archive identifier TSVs and JSON
measurements/fingerprints. The exported columns retain declared source feature
IDs, including zero columns. A fit-only observed-column mask is required before
experiments. The DNA source-ID-to-sequence order is unknown; do not silently map
these saved columns to the new extractor's lexical ACGT order.

## Extract from independently supplied raw inputs

```powershell
python -m afs_rgmnb extract --kind text --input 'D:\research\documents.csv' --output '.\outputs\raw_documents' --sampling-mode all --min-classes 2 --document-preprocessing senior --stopword-file 'D:\research\stopwords.txt' --porter-mode original --encoding utf-8
```

CSV requires `id,label,text`; JSONL and `documents/<class>/*.txt` are supported
with `--text-format`. `--encoding` and `--encoding-errors strict|replace` are
explicit. The senior-style option lowercases, tokenizes punctuation/space while
preserving hyphenated words, removes an explicit stoplist, then applies the named
Porter variant. `--stopwords english` is an explicitly named alternative list,
not the recovered historical list. `--remove-headers` is an optional RFC822
policy; default legacy tokenizer remains available. Counts stay integer; no
TF-IDF, binary conversion or length normalization is applied.

DNA accepts gzipped/multiline FASTA with `--dna-format silva|dada2|metadata`.
Select a verified `--rank family|genus`, `--rank-schema six|seven`, and domain.
Eight-base windows overlap with stride one; repeated occurrences count repeatedly.
Ambiguous symbols break windows, U maps to T, reverse complements remain separate.
These conventions do not imply the original archived extraction was reproduced.

`--sampling-mode all` keeps the natural distribution and reports duplicates;
`--duplicate-policy drop` explicitly removes conflicting content and collapses
same-label duplicates. Balanced mode is retained for controlled tests and warns
that proportions change. `--max-classes 0` controls only the class cap; it does not
cancel per-class sampling in balanced mode. Explicit include/exclude and size
filters are recorded. No rare class is silently removed to make splitting work.

Outer folds are stratified; each train portion is split into fit/search.
Vocabulary is learned on fit only, and search/test cannot add columns. The CLI
checks every actual split for class coverage before matrix export. Source
snapshots contain raw records, so output directories must remain ignored.
Duplicate/group independence still needs review before performance claims.

## Package and scientific traceability

- `data/`: readers, explicit preprocessing/taxonomy and senior count adapters.
- `features/`: independent DNA/document extraction and legacy fold export.
- `methods/`: verified Chapter 3 components as they are implemented.
- `experiments/`: adaptive CV runner and one-round development diagnostics.
- `utils/`, `cli.py`: I/O and reusable command interface.

See [AGENTS.md](AGENTS.md), [ROADMAP.md](ROADMAP.md),
[Chapter 3 mapping](docs/methodology/chapter3_implementation_mapping.md),
[open scientific questions](docs/methodology/open_questions.md), and
[senior method comparison](docs/references/senior_feature_extraction_comparison.md).
The fingerprint checker reads reviewed sources without editing either project.

Implemented mathematical helpers: `jmh_scores(X_fit, y_fit, alpha=...)`,
`initial_probabilities`, `inclusion_proportions`, `update_probabilities`,
`weight_change`, and `sample_subspace(weights, size, rng=...)`. Nine numerical
tests cover the exact Chapter 3 equations; 41 tests passed at that milestone.
The pre-PSO continuation below expands the suite to 68.
Initial all-zero scores raise an explicit error because the active normalization
equation is undefined. No generic entropy, PSO or ensemble substitute is used.
Older `docs/VALIDATION.md` and `docs/SENIOR_METHODS.md` retain historical context;
the dated report and current comparison supersede their archive-availability claims.

No raw datasets, senior programs, thesis source or heavy outputs belong in Git.
Review staged paths and `git check-ignore -v` before local commits. Never push,
merge, or upload private research materials without user authorization.

## Historical one-round pre-PSO diagnostic

The duplicate-aware 20News flow now runs end to end:
exact count-vector groups → outer/nested approximately stratified group splits →
Dfit-only feature map → JMH → initial q → G×L subspace MNB candidates →
first-best **Dsearch accuracy** choice per group. Dtest is not scored.

```powershell
# Activate an environment installed from requirements.txt first.
python scripts/run_pre_pso.py --archive '.\reference\107-林幸辰' --output '.\outputs\pre_pso_smoke'
```

Configuration lives in `configs/experiments/20news_pre_pso_smoke.json`; G=3,L=4,
m=128,alpha=1,epsilon=.05,seed=42 are development settings. The runner checks
the source fingerprint and refuses research outputs in tracked directories.
Original IDs/classes/source feature IDs, split arrays, per-class deviations,
candidate draws/accuracies/ties, runtime/memory and source-code hashes are saved
in the ignored output directory.

The canonical pinned-library run retains all 18,828 records / 20 classes. Outer fold 0
has fit/search/test sizes 12,049/3,020/3,759 and 65,540 fit-observed columns, with
zero sample/group overlap. All12 candidates are evaluated; the selected candidate
indices are `[0,0,0]`. Two fresh seeded runs match exactly. Library versions are
part of the protocol: the earlier sklearn 1.9 run differs from pinned sklearn 1.4.2.

New APIs: `group_aware_splits`, `prepare_feature_partitions`, `class_statistics`,
`estimate_parameters`, `SubspaceMNB`, `select_subspaces` and `run_pre_pso`.
**68 tests pass locally**, including all prior 41, under both current and pinned
dependencies on Python 3.12. The [Python 3.10 GitHub Actions run](https://github.com/jgw0915/AFS-RGMNB/actions/runs/44007208985)
also passed all68tests for milestone commit d3a44f8. The real-data smoke and remote
synthetic-test CI are separately recorded evidence.
See [implementation status](reports/2026-11-16/implementation_status.md),
[split audit](reports/2026-11-16/20news_split_audit.md),
[MNB validation](reports/2026-11-16/subspace_mnb_validation.md) and
[integration record](reports/2026-11-16/pre_pso_integration.md).

Exact-vector grouping cannot recover raw documents or undo historical upstream
preprocessing leakage. No random MNB initialization, PSO, voting, full AFS loop
or formal Chapter4 classifier comparison is included in this milestone.

## Historical one-round random MNB and PSO diagnostic

`methods/random_generation.py` implements the thesis's independent-uniform,
block-normalized random priors and conditionals, with stable particle serialization.
`methods/pso.py:optimize_mnb` searches each selected subspace using Dsearch accuracy,
fixed inertia .9, synchronous histories, coordinate velocity redraw and whole-particle
position reset. Exact position equality or the configured update limit stops it.
The explicit Q07 position-map interpretation remains documented in the methodology.

```powershell
# Use the pinned requirements environment; output remains private/ignored.
python scripts/run_pso.py --archive '.\reference\107-林幸辰' --output '.\outputs\pso_smoke'
```

`configs/experiments/20news_pso_smoke.json` provides small development settings,
not final thesis hyperparameters. The full 20News smoke run retains the verified
pre-PSO split, initializes one fresh swarm per selected group, and returns three
individual models. It evaluates no test accuracy and performs no ensemble vote.
The saved histories, boundary counters and model arrays reproduce exactly in a
second run. **93 tests pass locally** with pinned dependencies on Python 3.12;
new-code remote Python 3.10 CI has not yet run for this local milestone.

See [random initialization/PSO validation](reports/2026-11-17/random_mnb_pso_validation.md),
[November 17 checkpoint](reports/2026-11-17/implementation_status.md), and
[gene experiment prerequisites](docs/experiments/gene_experiment_requirements.md).
## Local datasets and newly supplied references

Use [data/readme.md](data/readme.md) for the storage convention: `data/raw/`
holds inputs before feature extraction, and `data/extracted/` holds preserved
or newly generated count representations. All data contents remain ignored;
only the README and directory markers are tracked. Experiment/model outputs
remain under `outputs/`. The new layout does not relocate reference originals.

The whole `reference/` directory is ignored and read-only. The newly supplied
`100-陳泓宇/` and `104-林修弘/` folders contain nucleotide sequences with archived
taxonomy headers. The earlier raw-sequence limitation referred to the 107 folder
alone. Recovery does not automatically resolve current ranks, rare classes,
duplicate/accession grouping, or equivalence to the 107 matrices. See the
[2012 sequence inventory](docs/references/senior_100_sequence_inventory.md) and
[2016 sequence inventory](docs/references/senior_104_sequence_inventory.md).

The [dataset candidate review](docs/experiments/dataset_candidate_review.md)
records all 30 table entries in the Week 16 PDF, source checks and eligibility
limits. A small official UCI Splice download is now stored locally in `data/`:
3,190 records / 3 functional classes, 47,888 observed DNA 8-mers, and all five
exact-count-vector-group split audits pass. This is a source/extraction/split audit, not
classification performance or a Family/Genus task. All private/local artifacts
and continuation instructions are in the
[reference inspection checkpoint](reports/2026-11-17/reference_data_checkpoint.md).

## 2026-11-23 advisor data-extraction demonstration

The 60-message **raw-text → AFS-RGMNB word-count** demo is implemented in `scripts/export_20news_60_demo.py`; the exact PowerShell commands and limitations are recorded in [the 11/23 run guide](docs/experiments/2026-11-23_20news_demo.md). The [local Codex handoff](docs/experiments/2026-11-23_codex_handoff.md) includes checks and professor-delivery outputs. Three *unmeasured candidate* dataset versions (Reuters R8, WebKB 4-class, BBC News) are documented in [new_dataset_candidates.csv](reports/2026-11-23/new_dataset_candidates.csv).

The script downloads public source messages when run locally; **no real 60-document CSV or observed demo dimensions are committed**. The data output must stay in ignored `outputs/`; the script and its offline synthetic tests are committed here.
