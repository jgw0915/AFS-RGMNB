# Reference recovery, candidate review and data-layout checkpoint

Continued 2026-11-17 on `refactor/full-thesis-framework` from clean tracked HEAD
`9b5cce39d9be781dc037821993a68c3647d918bc`. Initial Git status showed the user's
new `reference/` as untracked. The previous 107 folder had been moved there by
the user, not by this inspection. No existing user material was reset or removed.

## Protection and reference locations

- Added whole-directory `/reference/` ignore protection before further work;
  no reference file is tracked. Never execute its historical binaries/installers.
- Sources are `reference/100-陳泓宇/`, `reference/104-林修弘/`,
  `reference/107-林幸辰/`, and the Week 16 dataset-inspection PDF.
- The four-page PDF SHA-256 is
  `75379599e48ccac7ec05dce7b1ad74f4b6e91b5a411d95ef2cd0dc21c9d0768e`.
  All pages were extracted/rendered and visually inspected. Preview/text outputs
  stay under ignored `outputs/reference_inspection/pdf/`.
- Sibling `../Master-Thesis` remains read-only. All 12 reviewed source
  fingerprints match revision `669245c9`; its working tree is clean.
- Original senior programs, datasets and references were read, not changed,
  executed or uploaded. Only the small public official UCI source was downloaded.

## What was recovered

| Source | Nucleotide inputs found | Current qualification |
|---|---|---|
| Chen, 100 folder | Bacteria 3,672 / historical subset 2,035; Fungi 7,730 / historical subset 4,954 | Four unique FASTA contents across 10 copies; archived six-rank headers; singleton and later <10-row filtering already occurred upstream |
| Lin, 104 folder | `預處理為氨基酸/Project1/16SRNA.txt` (6,386) and `Fungi.txt` (7,730) | Genuine nucleotide strings before translation; Fungi uses RNA U; variable-depth Bacteria lineages and unresolved labels |
| Tsai, moved 107 folder | Previously audited preserved-count inputs | Current 20News source fingerprint still matches; 18,828 x 83,635 declared, 20 classes, 1,505,793 nonzeros |

Read [Chen's inventory](../../docs/references/senior_100_sequence_inventory.md)
and [Lin's inventory](../../docs/references/senior_104_sequence_inventory.md)
for exact paths, source-code/thesis evidence, full-source 8-mer inventories,
historical curation, label/duplicate diagnostics and limitations. The recovered
sources revise the earlier *107-only* absence of raw sequences; they do not
establish current taxonomy or a 107 row/feature bridge. Similar totals are leads,
not proof. Duplicate program-directory copies must not be concatenated.

The current seed-42 SGKF protocol failed all eight protocol-only checks of the
Chen historical subsets (Family/Genus x exact-sequence/accession-union groups).
These were marker-based allocation checks, not DNA matrices/model experiments;
they show a heuristic allocation failure, not universal mathematical impossibility.
The full unfiltered-class targets still have many genuinely undersized classes.
No extra records/classes were removed or balanced during these checks.

## Dataset candidates and the small public source audit

[The candidate review](../../docs/experiments/dataset_candidate_review.md)
lists all 30 PDF table entries with the PDF's reported counts, primary-source
verification, release differences and current decisions. It distinguishes binary
tasks, TF-IDF from integer counts, tokenized corpora from original prose and the
PDF's possible 7-mer dictionary from observed thesis 8-mer features.

Shortlist: separately acquired raw 20News/BBC and later AG News for documents;
UCI Splice for a small functional DNA task; human_ensembl_regulatory for a later
larger three-class sequence task. Only UCI Splice was downloaded/audited in this
pass; the other external candidates have source-page checks, not local matrices.

The official UCI ZIP is in ignored
`data/raw/uci_splice/retrieved-2026-11-17/`, with download URL/checksum manifest.
Its 8-mer arrays and audit are in ignored
`data/extracted/uci_splice/dna8_audit-2026-11-17/`. Retained original labels:
EI=767, IE=768, N=1,655; 3,190 rows, 47,888 full-source observed features,
168,027 nonzeros, positive int64 counts, no zero rows. All five exact-vector-group
fit/search/test split audits pass with Dfit vocabulary sizes 40,523-40,940;
all classes retained and no group/sample overlap. No model accuracy was computed.

There are repeated IDs/sequences and one cross-label vector group. Exact-vector
group protection does not establish source-ID or biological independence.
Formal source-region/ID grouping and the functional target must be reviewed.
This is a promising candidate, not automatic approval of final thesis settings.
Only [derived aggregate JSON](../../docs/experiments/uci_splice_inspection.json)
is intended for Git; IDs, sequence strings and matrices remain local.

The local evidence command was:

```powershell
& '.\outputs\ci_deps312\Scripts\python.exe' outputs/reference_inspection/uci_splice/audit.py
```

That private inspection script downloaded the official archive once, then used
its cached bytes, read `splice.data` in memory, strictly checked three CSV fields
and 60-base records, called existing `sparse_dna_matrix(..., 8, None)`, and reused
`group_aware_splits(..., n_splits=5, search_fraction=.2, seed=42)` plus
`prepare_feature_partitions`. Its hash is in the derived JSON. A reporting-only
attribute typo was corrected before final measurements; it changed no input.
No generic UCI reader/CLI or formal classifier experiment is claimed.

## Storage and CLI changes

Created `data/raw/`, `data/extracted/` and [data/readme.md](../../data/readme.md).
Only the README and two `.gitkeep` markers are whitelisted in Git; arbitrary
filenames/nested directories remain ignored. The senior references stay where
the user placed them. Existing large exports remain under `outputs/`; they were
not duplicated or silently relocated. The UCI download/audit is the only newly
populated dataset under `data/` in this session.

Current README commands use `--archive '.\reference\107-林幸辰'`; historical
reports keep their execution-time paths. Readers discover internal paths, so
no source format or mathematical code needed rewriting after the move.

`utils.io.ensure_output_outside_references` now rejects destinations inside the
whole reference tree, raw-data tree, thesis or supplied archive, including
resolved parent segments. All three source-audit/model runners apply this before
writing. The count-validation CLI also requires ignored in-repository outputs.
This prevents an ignored reference destination being mistaken for safe output.

## Validation and continuation

Validation commands use the pinned libraries in `outputs/ci_deps312/`:

```powershell
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python tools/check_thesis_spec.py
git diff --check
git check-ignore -v reference data/raw/uci_splice/retrieved-2026-11-17/source.zip data/extracted/uci_splice/dna8_audit-2026-11-17/audit.json
git diff --cached --name-status
```

Editable installation succeeded. **All 98 tests passed in 6.101s**, preserving
the previous 93 and adding five output-path protection tests, including real CLI
rejection before writing. `git diff --check` passed. Read-only source verification
matched all 12 thesis files and the moved 20News fingerprint. Ignore probes
confirmed that reference/raw/extracted research contents stay local while only
the three intended data-layout files can be tracked. Independent report review
found no material scientific overclaim; the exact-vector grouping limitation is
explicit in the README and roadmap. This is local Python 3.12.7 verification,
not a new Python 3.10 remote-CI result.

No professor-facing report/PDF or thesis methodology was edited. Mathematical
random-MNB/PSO code is unchanged. No push or merge is part of this inspection.

Next: resolve a biological grouping/schema protocol for recovered seniors' data
or UCI Splice before formal gene evaluation; acquire/audit a selected raw document
release with Dfit-only vocabulary. Do not restart completed model-search modules
or interpret these source/conversion audits as classification performance.
