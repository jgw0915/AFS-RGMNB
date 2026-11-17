# Requirements for defensible multiclass gene experiments

Reviewed 2026-11-17 against the preserved-count audit and the current split
implementation. This is a dataset/protocol requirements record, not a completed
gene classification experiment or approval of final settings. Original numeric
labels, natural class sizes and all source rows remain unchanged.

## What the local evidence establishes

The [source inspection](../references/senior_feature_extraction_comparison.md),
[dataset status](../../reports/2026-11-16/dataset_status.md) and
[validation results](../../reports/2026-11-16/validation_results.md) establish:

| Preserved source | Count records | Numeric classes | Class size min / median / max | Classes with <5 rows (rows) | Classes with exactly 2 rows | Full-matrix observed features |
|---|---:|---:|---|---:|---:|---:|
| Bacteria | 6,386 | 741 | 2 / 4 / 503 | 416 (1,122) | 210 | 64,934 |
| Fungi | 7,730 | 926 | 2 / 3 / 473 | 569 (1,499) | 302 | 61,563 |

Full original-code distributions are retained in
[bacteria_class_distribution.tsv](../../reports/2026-11-16/bacteria_class_distribution.tsv)
and [fungi_class_distribution.tsv](../../reports/2026-11-16/fungi_class_distribution.tsv).
Both matrices have 65,536 declared columns. The identity `4**8 = 65,536` does
not establish historical feature-ID-to-8-mer order, training-partition observed
dimension or reproduction of raw 8-mer extraction.

The gene count sources are the read-only archive's
`Program/00_前置處理/GenePreprocess/GenePreprocess/bin/Debug/bac_instance/`
and corresponding `fun_instance/`. Their archive identifiers identify numeric
class/sample count files. `ClassName_bac.txt` and `ClassName_fun.txt` supply
CP950 class-code-to-name tables. They do **not** supply original accession IDs,
sequence records, taxids, parent lineage or explicit biological ranks. Original
accessions cannot be recovered from the count filenames by the current reader.

Names such as `Photobacterium`, `Vibrio` and `Sparassis` are not evidence that
every class has a verified Genus rank. The saved names also include strings such
as `Polyporaceae[G]`, `uncultured_Tricholomataceae` and
`Capnodiales incertae sedis[G]`; preserve these strings as archive metadata
without interpreting their suffixes as a validated rank schema. The present
task is original numeric taxon-class classification. Calling it Family or
Genus classification requires additional evidence and a reviewed mapping.

The inspected archive did not recover priority raw sequences/FASTA or the
sample-to-accession bridge. A new database or FASTA without that bridge is a
new dataset, not proof that the preserved count rows have been identified.

## Required taxonomy and sequence evidence

For label confirmation alone, a trustworthy original class dictionary may be
sufficient **if** it establishes the exact taxon, rank and common taxon membership
of every row assigned to each original code. That reviewed mapping must connect
`original_class_code -> taxid -> target-rank taxid/lineage`, with an authoritative
source and unresolved cases explicitly recorded. A name lookup by itself is not
that evidence. A class-level dictionary does not recover original sequences,
validate raw extraction or establish biological independence.

The stronger record-level evidence bundle, needed to verify those additional
properties, establishes the complete chain for each retained observation:

`archive sample ID ↔ accession.version ↔ sequence record ↔ taxid ↔ ranked lineage`.

The evidence bundle needs the original sequence file(s), the sample bridge,
ranked taxonomy records and a provenance manifest. Record the source database,
sequence release or retrieval snapshot, taxonomy release/snapshot, retrieval
date, source paths/URLs, checksums and mapping procedure. Preserve original
identifiers and class codes alongside resolved identifiers; record any merged,
deleted, ambiguous or unresolved identifier explicitly. Do not resolve names by
an unreviewed online name match or silently select the first candidate.

A minimal auditable TSV bundle can separate record-level and lineage metadata:

NCBI identifies taxonomic nodes with TaxIds and records primary/secondary names
and merged identifiers; retain IDs and a dated source rather than relying on name
strings alone. See [NCBI Taxonomy Help](https://www.ncbi.nlm.nih.gov/sites/books/NBK53758/).
For a reproducible local taxonomy snapshot, the official bulk files include
`nodes.dmp`, `names.dmp`, `merged.dmp` and `delnodes.dmp`; expanded dumps also
provide lineage metadata. See [NLM bulk-download guidance](https://support.nlm.nih.gov/knowledgebase/article/KA-05543/en-us).
This is a requirement for acquiring reference evidence, not an automatic mapping
or an upload of laboratory data.

```text
# sample_taxonomy.tsv: one reviewed row per archive sample
dataset	archive_sample_id	original_class_code	accession_version	sequence_record_id	sequence_sha256	taxid	lineage_id	target_rank	target_taxid	mapping_status	evidence_id

# ranked_lineage.tsv: one row per node in each referenced lineage
lineage_id	node_order	taxid	parent_taxid	rank	scientific_name

# provenance.tsv: evidence/release manifest
evidence_id	source_location	sequence_release	taxonomy_release	retrieved_at	file_sha256	mapping_method
```

These are proposed evidence schemas, not existing files or implemented reader
interfaces. The manifest must also locate the sequence file containing each
`sequence_record_id`; use separate evidence records where multiple files or
releases are involved. Document sequence-hash normalization so that the digest
is reproducible. Retain the original class-name table as provenance. A name-only
`ClassName` table is insufficient to populate missing fields.

Before using a Family/Genus label, audit complete archive-ID coverage, unique
sample joins, accession-version resolution, sequence/hash agreement and the
explicit target-rank node in every lineage. Record missing or ambiguous joins,
target-rank gaps and disagreement with original labels in a review table.
Unknown ranks do not become a guessed label. Any accepted remapping must retain
the original-code-to-target mapping and explain changes in the prediction target.

Raw sequences and source metadata are also needed to assess biological
independence: repeated accessions, identical sequences, sequence similarity,
and shared isolate/specimen or other relevant sampling units. Specify and
review the grouping unit, sequence-normalization rule and any near-similarity
method/threshold before splitting; no similarity threshold is chosen here.
Exact count vectors alone neither recover sequences nor establish independent
observations. Keep conflicting-label groups together and flag them for review;
do not automatically repair labels or delete rows.

## Small classes and the implemented split requirements

Every outer fold must contain all retained classes in three disjoint roles:
`Dfit` for vocabulary, supervised feature scoring and fitted parameters;
`Dsearch` for candidate/PSO fitness; `Dtest` for final evaluation only. A class
with two rows cannot occupy all three roles even if its rows are independent.
Changing the random seed, balancing, or assigning two folds cannot create a
third independent observation.

Eligibility must count groups at the reviewed independence level, not only
rows. For five outer test folds with all-class test coverage, at least five
distinct groups per retained class is necessary. It is **not sufficient** for
every outer training portion to admit the required all-class fit/search split,
or for an approximate stratification algorithm to find such a split.

The current [group_aware_splits](../../src/afs_rgmnb/data/splitting.py) does the
following:

1. Hashes canonical sparse count vectors, excluding labels, so all equal vectors
   stay together, including vectors with conflicting labels. It retains every row.
2. Rejects a class with fewer distinct vector groups than the requested number
   of outer folds. Outer `StratifiedGroupKFold` allocations must then actually
   contain every class in each `Dtest`.
3. Within each outer training portion, constructs nested SGKF candidates with
   `k = max(2, floor(1/min(f, 1-f) + 0.5))`, where `f` is the requested search
   fraction. Only candidates containing every class in both `Dfit` and `Dsearch`
   are eligible. It selects by search-size/class-proportion allocation error,
   with deterministic first-fold ties, without evaluating a model or test score.
4. Fails if any outer fold has no eligible inner candidate. Successful plans
   verify zero group/row overlap, all rows retained once per three-role split,
   every row used for outer test exactly once, and realized class distributions.

For the existing illustrative `f=0.2`, nested SGKF has five candidates; this
does not impose five groups per class in each outer training portion as an
independent theorem. The actual selected fit/search class coverage is checked.
Conversely, passing the outer five-group precheck does not prove an inner
candidate exists. A heuristic failure is not a proof that no conceivable
allocation exists, but the run must stop until a documented protocol is reviewed.

The current function accepts no external accession/specimen/sequence-similarity
group array. It protects exact count-vector groups only. A formal biological
group protocol will require explicit integration and validation of those groups;
it cannot be claimed merely by preparing a taxonomy table.

A read-only audit of the existing exported matrices on 2026-11-17 found:

| Source | Distinct count-vector groups | Minimum vector groups per class | Classes with <5 vector groups | Classes with ≥5 vector groups |
|---|---:|---:|---:|---:|
| Bacteria | 6,350 | 1 | 419 | 322 |
| Fungi | 7,284 | 1 | 581 | 345 |

The all-source fivefold preflight fails without dropping classes (first failure:
Bacteria code 5; Fungi code 0). These are count-vector safety groups, **not**
verified biological independence groups. The source audit also found 31/225
repeated-vector groups for Bacteria/Fungi and 0/16 conflicting-label groups.
The November 16 two-way gene conversion checks remain representation audits;
they do not satisfy the three-role evaluation protocol.

## Protocol choices requiring a documented scientific decision

The following are alternatives to prepare for review, not automatic data rules:

- **Prefer a scientifically appropriate, verified coarser taxonomic target when
  it matches the research question.** After lineage recovery, pooling descendants
  at a chosen rank may supply more groups per class while retaining all records.
  This changes the prediction target; document the rank, mapping, unresolved
  cases, class distribution and comparability to the original task. Re-audit all
  fold/role group coverage. Choosing Family solely because a name looks related
  is insufficient, and a coarser rank is not guaranteed to be feasible.
- **Use an explicitly approved eligible-class subset if that narrower target is
  scientifically justified.** Declare retained original codes before model
  evaluation and keep all records in every retained class with their natural
  counts. Publish excluded codes, sample/group counts, reasons and population
  coverage. The row-based ≥5 subsets (325 Bacteria classes/5,264 rows;
  357 Fungi classes/6,231 rows) are descriptive only and are not approved subsets;
  the vector-group counts above already show that this row rule is inadequate.
- **Retain the full original target and recover sufficient independent data.**
  If neither target coarsening nor a restricted population is justified, more
  independent observations are needed. Until then, report conversion/source
  audits and the infeasibility instead of gene performance results.

Do not automatically merge rare labels, exclude classes, resample, equalize
class sizes or convert the task to binary. Different fractions/fold counts are
reviewable protocol changes, not a remedy for a two-observation three-role
class. No final fraction, seed schedule or inclusion threshold is approved here.

## Next actionable local audit

The existing exports permit a quick feasibility check without extracting the
archive again or fitting a classifier. From the repository root in PowerShell:

```powershell
git status --short
git check-ignore -v outputs/2026-11-16/verified_run/bacteria_counts.npz outputs/2026-11-16/verified_run/fungi_counts.npz
@'
from pathlib import Path
import numpy as np
from scipy import sparse
from afs_rgmnb.data.splitting import group_count_vectors, group_aware_splits
root = Path("outputs/2026-11-16/verified_run")
for key in ("bacteria", "fungi"):
    X = sparse.load_npz(root / f"{key}_counts.npz")
    y = np.load(root / f"{key}_labels.npy", allow_pickle=False)
    groups = group_count_vectors(X)
    counts = [np.unique(groups[y == cls]).size for cls in np.unique(y)]
    print(key, "rows", X.shape[0], "classes", len(counts),
          "vector_groups", np.unique(groups).size,
          "classes_with_lt5_vector_groups", sum(n < 5 for n in counts))
    try:
        group_aware_splits(X, y, n_splits=5, search_fraction=0.2, seed=42)
    except ValueError as exc:
        print("INFEASIBLE:", exc)
'@ | python -
```

The numerical settings in this diagnostic match the existing development/audit
example; they are not final experimental settings. The command was executed
successfully on 2026-11-17 and produced the expected explicit feasibility errors.
It writes no research artifacts and does not establish classification accuracy.

The next evidence acquisition is the original sample/accession bridge, sequences
and ranked taxonomy release, followed by a reviewed mapping/independence audit.
Keep those materials and detailed record-level outputs ignored or external. No
research upload, historical executable, guessed taxonomy join or automatic class
filter is needed to complete this requirements stage.
