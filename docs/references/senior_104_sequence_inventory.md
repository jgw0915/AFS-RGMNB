# Lin (academic-year 104) nucleotide-source inspection

Inspected 2026-11-17. Reference root: `reference/104-林修弘/`.
The directory was inspected read-only: no historical executable or installer was
run, no reference file was edited, and no material was uploaded. This inventory
changes the earlier conclusion about what is available locally: the academic-year
107 folder alone supplied counts, but this newly supplied folder contains
**nucleotide sequences before amino-acid translation**.

## Confirmed nucleotide inputs

Both files below are ASCII, FASTA-like text. Each record has a `>` identifier
and tab-separated semicolon taxonomy path, followed by a nucleotide sequence.
In these particular files every record occupies two lines. Headers and sequence
contents, rather than the `.txt` extension, establish the representation.

| Source relative to reference root | Representation | Records | Unique full header identifiers | Length min / median / max | Full-source observed 8-mers |
|---|---|---:|---:|---|---:|
| `預處理為氨基酸/Project1/16SRNA.txt` | Nucleotide sequence with DNA alphabet and IUPAC ambiguity symbols | 6,386 | 6,386 | 204 / 1,466 / 3,035 | 64,934 |
| `預處理為氨基酸/Project1/Fungi.txt` | Nucleotide sequence with RNA U and IUPAC ambiguity symbols | 7,730 | 7,730 | 47 / 912 / 2,128 | 61,563 |

Both sources satisfy the sample threshold and contain over 5,000 observed
8-mers under the explicitly audited representation: uppercase, U-to-T,
overlapping windows with stride one, and break windows at any non-ACGT symbol.
The observed dimensions are descriptive full-source inventories, **not** a
vocabulary fitted for model evaluation. Dfit must establish each fold's columns.
No matrix, split, model or classification score was generated in this inspection.

| Source | SHA-256 of original bytes |
|---|---|
| `16SRNA.txt` | `54783036f1e65f5609f82f1c44d5b7bbf4d811c310e1ac528104b7631d8c9914` |
| `Fungi.txt` | `4cf3d15c1360e7efb21ff863aad61bd236e943cd1e9b23ee729fd9a37e655bf1` |

The first header identifier component has accession-like syntax; Bacteria also
has a second pipe-separated database-style identifier. These are recovered
archive identifiers, not online-confirmed accession.version/TaxId joins. Detailed
headers and record-level evidence stay in ignored local inspection artifacts.

## Source-code and thesis evidence

`預處理為氨基酸/Project1/main.cpp` is readable UTF-8 with a BOM. Static inspection
confirms the following pipeline; the program was not compiled or executed:

- `FileInputFungi`, lines 22-129, opens `Fungi.txt` at line 30, separates header
  metadata and assigns sequence text to `Feature` at line 113.
- `FileInputRDP`, lines 132-241, opens `16SRNA.txt` at line 140 and assigns
  nucleotide text at line 228.
- `Fungitest` / `RDPtest`, beginning at lines 244 / 310, transform nucleotide
  symbols before translation. These routines include deterministic replacement
  of ambiguity symbols; that historical choice is **not** silently adopted by
  the current thesis's canonical 8-mer input.
- `Fungipro` / `RDPpro`, beginning at lines 566 / 1214, translate reading frames.
  `FungiproS` / `RDPproS`, lines 1043 / 1679, implement alternative start handling.
- Output functions at lines 1850-1993 write `RDP1/2/3/S.txt` and
  `Fungi1/2/3/S.txt`. These are translated outputs, not original DNA.

`林修弘碩士論文提交版.docx`, Chapter 4 section 4.1, reports an original Bacteria
population of 7,045 sequences and removal of singleton-label classes before
leave-one-out evaluation, leaving 6,386 records. Thus the recovered input is
raw **relative to feature extraction**, but cannot be claimed to be the complete,
unfiltered database download. The thesis's table and example contain historical
label/count inconsistencies; their rank counts are not substituted for inspected
file contents. It discusses amino-acid 3-mer extraction after genetic-code
translation. That is a different representation from direct DNA 8-mer counts.

## Other data files are translated variants

The four program trees beginning with `4.17 - 三种序列结合/`,
`三种序列结合 - 计算各类别值对应笔数/`, `直接運行簡易貝氏分類器/` and
`计算各类别值对应笔数/` contain duplicated copies and variations of
`Fungi1/2/3/S.txt`, `RDP1/2/3/S.txt`, `FungiSN.txt` and `RDPSN.txt`.
Their sequence bodies contain amino-acid symbols beyond the nucleotide IUPAC
alphabet, and the preprocessing source establishes their origin. Do not apply
the DNA extractor to those translated sequence bodies.

Some directories are explicitly named as versions with unclassified RDP records
removed; they are historical alternatives, not independently collected datasets.
Record counts also differ across translated variants (for example 7,418-record
`FungiS.txt` versus 7,730 records in the nucleotide input). No historical
exclusion or translation was reproduced here. The two nucleotide input files
above are the starting candidates for a new audited extraction pipeline.

The full folder contains 219 files / 435,476,064 bytes, including five C++ sources,
75 text files, build products, thesis files and one workbook. No ZIP/RAR/7z/gzip
archive was present in this directory inventory. Binary build products were not
treated as dataset evidence and were not executed.

## Taxonomy and duplicate limitations remain

The new headers recover important missing evidence, but they do not by
themselves certify a current biological target or independent observations:

- Bacteria taxonomy paths contain 4, 6, 7, 8 or 9 fields including `Root`
  (152 / 169 / 4,016 / 414 / 1,635 records). A fixed split position cannot
  automatically mean the same taxonomic rank across those paths. Historical
  lineage/schema rules and authoritative rank mappings need review.
- Fungi has seven fields for all 7,730 records, with terminal labels consistent
  with the historical genus-oriented schema. There are 926 distinct terminal
  strings. Some contain `uncultured`, `incertae sedis` or bracketed rank markers;
  these remain unresolved metadata, not automatically valid genus names.
- No dated taxonomy release, accession-version verification, TaxId lineage table
  or specimen/isolate grouping file was established by this inspection.
- The recovered record totals and full observed 8-mer dimensions match those
  measured in the academic-year 107 preserved counts. This supports investigation
  of a common lineage; it **does not prove** a complete row/label/feature-ID bridge
  or exact reproduction of those matrices.

| Source | Records with ambiguity symbols | Unsupported nucleotide symbols after U-to-T | Empty records / no valid 8-mer | Exact normalized sequence groups | Repeated groups / participating records |
|---|---:|---:|---|---:|---|
| Bacteria | 2,557 | 0 | 0 / 0 | 6,350 | 31 / 67 |
| Fungi | 1,822 | 0 | 0 / 0 | 7,286 | 223 / 667 |

Exact sequence identity here ignores case and maps U to T, while retaining
ambiguity symbols. It does not assess reverse complements, near similarity,
accession equivalence or shared biological sampling units. Fungi has 16 exact
sequence groups with conflicting terminal taxon strings; keep them together
and flag them rather than automatically repair or delete labels. Bacteria has
no conflicting groups for the inspected fixed-position metadata strings, but
its variable-depth paths prevent treating that statement as a verified rank audit.

The following are literal-string diagnostics, not approved biological targets:

| Source / literal metadata target | Strings | Minimum rows / exact sequence groups per string | Strings with <5 rows / <5 sequence groups | Missing-position records |
|---|---:|---|---|---:|
| Bacteria terminal field | 742 | 1 / 1 | 417 / 420 | 0 |
| Bacteria field 5 (zero-based, structural family position only) | 194 | 2 / 2 | 49 / 50 | 152 |
| Fungi terminal field | 926 | 2 / 1 | 569 / 580 | 0 |
| Fungi field 5 (zero-based, structural family position only) | 272 | 2 / 1 | 101 / 104 | 0 |

The Bacteria terminal-string count is 742 rather than the old count source's
741 numeric classes. This is another reason to verify the actual historical
label construction instead of assuming that last-field text is the old label.
The Bacteria structural-position diagnostic includes one explicit missing-field
category for 152 records; those records were retained. Variable-depth paths mean
that the remaining strings cannot automatically be asserted to have a uniform
rank. Both Fungi targets include unresolved labels; the structural family
diagnostic also has five exact-sequence groups with conflicting target strings.
These diagnostics **are not approved Family/Genus tasks or subsets**. Fivefold
all-class fit/search/test feasibility remains a separate scientific decision.
These sequence-group numbers differ slightly from count-vector grouping because
feature extraction can map different sequences to equal feature vectors.

## Next use in this thesis

1. Preserve the nucleotide inputs and hashes; use these rather than translated
   files for the direct 8-mer pipeline. Keep original files read-only.
2. Parse headers with an explicit schema and build a reviewed record/lineage
   manifest, retaining original IDs and taxon strings. Resolve variable-depth
   Bacteria paths and uncertain Fungi nodes before claiming Family/Genus targets.
3. Explicitly configure U handling, ambiguous-window policy and duplicate grouping;
   produce sparse occurrence counts with vocabulary fitting confined to Dfit.
4. Re-evaluate all-class grouped fit/search/test feasibility after the target and
   biological grouping protocol are justified. Do not silently filter rare labels.
5. If comparison to the old count sources is needed, inspect their source program's
   class-code construction, filtering and within-class record order. Establish a
   row-label bridge and feature-index mapping, then compare sample count signatures
   and actual per-feature counts. Matching sample totals or dimensions alone is
   insufficient, and a count signature may be ambiguous for duplicate vectors.

Private inspection evidence is in `outputs/reference_inspection/104/`:
`file_manifest.json`, `raw_sequence_summary.json`, positional-label distributions,
decoded source and extracted thesis text. Only this derived inventory is intended
for Git; headers, sequences, historical source programs and full manifests remain
ignored. See [gene experiment requirements](../experiments/gene_experiment_requirements.md)
for the broader protocol conditions.
