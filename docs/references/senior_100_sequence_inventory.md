# Chen (2012): recovered nucleotide-sequence inventory

Inspected locally on 2026-11-17. Read-only source:
`reference/100-陳泓宇/`. This new archive contains genuine nucleotide FASTA
records, unlike the previously inspected 107 preserved-count source. The
inspection executed no historical Java class, binary or installer, changed no
reference file, and uploaded no laboratory material.

## Identity and inspection scope

`R76991135.pdf` identifies 陳泓宇 and the thesis *結合階層式概念與特徵選取技術於基因序列分類之研究*
(2012). Its PDF metadata, §4.1 (printed p.33 / PDF p.43), Table 4.1, and §4.2.5
(printed p.41 / PDF p.51) were inspected. Both relevant PDF pages were rendered
and visually checked against extracted text. The folder contains 10 `.fasta`
files, 977 `.txt`, 38 `.java`, 38 `.class`, one PDF and one DOCX. No compressed
archive was present in that recursive inventory. Compiled `.class` files were
not executed.

All ten FASTA files were read completely. SHA-256 identifies **four unique
contents**, with filtered sources duplicated under three program directories.
The text/source inventory and record-level evidence remain ignored under
`outputs/reference_inspection/100/`.

## Actual sequence sources

All paths below are relative to `reference/100-陳泓宇/`:

| Local source | Verified records | Historical status |
|---|---:|---|
| `資料檔/過濾前資料檔/Bacteria_3672_Nonsingleton.fasta` | 3,672 | Singleton-filtered source before the later <10-class filter |
| `資料檔/過濾前資料檔/Fungi_7730_Nonsingleton.fasta` | 7,730 | Singleton-filtered source before the later <10-class filter |
| `資料檔/過濾後資料檔/Bacteria_2035_Nonsingleton.fasta` | 2,035 | Historical <10-class filtering already applied |
| `資料檔/過濾後資料檔/Fungi_4954_Nonsingleton.fasta` | 4,954 | Historical <10-class filtering already applied |

The last two files also occur, byte-identically, in
`My_method_new/{bacteria,fungi}/classification/`,
`My_method_new/{bacteria,fungi}/feature_selection/entropy_calculated/`, and
`My_method_new/{bacteria,fungi}/feature_selection/number_of_feature_choosed/`.
These are duplicate copies, not additional observations to concatenate.

Each FASTA header has an accession-like identifier, a tab, and six semicolon-
separated taxonomy strings. It is followed by nucleotide sequence text. Every
record has all six nonempty taxonomy fields and a nonempty sequence. The actual
alphabet is A/C/G/T plus IUPAC ambiguity letters; these are not integer vectors
or amino-acid translations.

The archived interpretation of the six fields is supported by
`My_method_new/bacteria/classification/classification_genus.java`, lines 449-465:
identifier, sequence and hierarchy are read separately, and fields are explicitly
mapped to Domain/Kingdom, Phylum, Class, Order, Family and Genus. The thesis's
sequence example and §4.1 agree with this interpretation. This confirms an
**archived rank schema**, not a current taxonomy reconciliation or TaxId mapping.

## Verified sequence and archived-label summaries

Class counts below use the complete lineage prefix through the target rank,
preserving ancestor context. These counts also equal the number of distinct
rank-name strings in these files.

| Source | Length min / median / max (bases) | Family classes | Genus classes | Family / Genus minimum rows | Full-source observed valid DNA 8-mers |
|---|---|---:|---:|---|---:|
| Bacteria 3,672 | 204 / 1,470 / 1,987 | 138 | 500 | 2 / 2 | 62,646 |
| Fungi 7,730 | 47 / 912 / 2,128 | 272 | 926 | 2 / 2 | 61,563 |
| Bacteria 2,035 | 204 / 1,474 / 1,833 | 56 | 79 | 10 / 10 | 56,026 |
| Fungi 4,954 | 130 / 887 / 1,800 | 80 | 159 | 10 / 10 | 52,197 |

The feature inventory was a read-only rolling-window scan: k=8, stride one,
ACGT only, ambiguity symbols break windows, no reverse-complement merging.
Repeated windows were counted repeatedly. It found respectively 5,227,950;
7,226,107; 2,897,595; and 4,429,711 valid window occurrences. These figures
verify the presence of more than 5,000 usable full-source feature types; they
are **not** Dfit-only dimensions, exported matrix validation, a reproduction of
the original classifier, or a classification performance experiment.

The historical 65,536 theoretical DNA 8-mer space is separate from these observed
inventories. No numeric feature ordering is inferred for the 107 count matrices.

## Duplicates, ambiguity and small-class feasibility

| Source | Unique accession-like IDs | Unique exact sequence strings | Repeated sequence groups / participating rows | Ambiguity-containing records | Family / Genus classes with <5 exact-sequence groups |
|---|---:|---:|---|---:|---|
| Bacteria 3,672 | 3,660 | 3,652 | 16 / 36 | 1,647 | 23 / 285 |
| Fungi 7,730 | 7,730 | 7,286 | 223 / 667 | 1,822 | 104 / 580 |
| Bacteria 2,035 | 2,023 | 2,023 | 8 / 20 | 941 | 0 / 0 |
| Fungi 4,954 | 4,954 | 4,575 | 171 / 550 | 1,242 | 0 / 0 |

Exact sequence strings here are uppercased and include ambiguity letters as
written; no alignment, orientation normalization or near-similarity grouping was
performed. These are safety groups, not proof of independent organisms.

Both Bacteria variants contain 12 excess records sharing accession-like IDs;
one repeated ID has more than one sequence string but the same archived lineage.
Consequently, exact-sequence grouping alone is insufficient to keep all shared
accession records together. Fungi has no repeated header IDs in these files,
but identical sequences cross archived labels: the 7,730-record file has 5
conflicting groups at Family and 16 at Genus; the 4,954-record file has 1 and 9.
No record or label was automatically removed, changed, or balanced.

The historically filtered Bacteria source has at least 9 exact-sequence groups
per Family/Genus class. The filtered Fungi source has at least 9 per Family and
5 per Genus. Passing this necessary group-count condition **does not establish
actual fivefold fit/search/test feasibility**; the allocation audit below fails.
The current production splitter still needs explicit integration of combined
accession, exact-sequence and any scientifically selected biological grouping
rules.

## Protocol-only allocation check of the historical filtered sources

The existing `group_aware_splits` was reused without changing its algorithm:
five outer folds, search fraction .2, seed 42, Python 3.12.7, NumPy 1.26.4,
SciPy 1.11.4 and scikit-learn 1.4.2. Two grouping definitions were inspected:
uppercased exact sequence identity, and connected components joining any rows
that share either an accession-like ID or an exact sequence. Group keys were
represented by distinct positive integer markers in a one-column CSR matrix,
so equal marker vectors encode the desired grouping. **These marker vectors
are not DNA features, and no classifier, JMH score or performance metric was
computed.** Complete lineage prefixes define the archived rank labels; unique
archive-row IDs preserve repeated accessions without dropping records.

| Population | Rank | Grouping | Groups | Current split result |
|---|---|---|---:|---|
| Bacteria 2,035 | Family | Exact sequence | 2,023 | Outer fold 0 lacks a required test class |
| Bacteria 2,035 | Genus | Exact sequence | 2,023 | Outer fold 0 has no nested candidate covering all classes |
| Bacteria 2,035 | Family | Accession/sequence union | 2,016 | Outer fold 0 lacks a required test class |
| Bacteria 2,035 | Genus | Accession/sequence union | 2,016 | Outer fold 0 lacks a required test class |
| Fungi 4,954 | Family | Exact sequence | 4,575 | Outer fold 0 lacks a required test class |
| Fungi 4,954 | Genus | Exact sequence | 4,575 | Outer fold 0 lacks a required test class |
| Fungi 4,954 | Family | Accession/sequence union | 4,575 | Outer fold 0 lacks a required test class |
| Fungi 4,954 | Genus | Accession/sequence union | 4,575 | Outer fold 0 lacks a required test class |

Every run stopped explicitly at the first failure; no class/record removal or
balancing occurred. Consequently, there is no successful fivefold all-class plan
to report. These are failures of the **current approximate SGKF allocation with
the stated marker representation and seed**, not proofs that no conceivable
group-disjoint allocation exists. Alternative allocations, ranks or protocol
settings require review rather than searching seeds until a favorable model
result appears. Group-ID encoding can change the heuristic's realized allocation;
the marker result does not predict a future actual-8-mer split byte-for-byte.
The private `filtered_split_feasibility.json` records the source fingerprints,
exact settings, grouping definitions and errors. It does not claim verified
biological independence or final experimental eligibility.

## Historical curation and applicability

The thesis §4.1 reports initial NCBI collections of 4,354 Bacteria and 8,506
Fungi sequences, reduced by singleton removal to 3,672 and 7,730. The initial
4,354/8,506 populations were **not found as FASTA files** in this folder.
Thus the recovered files contain original sequence strings, but are already
historically curated datasets rather than complete unfiltered acquisitions.

Section 4.2.5 describes a further exclusion of classes with fewer than ten
records, yielding 2,035 and 4,954 records. These historical subsets are not newly
approved thesis populations. Selecting one requires an explicit scientific
justification and disclosure of its source filtering; it is not class balancing.

The newly recovered Fungi 7,730 aggregate class count and observed 8-mer count
match those of the 107 preserved Fungi source. This is a lead for provenance
reconstruction, **not proof of row correspondence or exact matrix equivalence**.
The Bacteria 3,672 population differs from the 107 source's 6,386 records.
Do not merge or substitute these datasets based only on matching domain names.

For subsequent work: preserve file fingerprints, create unique archive-row IDs
alongside accession strings, document archived/current rank handling, audit
ambiguous sequences and conflicting labels, implement the reviewed grouping
unit, learn the observed feature list on Dfit only, and check all fold/role class
coverage. All four sequence populations meet n>=500 and their full-source
inventories exceed 5,000 features; final experimental eligibility is still
conditional on those checks.
