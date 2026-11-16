# Dataset status - verified November 16 audit

The three priority datasets were read in full with original multiclass codes preserved.
No samples or classes were excluded, balanced, or converted to binary in this audit.
These sources contain **precomputed occurrence counts**. They were not re-extracted
from raw documents or DNA sequences.

| Dataset | Source count records | Valid retained records | Classes | Observed features, full matrix | Declared columns | Meets n>=500 and d>=5,000 |
|---|---:|---:|---:|---:|---:|---|
| 20Newsgroup | 18,828 | 18,828 | 20 | 83,415 | 83,635 | Yes |
| Bacteria | 6,386 | 6,386 | 741 | 64,934 | 65,536 | Yes |
| Fungi | 7,730 | 7,730 | 926 | 61,563 | 65,536 | Yes |
| MDR88 (optional reduced source) | 14,680 | 14,680 | 3 | 12,331 | 12,511 | Yes, different population from thesis table |

## Published statistics versus available files

[Tsai (2019), section 4.1/Table 4.1](https://drive.google.com/file/d/1a7DUzvxAkILtds3-VvQh8Jk4P1rccGol/view)
reports 20News **18,828/83,635/20**, Bacteria **6,386/65,536/741**,
Fungi **7,730/65,536/926**, and MDR88 **14,687/12,511/4**
(samples/features/classes). These are published figures, not observed-fold dimensions.
The priority sample/class counts match; the all-zero source columns explain why
observed feature counts are lower than declared dimensions. The DNA universe
4**8=65,536 does not establish which 8-mers occur in any training partition.

MDR88 is actually available locally as `MDR88.txt`, `MDR88Output.txt` and
`MDR88TotalWord.txt`. Both the text file and count file contain **14,680** records,
with 7,919 serious-injury, 6,224 malfunction and 537 death records. The seven-record
`other` class from the thesis table is absent from these local input files; this
continuation removed nothing. MDR88 remains optional. Its raw text has two
non-ASCII bytes, requiring an explicit encoding decision before re-extraction.

## Exact priority sources

Paths below are relative to the ignored `107-???/` archive:

| Dataset | File or directory | Actual format and identifiers |
|---|---|---|
| 20News | `Program/00_????/text/20Newsgroup/20newsOutput.txt` | ASCII rows: class,historical-fold,one-based-feature,count,...; archive row IDs, not Message-IDs |
| Bacteria | `Program/00_????/GenePreprocess/GenePreprocess/bin/Debug/bac_instance/` | Numeric class/sample `.txt` files: zero-based-feature count; `ClassName_bac.txt` decoded CP950 |
| Fungi | Corresponding `fun_instance/` directory | Same sparse schema; `ClassName_fun.txt` decoded CP950 |
| MDR88 | `Program/00_????/text/MDR88/` | Raw labelled text and precomputed counts; only three classes retained upstream |

There were **zero schema-invalid records, missing class codes, duplicate archive
identifiers, or all-zero count rows** in all priority sources. Readers fail on
invalid rows rather than silently skipping them. Raw-record counts *before historic
preprocessing* are unknown for the priority datasets; source count records are not
raw-sequence/text counts.

## Class distributions and evaluation eligibility

Full distributions are saved as derived tables:
[20News](20news_class_distribution.tsv), [Bacteria](bacteria_class_distribution.tsv),
[Fungi](fungi_class_distribution.tsv). No equal-size class sampling was applied.

| Dataset | Minimum / median / maximum class size | Classes with fewer than 5 samples | Samples in those classes | Classes with exactly 2 samples |
|---|---|---:|---:|---:|
| 20Newsgroup | 628 / 981.5 / 999 | 0 | 0 | 0 |
| Bacteria | 2 / 4 / 503 | 416 | 1122 | 210 |
| Fungi | 2 / 3 / 473 | 569 | 1499 | 302 |

20News supports a stratified fivefold outer split and internal fit/search audit.
Gene datasets cannot place a two-sample class in three disjoint partitions; their
full-class fivefold thesis evaluation remains unresolved. Counting classes with
at least five samples is descriptive, not an instruction to exclude the others;
it also does not guarantee every inner search partition contains every class.

**Biological rank remains unverified.** Names alone do not establish Genus, and
Family cannot be recovered without parent lineage or accession mapping. The
retained gene task is presently identified as original numeric taxon-class
classification, not a validated Family/Genus experiment. Obtaining original
sequences and a reviewed taxonomic map is the next dataset task.
