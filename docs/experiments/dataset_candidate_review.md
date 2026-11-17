# Week 16 dataset candidates and current suitability

Reviewed 2026-11-17 against the user's current objective: at least 500 records,
at least 5,000 **observed** features, single-label multiclass prediction, and
nonnegative integer occurrence counts. Natural class sizes need not be equal.
No class balancing, binary conversion or performance-driven class filtering is
part of this review. A candidate is not a finalized experiment configuration.

## PDF inventory and interpretation

Read-only source: `reference/Master_Thesis_Week_16_2026-10-27_Kuan-Wei_Chiang_Dataset_Inspection.pdf`.
All four pages were text-extracted and rendered for visual inspection. Pages
1/2/3 contain 13 document, 10 DNA and 7 further document entries respectively:
**30 table entries**. Page 4 lists S1-S19 source links. The tables below preserve
the PDF's reported values separately from new measurements and source corrections.
No dataset has been silently substituted for a similarly named release.

The PDF's DNA table proposes 7-mers and a possible dictionary of 16,384. The
active thesis implementation uses overlapping **8-mers**, with 65,536 possible
types. Neither possible dictionary size proves 5,000 observed types. Training
fold inventories, classes, counts, groups and leakage must be checked on actual
files. The present review changes no thesis equation or extraction rule.

## All document entries from PDF page 1

Numbers in the first four columns are PDF-reported, not local extraction results.
“Candidate” always means subject to local source/feature/group/split validation.

| Dataset / release | Samples | Classes | Reported d | Current decision / required check |
|---|---:|---:|---:|---|
| 20 Newsgroups, scikit-learn | 18,846 | 20 | 130,107 | Strong raw-text candidate; separate release from senior's 18,828 rows. Refit vocabulary on Dfit. |
| WebKB, four-class processed version | 4,199 | 4 | 7,770 | Candidate; already tokenized. Document the released four-class population rather than call it the original seven-class corpus. |
| Reuters R8, single-label | 7,674 | 8 | 17,387 | Candidate; count released tokens. Verify natural class/group coverage; uneven counts are acceptable. |
| Reuters R52, single-label | 9,100 | 52 | 19,241 | Conditional; small classes require the same all-class grouped feasibility audit as other datasets. |
| CADE12, processed Portuguese pages | 40,983 | 12 | 193,997 | Candidate; preserve Portuguese preprocessing provenance. Do not apply an English stoplist/stemmer blindly. |
| BBC News, DePaul processed release | 2,225 | 5 | 6,167 | Good small candidate; these are preprocessed counts, not raw text. Prefer raw BBC for fold-fitted extraction where accessible. |
| LEDGAR, LexGLUE / LIBSVM | 80,000 | 100 | 19,996 | Candidate via raw legal text; published TF-IDF columns are not the proposed count representation. |
| SCOTUS, LexGLUE / LIBSVM | 7,800 | 13 | 126,405 | Conditional; source class descriptions differ (see below). Raw text available; rebuild counts and audit actual labels. |
| Sector | 9,619 | 105 | 55,197 | Conditional; verify original frequency values. Transformed/normalized files cannot simply be declared integer counts. |
| RCV1 multiclass, single-label derivative | 534,135 | 53 | 47,236 | Conditional, lower priority; transformed features, release-specific label mapping, source access and scale. |
| RCV1 binary | 697,641 | 2 | 47,236 | Exclude from the current main multiclass shortlist; also requires original count evidence. |
| real-sim | 72,309 | 2 | 20,958 | Exclude from main multiclass shortlist; original count representation unresolved. |
| news20.binary | 19,996 | 2 | 1,355,191 | Exclude from main multiclass shortlist; not the 20-class task. |

Primary-source checks:

- [scikit-learn raw 20News documentation](https://scikit-learn.org/1.4/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
  confirms 18,846 documents, 20 classes and explicit header/footer/quote controls.
  [Its vectorized loader](https://scikit-learn.org/1.4/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html)
  normalizes by default; `normalize=False` retains counts. For this repository's
  nested evaluation, raw text plus Dfit-only CountVectorizer is preferable to a
  precomputed dictionary. Published 130,107 does not fix future fold vocabularies.
- [Cardoso-Cachopo's source page](https://ana.cachopo.org/datasets-for-single-label-text-categorization)
  supplies class-plus-token lines, several preprocessing variants and research
  terms. It explicitly removed three of WebKB's seven original classes. These
  are released processed populations, not recovered original documents; retain
  their source identity and select a documented token-processing variant.
- [DePaul's BBC release](https://facweb.cs.depaul.edu/mobasher/classes/CSC575/Assignments/assign4-W2023.html)
  confirms 2,225 documents, five topics and 6,167 processed features, with an
  existing 80/20 split. Its first matrix column is a document ID, not a feature.
  [Greene's BBC source page](https://derekgreene.com/bbc/) advertises both raw text
  and processed Matrix Market frequencies; the original matrix is term-by-document.
  The direct page fetch returned 403 here, but its indexed source content was
  readable. No BBC download or local count validation was completed.
- [LIBSVM's release page](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html)
  provides LEDGAR/SCOTUS raw text alongside TF-IDF and describes Sector's
  transformation and the single-label RCV1 derivative. Raw text is the appropriate
  starting point for new integer counts. Its SCOTUS entry lists 13 classes;
  [LexGLUE's own table](https://github.com/coastalcph/lex-glue) describes 14.
  This difference remains unresolved pending file/label audit; do not relabel or
  drop a class to force agreement. LEDGAR's task has 100 classes.

## All DNA entries from PDF page 2

The PDF reports possible **7-mer d=16,384 for every row**, not measured features.
All require extraction; this thesis should audit its existing 8-mer policy.

| Dataset | PDF samples | Classes | Current decision |
|---|---:|---:|---|
| dummy_mouse_enhancers_ensembl | 1,210 | 2 | Binary; optional prototype only, outside main multiclass objective. |
| demo_coding_vs_intergenomic_seqs | 100,000 | 2 | Binary demonstration task; outside main multiclass objective. |
| demo_human_or_worm | 100,000 | 2 | Binary demonstration task; outside main multiclass objective. |
| drosophila_enhancers_stark | 6,914 | 2 | Binary; outside main multiclass objective. |
| human_enhancers_cohn | 27,791 | 2 | Binary; outside main multiclass objective. |
| human_enhancers_ensembl | 154,842 | 2 | Binary; outside main multiclass objective. |
| human_ensembl_regulatory | 289,061 | 3 | Promising larger multiclass functional sequence candidate; download/observed-dimension/group audit pending. |
| human_nontata_promoters | 36,131 | 2 | Binary; outside main multiclass objective. |
| human_ocr_ensembl | 174,756 | 2 | Binary; outside main multiclass objective. |
| UCI Splice-junction DNA | 3,190 | 3 | Locally audited below: suitable for the next controlled pipeline test, with explicit provenance/group caveats. |

[The Genomic Benchmarks paper, Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10150520/)
reports the sequence/class totals and the three regulatory categories.
[Project metadata](https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/datasets/human_ensembl_regulatory/metadata.yaml)
identifies enhancer, open-chromatin and promoter classes and the reference genome.
[Its data format](https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/main/datasets/README.md)
includes region, interval, strand and version metadata; these matter for overlap
and genomic-dependence grouping. The metadata's version 0 is explicitly a
development version. Pin a release/checksum and verify genomic overlap before
claims of independent testing. No external sequence corpus was downloaded here
except the small official UCI archive below. The PDF's Hugging Face aggregate
link is not treated as proof of a specific regulatory release or row count.

These are functional sequence tasks; they do not replace a Family/Genus task
under the same name. They can test the general high-dimensional MNB method if
that prediction target is explicitly adopted.

## All further document entries from PDF page 3

All feature dimensions in this table were **unmeasured in the PDF**. They remain
pending local Dfit-only token-count audits here.

| Dataset | PDF samples | Classes | Current decision |
|---|---:|---:|---|
| IMDB Sentiment | 50,000 | 2 | Binary; outside main multiclass objective. |
| AG News | 127,600 | 4 | Strong raw-text candidate; larger than BBC/20News, observed vocabulary still to measure. |
| DBpedia-14 | 630,000 | 14 | Candidate, later scale study; local counts/splits not audited. |
| Yelp Review Full | 700,000 | 5 | Candidate, later scale study; source terms and local counts/splits not audited. |
| Yahoo! Answers Topics | 1,460,000 | 10 | Candidate, later scale study; local counts/splits not audited. |
| Amazon Reviews Full | 3,650,000 | 5 | Candidate, expensive later study; local counts/splits not audited. |
| SMS Spam Collection | 5,574 | 2 | Binary; outside main multiclass objective, and observed vocabulary unverified. |

[TensorFlow Datasets' AG News description](https://www.tensorflow.org/datasets/catalog/ag_news_subset)
confirms four labels, title/description text and 120,000 train / 7,600 test
records. Its published source task is already a selected four-class release;
preserve that identity. This review did not rebalance it. Choose and record
title/description concatenation, tokenization and source-split policy before
extracting Dfit counts. No TensorFlow/GPU dependency is needed for the thesis's
count-based classifiers merely because the catalog is used as a source reference.

DBpedia/Yelp/Yahoo/Amazon are catalogued from the PDF only in this pass; their
current downloads, complete class distributions and actual observed features
are not independently verified. The page-4 references additionally mention
Amazon Commerce Reviews and TTC-3600 without table rows. They are not counted
among the 30 entries. The PDF calls TTC's processed dimension 4,814, below the
threshold; neither reference is promoted to the shortlist without a new audit.

## New local UCI Splice extraction and split audit

[The official UCI source](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)
provides 3,190 sequences of 60 bases and three labels: exon/intron boundary EI,
intron/exon boundary IE, and neither N. The original 60 categorical base columns
are not high-dimensional count features; direct overlapping 8-mer extraction
creates the representation required here. This preserves the original task.

Downloaded only the official 167,041-byte ZIP. SHA-256:
`3e7ce5dcbeec8c221f57dda495611b9d6ec9525551f445419f5c74cc38067e4e`.
Originals remain at `data/raw/uci_splice/retrieved-2026-11-17/source.zip` with
`provenance.json`. ZIP members were read in memory, not executed or unpacked
with unrestricted archive extraction. Derived arrays/audit are under
`data/extracted/uci_splice/dna8_audit-2026-11-17/`. Both directories are ignored.

| Measured property | Result |
|---|---|
| Retained records / labels | 3,190; EI=767, IE=768, N=1,655; no exclusions/relabeling/balancing |
| Sequence length | 60 for every record |
| Ambiguous-base records | 15; D/N/R/S break windows, no guessed substitutions |
| Extraction | Existing `sparse_dna_matrix`, k=8, stride 1, U-to-T convention, no reverse-complement merge |
| Matrix / observed features | 3,190 x 65,536 declared; 47,888 full-source observed |
| Nonzeros / zero rows | 168,027 / 0 |
| Stored values | int64, positive integers, 1-13; no TF-IDF/normalization |
| Sparsity | 99.919627% |
| Exact-vector groups | 3,005; 125 repeated groups / 310 participating rows |
| Conflicting-label vector groups | 1, retained together without label repair |
| Original instance IDs | 12 excess repeated IDs; keep original IDs plus unique source-row keys |
| Test model scores | None; no classifier was fitted |

Group-aware outer fivefold + nested search fraction .2, seed 42, pinned
scikit-learn 1.4.2 passed **all five actual split audits**. Every role contains
all three classes; sample and exact-vector group overlaps are zero. Fit-only
observed columns were determined separately in each fold:

| Fold | Dfit | Dsearch | Dtest | Dfit observed features |
|---|---:|---:|---:|---:|
| 0 | 2,044 | 510 | 636 | 40,523 |
| 1 | 2,039 | 503 | 648 | 40,890 |
| 2 | 2,044 | 510 | 636 | 40,604 |
| 3 | 2,054 | 499 | 637 | 40,795 |
| 4 | 2,052 | 505 | 633 | 40,940 |

Environment: Python 3.12.7, NumPy 1.26.4, SciPy 1.11.4. The corrected cached-source
audit took 1.740s, excluding the first download. An initial reporting-only
attribute typo was corrected; no input was changed. The complete measured
aggregate record is [uci_splice_inspection.json](uci_splice_inspection.json).
The local audit command is recorded in the checkpoint; a generic UCI reader/CLI
has not been added. Actual source IDs/arrays remain private/local, not in Git.

This passes the numerical size/count and **exact-vector-group split** requirements.
It is not a guarantee of biological independence: repeated instance names,
overlapping source regions, reverse complements and near-similar sequences need
review. In particular, the current splitter does not take the union of external
instance/source-region groups. Do not infer that original biological IDs were
held apart merely from zero vector-group overlap. Define that grouping before
formal performance claims. The three functional labels avoid a Family/Genus
taxonomy reconstruction but still need a clear target description.

## Recommended order

1. Continue existing 20News method development; obtain a separately identified
   raw-text 20News release for a clean fold-fitted benchmark.
2. Use UCI Splice for the next small raw-sequence pipeline validation after
   reviewing its instance/source-region grouping. It already exceeds n/d and
   has a successful exact-vector split audit; no accuracy claim is made.
3. Audit raw BBC next for a small five-class document task. R8/WebKB are useful
   processed-token alternatives; disclose their source filtering. Add AG News
   when a larger four-class document run is useful.
4. Evaluate human_ensembl_regulatory as the larger multiclass functional DNA task.
   Keep the recovered seniors' sequence populations as separate taxonomic
   candidates while resolving rank, grouping and small-class allocation.

Original official train/test splits must be retained as provenance. Pooling them
for thesis CV is a protocol decision, not an automatic loader behavior. Candidate
eligibility must not be decided by final test accuracy. This inspection runs no
formal accuracy comparison and changes no scientific source or professor report.
