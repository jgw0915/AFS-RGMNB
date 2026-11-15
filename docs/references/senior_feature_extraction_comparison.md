# Senior methods and the current count representation

Inspected 2026-11-16. The current master's thesis, `../Master-Thesis/main.tex`
→ `Chapter/Third Chapter.tex`, §3.2.1 (`sec:method-extraction`), defines the
implementation representation: overlapping DNA 8-mer **occurrence counts** and
document term **occurrence counts**, stored as nonnegative integers. A repeated
feature contributes repeatedly. Feature selection and model optimization are
separate stages.

## Evidence and identity

The local extracted folder is `107-林幸辰/`, wholly ignored by Git. It contains
19,866 files, 7,529,422,472 bytes, including original source and derived data;
there is **no PDF, DOC/DOCX, FASTA, or raw document corpus in this extraction**.
It is not safe to infer the author from the folder name. The
[NCKU thesis record](https://thesis.lib.ncku.edu.tw/thesis/detail/ca966c2629f96e2ac727fedc7641cc9d/)
identifies **蔡幸辰 (Tsai, Hsing-Chen), 2019, academic year 107**, advised by
翁慈宗 (Wong, Tzu-Tsung), and the title
*具廣義狄氏先驗分配之多項式簡易貝氏分類器於高維度不平衡資料集之研究*.
The separate [original thesis PDF](https://drive.google.com/file/d/1a7DUzvxAkILtds3-VvQh8Jk4P1rccGol/view)
(`蔡幸辰_論文.pdf`, DOI `10.6844/NCKU201900958`) was read through the connected
Drive text tool in this session: §3.1, printed pp.24–29, and §4.1, pp.49–53.
It is not bundled with this repository. No research files were uploaded and
no historical executables or installers were run.

The Chen/Lin rows below distinguish the repository's prior source inspection
record (`docs/SENIOR_METHODS.md`) from the newly inspected local Tsai archive.
Their original Java/C++ sources were not independently re-read from this local
archive and are not asserted to be present here.

## Comparison

| Aspect | 陳泓宇, 2012 | 林修弘, 2016 | 蔡幸辰, 2019 / academic year 107 |
|---|---|---|---|
| Domain | Bacterial/fungal taxonomy | DNA/RNA taxonomy | Bacteria, Fungi, 20Newsgroups in the three locally available multiclass sources; thesis also discusses other data |
| Original format | FASTA with taxonomic hierarchy | DNA/RNA and genetic-code/frame inputs; prior record names `RDP1.txt`, `RDPS.txt` | Thesis: sequence identifier, sequence and NCBI hierarchy; documents as text. Local archive: **already extracted counts**, numeric classes and feature IDs |
| Preprocessing | Direct nucleic-acid extraction | Genetic-code translation, reading-frame choices | Thesis §3.1: document tokenization, lowercase, stopwords, stemming; DNA windows. Local C#/VB source starts after feature extraction |
| Feature extraction | Overlapping **DNA 8-mer**, stride 1 | Overlapping **amino-acid 3-mer** after translation; DNA 8-mer used for comparison | Thesis: overlapping DNA 8-mer and document terms; archive gene columns span 65,536 source IDs, document declared dictionary 83,635 IDs |
| Feature values | Occurrences enumerated, but entropy selection uses **sequence-level presence** per class | Occurrence and per-sequence normalized statistics | Preserved instance files contain positive integer occurrence counts; repetitions retained, no TF-IDF, no binary conversion in the new reader |
| Labels | Hierarchical, node-specific family/genus targets | Taxonomic classes according to transformation experiments | Local originals: 741/926 numeric taxon classes, 20 document classes. Historical preprocessing changes selected class to 2 and all others to 1 |
| Feature selection/grouping | Entropy ranking and node-specific feature sets | Translation/frame comparisons and combinations | Class-specific frequency/variance-based grouping to reduce generalized-Dirichlet parameter-estimation cost |
| Classifier | Hierarchical Naive Bayes | NB variants on translated representations | Generalized-Dirichlet MNB, distinct majority/minority parameter procedures, optional removal of class prior; **not AFS-RGMNB or PSO** |
| Source evidence | Prior inspection: `entropy_sort_genus.java`, `num_choosed_genus.java`, `classification_genus.java` | Prior inspection: `main.cpp`, `Source.cpp` and thesis Ch.3 | Exact local source paths and line references below; separately read original thesis |
| Applicability | DNA window convention useful; presence statistics must not replace MNB counts | Optional alternative requiring explicit experiment design; never confuse AA 3-mer with DNA 8-mer | Preserved counts support multiclass loading and representation validation; original binary/grouping/parameter search is a separate method |

No senior method has been experimentally reproduced end to end by this change.

## Newly verified source paths and behavior

All paths in this section are relative to the ignored `107-林幸辰/` archive.

1. `Program/00_前置處理/GenePreprocess/GenePreprocess/Form1.cs`:
   lines 37–75 contain configured dataset sizes; lines 189–226 remap one class
   versus the rest, assign `Random.Next(5)+1` historical folds and accumulate
   occurrence counts; lines 235–288 read numeric class/instance files and add
   one to zero-based feature IDs for the legacy interchange format.
   There is no raw-sequence 8-mer extraction in this C# program.
2. `Program/00_前置處理/GenePreprocess/GenePreprocess/bin/Debug/bac_instance/`
   and `fun_instance/` contain the **preserved multiclass** sources used here.
   Each `<class_code>/<instance_number>.txt` has rows `feature_id count`.
   The feature IDs are zero-based, between 0 and 65,535. Counts are positive
   integers. The corresponding `ClassName_bac.txt`/`ClassName_fun.txt` uses CP950
   and maps numeric codes to taxon names. There is no accession ID, original
   sequence, parent lineage or explicit taxonomic-rank field.
3. `Program/00_前置處理/text/20Newsgroup/20newsOutput.txt` is the full multiclass
   document count source. Its ASCII rows are
   `class_code,historical_fold,one_based_feature_id,count,...`.
   `20newsTotalWord.txt` lists 83,635 token IDs. Its non-ASCII bytes are not
   needed for numeric conversion; interpreting its full text encoding requires
   a separate explicit decision. Row numbers provide archive identifiers, not
   original Message-IDs.
4. `Program/00_前置處理/text/20Newsgroup/Form1.frm` (CP950 VB6) and
   `說明檔.txt` document conversion of an already preprocessed 20-class corpus
   to minority-versus-rest and 5 folds. The input retains historical folds
   1–10. The note attributes earlier text preprocessing to 黃于珊. Neither
   file contains the tokenizer, full stopword list or stemming implementation.
5. `Program/00_前置處理/wekaProcess/wekaProcess/Form1.cs`, lines 128–151,
   writes `neg`/`pos` and repeats each numeric token exactly its count for Weka.
   It does not recover raw words.
6. `Program/01_特徵分組/FeatureGrouping/Form1.cs`, lines 120–174, computes
   per-class frequency average (FA), variance average (VA) and normalized
   variance (NV); lines 177–226 sort by FA and find adjacent NV gaps;
   lines 230–300 group features and aggregate counts. The UI range is 5–100
   groups in steps of 5. These statistics are computed from all input rows in
   that program, rather than the current thesis's `Dfit` scope.
7. `Program/02_GDRCMNB/Parameter_matrix/Form1.cs` declares 2 classes and 5
   folds (lines 20–21), uses 20 search intervals (line 131), estimates
   majority-class parameters using min/max/average/MSE alternatives (lines
   222–281), searches minority-class parameters (lines 296–324), and updates
   best parameters using F-measure aggregated from historical fold predictions
   (lines 464–501). This is evidence of the source's behavior, not a verified
   reproduction or a leakage-safe substitute for current Chapter 3.

The numerous `Dataset/Gene/{bac,fun}/...` and `Dataset/Document/20News/...`
folders contain binary conversions, class-level word statistics, grouping,
covariance and classification outputs. They are **not** concatenated into new
datasets, since that would duplicate observations and mix experiments.

## Original document pipeline versus current implementation

Tsai §3.1 describes English words split at spaces/punctuation, preserving forms
such as `real-time`, lowercase normalization, a stopword stage and Porter
(1980) stemming. A stopword-table source is given as `ranks.nl/stopwords`;
the thesis table is not a complete, versioned list. §4.1 discusses removal of
document-related terms such as `article`, `subject`, `contact`, `information`.
This does **not** establish that every RFC-style header was removed. The saved
dictionary has lowercase stem-like entries (e.g. `abandon`, `abil`, `accur`),
consistent with the thesis description, but the saved counts alone cannot
prove the exact tokenizer, stoplist, stemmer version or handling of malformed
text. Raw-input re-extraction and historical reproduction therefore remain
unverified. Current Chapter 3 count utilities must make preprocessing settings
explicit and fit their vocabulary on `Dfit` only.

## Taxonomy and dataset eligibility

The gene names are often genus-like (e.g. Photobacterium, Vibrio, Sparassis),
but names alone do not prove a consistent Genus target. Tsai §3.1 describes an
original hierarchy; the hierarchy is absent from the preserved per-instance
files. The illustrative figure also does not establish a uniform rank schema
for these reduced files. **Family labels cannot be reconstructed from these
files, and the new reader records biological rank as unverified.** Do not
join taxonomy to online records by names without a reviewed mapping.

All three full sources meet 500 samples and 5,000 observed features. Bacteria
and Fungi retain hundreds of classes of size 2; a split containing every class
in `Dfit`, `Dsearch` and `Dtest` is impossible for those classes. The validation
does not silently remove/merge classes or equalize the natural distribution.
A reviewed rare-class/grouping policy and raw biological inputs are required
before thesis performance experiments. Detailed measurements appear in
[feature_extraction_validation.md](../experiments/feature_extraction_validation.md).

The historical 20News class-19 folder/title says 755, while the actual full
multiclass source contains **775**. The loader preserves and reports the file
contents. It does not alter data to match a published table.
