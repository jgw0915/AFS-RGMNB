# Lab senior methods: extraction vs selection vs MNB modeling

## What is verified

| Senior | Source | Sequence or text representation | Feature *value* | Distinct research contribution |
|---|---|---|---|---|
| 陳泓宇 (2012) | Original thesis plus Java source files `entropy_sort_genus.java`, `num_choosed_genus.java`, `classification_genus.java` in the previously accessible Drive folder | Direct overlapping **DNA 8-mer** (stride 1) from Bacteria/Fungi FASTA, taxonomy Domain → Phylum → Class → Order → Family → Genus | DNA 8-mer counts are enumerated, but selection code uses *sequence-level presence* per class (a feature seen twice in one sample contributes one to its class document frequency). | Node-specific entropy scoring, feature-number selection, hierarchical Naive Bayes prediction; not the same conditional-count estimation as conventional MNB. |
| 林修弘 (2016) | Original thesis (Ch. 3), `main.cpp` and `Source.cpp`, `RDP1.txt`/`RDPS.txt` previously accessible in Drive | DNA/RNA first translated using genetic code to amino acids with multiple reading frames, then overlapping **amino-acid 3-mer** | 3-mer occurrence counts and derived per-sequence frequency/normalization statistics; 20^3=8,000 (or 21^3=9,261 including stop symbol) theoretical features. | Frame/translation comparison and combinations; nucleic-acid 8-mer is used as a benchmark, not the same main feature representation. |
| 蔡幸辰 (2019; graduation academic year 107) | Official NCKU public thesis *具廣義狄氏先驗分配之多項式簡易貝氏分類器於高維度不平衡資料集之研究* (abstract, table of contents). The linked ZIP is not downloadable through the connected service (1.17 GB > 256 MiB). | **Not verified from source programs or actual three datasets.** Thesis Ch. 3.1 is “資料前置處理”; experiment includes high-dimensional imbalanced datasets. | **Not verified.** Do not claim that the three datasets all use one particular tokenizer, FASTA header layout, or k-mer normalization. | Generalized Dirichlet priors for high-dimensional class-imbalanced MNB, class-dependent parameter setting and feature grouping to accelerate parameter fitting. |

NCKU thesis record (蔡幸辰): https://thesis.lib.ncku.edu.tw/thesis/detail/ca966c2629f96e2ac727fedc7641cc9d/

**Source caution:** Drive file's *filename* is `107-林幸辰.zip`, but the NCKU thesis record spells the author `蔡幸辰`. The ZIP contents remain unverified; do not assume the filename is accurate.

## Baseline selected for AFS-RGMNB

* DNA taxonomy task: raw nucleotide 8-mer **integer counts**, not translated amino-acid 3-mer and not Chen's binary document-frequency statistic.
* Document task: word occurrence integer counts from explicit text labels; no TF-IDF or TF scaling, to align with standard Multinomial NB.
* Family and Genus are separate prediction targets. All rank paths are validated with an explicit schema; rank ambiguities are not silently collapsed.
* Vocabulary/feature dictionary is learned from `Dfit` only; `Dsearch` and `Dtest` remain held out for subsequent random subspace/MNB/PSO experiments.
* Feature selection is **outside this module**: the thesis's JMH scoring, adaptive weights, random subspaces and PSO should be implemented as separate stages using these CSR matrices.
* The original senior implementations are research references. This code is **not** an exact reproduction of any senior's full pipeline.