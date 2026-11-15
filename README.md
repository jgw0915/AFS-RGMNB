# AFS-RGMNB — Python 3.10 feature extraction for high-dimensional MNB

Reproducible, **count-based** preprocessing for high-dimensional **DNA sequence** and **document** classification. This is the feature extraction layer for the NCKU master thesis (randomly-generated ensemble of Multinomial Naive Bayes). It does **not** implement the thesis's JMH feature scoring, random subspace search, random MNB model generation, or PSO yet.

**Research integrity and current status (2026-11-16):** Senior archive and datasets are **not bundled in this public repo**. The extracted archive was inspected locally, and its **precomputed multiclass count matrices** for 20Newsgroup, Bacteria, and Fungi were successfully audited (18,828/6,386/7,730 samples; **83,415/64,934/61,563 observed full-matrix feature columns**). This does **not** mean raw-document or DNA feature re-extraction is completed. Taxonomic ranks of the gene class labels remain unverified. See [`reports/2026-11-16/`](reports/2026-11-16/) for up-to-date results; `docs/VALIDATION.md` is an earlier dated baseline.

## Install (lab server, Python 3.10)

```bash
git clone https://github.com/jgw0915/AFS-RGMNB.git
cd AFS-RGMNB
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

`requirements.txt`: NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.4.2 (compatible with Python 3.10). Automated tests run on GitHub Actions with Python 3.10; initial local development checks ran under Python 3.13 because Python 3.10 was not installed in the execution environment.

## Dataset inventory first (especially 蔡幸辰 archive)

Extract the ZIP **locally on your lab server**, not into the Git repo. Then inspect its files to determine the actual dataset names and schemas:

```bash
mkdir -p "$HOME/thesis_data/tsai107"
unzip '/path/to/107-林幸辰.zip' -d "$HOME/thesis_data/tsai107"
python tools/preflight.py "$HOME/thesis_data/tsai107" --output "$HOME/thesis_data/tsai107_inventory.json"
```

`preflight.py` reads the directory recursively, reports file sizes, extensions, first FASTA header or table column names, and source code filenames, without copying raw datasets into the repo. `--sha256` additionally hashes entire files (slower).

## DNA: overlapping nucleotide 8-mer counts (MNB-ready)

Supports gzipped/multiline FASTA and either:
- `--dna-format silva`: `>sequence_ID Domain;Phylum;Class;Order;Family;Genus` (the older 陳泓宇 FASTA format works here too).
- `--dna-format dada2`: taxonomy is the full `>` header, e.g. `>Bacteria;Phylum;Class;Order;Family;Genus`.
- `--dna-format metadata`: FASTA identifiers paired with `id`, `family`, `genus` columns in a TSV/CSV metadata file. Use an explicit mapping if ranks are irregular.

Use **only the six-rank schema** if the header really contains Domain/Phylum/Class/Order/Family/Genus. For a separately documented seven-rank schema, select `--rank-schema seven`. Never infer biological ranks by arbitrary token position when intermediary ranks are inconsistent.

```bash
# Inspect label distribution without extracting the matrix
python feature_pipeline.py inspect --kind dna \
  --input /lab/datasets/Bacteria_2035_Nonsingleton.fasta \
  --dna-format silva --domain Bacteria --rank-schema six --rank genus \
  --max-classes 25 --samples-per-class 20 --min-class-size 20 \
  --output /lab/results/inspect_bacteria

# Extract a balanced, 500-sample, 25-genus test subset
python feature_pipeline.py extract --kind dna \
  --input /lab/datasets/Bacteria_2035_Nonsingleton.fasta \
  --dna-format silva --domain Bacteria --rank-schema six --rank genus \
  --k 8 --max-classes 25 --samples-per-class 20 --min-class-size 20 \
  --folds 5 --min-features 5000 --min-total-samples 500 --baseline \
  --output /lab/results/bacteria_genus_25x20

# For Fungi
python feature_pipeline.py extract --kind dna \
  --input /lab/datasets/Fungi_4954_Nonsingleton.fasta \
  --dna-format silva --domain Fungi --rank-schema six --rank family \
  --max-classes 20 --samples-per-class 25 --min-class-size 25 \
  --output /lab/results/fungi_family_20x25
```

A DNA 8-mer has `4^8 = 65,536` **possible** values. Actual features `d` are the 8-mers **observed in each Dfit fold**, checked against `--min-features 5000`. The output value `X[i,j]` is a nonnegative integer occurrence count, **not** a binary presence indicator, TF-IDF, or length-normalized frequency. Overlapping windows advance one base; ambiguity (`N`, `R`, etc.) resets a window. `U` is treated as `T`; reverse complements are **not** merged. This is correct for RNA gene sequences such as 16S; genetic-code translation is intentionally excluded.

## Documents: word occurrence counts

```bash
# CSV with label,text[,id] columns
python feature_pipeline.py extract --kind text \
  --input /lab/datasets/documents.csv --text-format csv \
  --label-col label --text-col text --id-col id \
  --max-classes 5 --samples-per-class 100 --min-class-size 100 \
  --min-features 5000 --min-total-samples 500 \
  --output /lab/results/documents --baseline

# TSV: add --delimiter $'\t'
# JSON Lines: --text-format jsonl --label-col topic --text-col body
# Folder layout: documents/<label>/*.txt: use --text-format dir
```

`CountVectorizer` lowercases text, strips accents, and uses Unicode word tokens of length >=2. Optional `--stopwords english`; default: **none**. It does **not** perform Chinese word segmentation; presegment Chinese documents before passing them in. The word vocabulary is **fitted exclusively on Dfit**. No TF-IDF or normalization is applied; use raw counts to match MNB.

## Outputs

```text
<output>/
    manifest.json              # exact CLI parameters and fold summaries
    class_index.tsv
    selected_samples.tsv.gz   # input snapshot: contains raw sample text/sequences!
    fold_01/ ... fold_05/
       X_fit.npz, X_search.npz, X_test.npz   # SciPy CSR sparse int32
       y_fit.npy, y_search.npy, y_test.npy   # class label IDs
       ids_fit.tsv, ids_search.tsv, ids_test.tsv
       features.tsv           # dimension -> 8-mer / word
       summary.json           # actual dimensions, counts, optional sanity-check MNB metrics
```

`Dtest` is held out; `Dsearch` is an internal part of the outer training fold. Model feature scoring and PSO must not use `Dtest`. The vocabulary is fitted only on `Dfit` as an even stricter separation. Outer split: stratified 5-fold CV; inner split: stratified Dfit/Dsearch. **Stratification preserves label proportions, not sequence independence**. Exact duplicate handling is implemented for sampled records, but near-identical biological sequences/homologous groups are not removed and may cause overly optimistic metrics. A group/similarity-aware test is still required before scientific conclusions.

Load an output fold:

```python
import numpy as np
from scipy.sparse import load_npz
fold = '/lab/results/bacteria_genus_25x20/fold_01'
X_fit = load_npz(fold + '/X_fit.npz')
y_fit = np.load(fold + '/y_fit.npy')
print(X_fit.shape, X_fit.dtype, X_fit.nnz)
```

## Smoke tests (not biological evidence)

```bash
python examples/make_synthetic_data.py --out /tmp/afsr_demo
python feature_pipeline.py extract --kind dna \
  --input /tmp/afsr_demo/synthetic_dada2.fa.gz --dna-format dada2 \
  --samples-per-class 100 --max-classes 5 --output /tmp/afsr_dna --baseline
python feature_pipeline.py extract --kind text \
  --input /tmp/afsr_demo/synthetic_documents.csv \
  --samples-per-class 100 --max-classes 5 --output /tmp/afsr_text --baseline
```

**Data safety:** The repository is public. `.gitignore` excludes datasets, `.npz`, `.npy`, ZIP archives, generated validation output. Do not commit datasets, original senior programs, or privately shared material without explicit reuse permission. Prefer `README`/manifests with source URLs and checksums; see `docs/STORAGE.md`.