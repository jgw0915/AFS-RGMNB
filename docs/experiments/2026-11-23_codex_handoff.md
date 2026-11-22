# Local Codex handoff — 11/23 noon advisor delivery

**Goal:** Execute the committed 20 Newsgroups **raw-text → integer word occurrence count** demonstration on 60 actual public examples, verify every exported value, and produce a concise Traditional Chinese professor-facing run summary. This is not a request to rewrite Chapter 3, run PSO, or create invented data.

## Sources and safeguards

- Work only in `AFS-RGMNB`, branch `refactor/full-thesis-framework`; preserve uncommitted changes and current work.
- Read `AGENTS.md`, `docs/experiments/2026-11-23_20news_demo.md`, `scripts/export_20news_60_demo.py` and `reports/2026-11-23/new_dataset_candidates.csv` before running.
- Program must use `sklearn.datasets.fetch_20newsgroups` **non-vectorized** as raw text and `afs_rgmnb.features.document.document_vectorizer` for word counts.
- Never substitute `fetch_20newsgroups_vectorized` for raw documents, never use the senior 18,828-row matrix as raw text, and never invent experimental metrics.
- Private/raw texts and row-level CSV outputs stay under ignored `outputs/` and must **not** be pushed to public GitHub.
- Formal vocabulary/fit-search-test protocol is separate; all 60 examples are fitted for a representation demo only.

## Run (PowerShell from repo root)

```powershell
git status --short --branch
git pull --ff-only origin refactor/full-thesis-framework
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/export_20news_60_demo.py --selection first60 --output outputs/2026-11-23/20news_demo
python -m unittest discover -s tests -v
python tools/check_thesis_spec.py
git check-ignore -v outputs/2026-11-23/20news_demo/20news_original_60.csv
git status --short
```

If the repository is dirty, investigate carefully and do not overwrite files with pull. If download fails, diagnose network/certificate/cache access, preserve traceback and do not fabricate CSV contents. If the latest commit already has this exporter, do not reimplement it.

## Verify actual outputs

1. Inspect and count CSV rows with Python's `csv.DictReader`: precisely 60 original records, sample ID unique.
2. Confirm the extracted long-form CSV contains exclusively the same IDs, nonnegative integer counts and valid `feature_index`/term mapping.
3. Cross-check at least 5 sampled original documents against the project's analyzer and counts (not a different CountVectorizer), including a word repeated multiple times.
4. Validate actual `demo_manifest.json`: sample count, class distribution, vocabulary size, nnz, versions and source selected indices.
5. Do not assume 60 documents cover all 20 classes. Record actual number of represented classes.
6. Optionally run `--selection three_per_class` to a **different ignored output directory**; this would guarantee 3 per class if the public source is complete, and may make a better class-diversity display.
7. Do not conflate original raw text, text cleaning, count extraction, feature selection, or PSO search.

## Deliverables

- Local `outputs/2026-11-23/20news_demo/20news_original_60.csv`.
- Local `outputs/2026-11-23/20news_demo/20news_extracted_word_counts_60.csv`.
- Local `outputs/2026-11-23/20news_demo/demo_manifest.json`.
- Safe-to-commit **summary only** in `reports/2026-11-23/20news_demo_execution_summary.md` (Traditional Chinese, no raw excerpts, email addresses, record-level data, or sample-specific sensitive identifiers). Include exact command, observed class count, feature count, nnz, total term occurrences, selection scheme, extraction parameters, actual environment, test results, and caveats.
- Brief professor email draft in `reports/2026-11-23/email_draft_zh-TW.md`: one paragraph on extraction, one paragraph on Reuters R8 / WebKB / BBC candidate research, and any clear next steps. **Do not send**.
- If a figure is useful, use a tiny aggregated safe example rather than a raw document quotation.

## Completion

Run the test suite and thesis fingerprint checker; ensure Git doesn't show private data tracked. Commit only the code/documentation updates and summary report with Conventional Commits, but follow `AGENTS.md` and do not push additional changes without the user's explicit approval. Return the paths, exact executed commands, measured results, blockers, and commit IDs to the user. Never claim actual execution unless it occurred.
