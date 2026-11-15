# Recovery checkpoint — 2026-11-16 (Asia/Taipei)

The November 16 professor report takes priority over PSO and further restructuring.

- Branch: `refactor/full-thesis-framework`; no push or merge performed.
- Recovered edits: whole-folder archive protection in `.gitignore`, compatibility wrapper `feature_pipeline.py`, extracted package modules under `src/afs_rgmnb/`, and Chapter 3 mapping/open questions under `docs/methodology/`.
- Original baseline: **11 unittest tests passed** on Python 3.12.7 before refactoring. Post-recovery checks will be recorded in `validation_results.md`.
- No PSO, adaptive ensemble, or completed classification experiments are claimed at this checkpoint.
- Authoritative thesis: `../Master-Thesis/main.tex`, including `Chapter/First Chapter.tex`, `Chapter/Second Chapter.tex`, `Chapter/Third Chapter.tex` and `Chapter/chapter3-notation-table.tex`. No Chapter 4 is included. Thesis is read-only.
- Reference archive: `107-林幸辰/`; `git check-ignore -v` confirms the entire folder is excluded. No archive files were tracked at initial inspection. Originals remain read-only.
- Initial archive inspection located multiclass precomputed counts for 20Newsgroup, Bacteria and Fungi. Raw document/sequence reconstruction and biological rank are not yet verified.
- Updated scope: multiclass classification; preserve original class counts. No binary minority-versus-rest transformation, silent balancing or imbalance-specific learning.
- Saved initial usable report before full extraction runs. Subsequent report updates must distinguish senior-reported figures from measured results.

The interrupted refactor is preserved rather than restarted. Heavy outputs belong in ignored `outputs/2026-11-16/`.
