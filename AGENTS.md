# AFS-RGMNB research and development

## Objective and current priority

Implement and evaluate the complete NCKU master's thesis method for high-dimensional
**multiclass** classification. Feature extraction is one layer. The November 16
report in `reports/2026-11-16/` must remain usable during interruptions. Read its
recovery checkpoint, `ROADMAP.md` and Git status before continuing unfinished work.
Preserve existing user changes and the current branch; do not push or merge without
the user's explicit approval. Local reviewed commits are authorized.

## Read-only scientific references

- Sibling thesis: `../Master-Thesis/main.tex`. Resolve the actual inclusion graph;
  active chapters are `Chapter/{First,Second,Third} Chapter.tex` and
  `Chapter/chapter3-notation-table.tex`. Read that project's instructions before
  consulting it. Never edit, commit to, or copy the thesis project.
- Authority: active Chapter 3, supporting active Chapters 1/2/4 if present, senior
  thesis, senior source programs, then implementation docs. Current assembly has
  no Chapter 4. Historical notes are not active formulas.
- Read-only local references now live in ignored `reference/`: `107-林幸辰/`
  (author is 蔡幸辰; folder spelling is historical), `100-陳泓宇/`,
  `104-林修弘/`, and the Week 16 dataset-inspection PDF. The 107 folder was moved
  from the repository root; use `--archive '.\reference\107-林幸辰'`.
  Original programs/data are read-only; never execute their binaries/installers.
- Traceability: `docs/methodology/chapter3_implementation_mapping.md`,
  `specification_status.md`, `open_questions.md`. Run
  `python tools/check_thesis_spec.py` before changing mathematical code. A changed
  source requires review, not automatic rewriting of the thesis or fingerprint.

## Dataset and reporting rules

- Preserve original multiclass labels and natural sample counts. No automatic
  class balancing, binary one-versus-rest conversion or imbalance-specific method.
- `--sampling-mode all` retains records; class filters and duplicate removal are
  explicit choices. Balanced mode exists for controlled legacy subset tests only.
- Original counts are not raw documents/sequences. The three priority local
  107 datasets are precomputed counts. Newly added 100/104 folders contain
  nucleotide inputs and archived taxonomies; read their derived inventories
  under `docs/references/`. Do not claim current taxonomy verification, a bridge
  to 107 matrix rows/feature IDs, or end-to-end reproduction from aggregate matches.
- Report actual observed features separately from declared columns and 4**8.
- Vocabulary and supervised feature scoring use Dfit only; candidate/PSO fitness
  uses Dsearch; final test is evaluation only. Split feasibility must fail clearly
  rather than silently dropping small classes. Identical count vectors do not
  prove raw identity or independent observations.
- Use ignored `data/raw/` for source inputs before extraction and
  `data/extracted/` for preserved/new feature counts; `data/readme.md` defines
  provenance and split rules. Only its README and two directory markers are
  tracked. Keep model/results artifacts in ignored `outputs/` or external
  storage. Verify `git check-ignore -v` and review staged file names before commits.
  Commit only code, synthetic fixtures, derived statistics and documentation.
- Do not upload research materials or send the email draft automatically.

## Engineering and validation

Python >=3.10; NumPy/SciPy sparse CPU computation; reproducible recorded seeds.
Use the `src/afs_rgmnb/` package and `python -m afs_rgmnb`; preserve the
`feature_pipeline.py` compatibility interface. Do not create plausible-looking
scientific placeholders. Mathematical code requires small numerical tests tied
to the thesis equations. Distinguish synthetic tests, full-source audits and
completed performance experiments in all reports.

```powershell
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python tools/check_thesis_spec.py
git diff --check
```

Use exact dependency versions from `requirements.txt` for the Python 3.10 CI
environment; record actual local versions separately. A successful source audit
is not a classification result. Re-render and visually inspect any changed PDF.
