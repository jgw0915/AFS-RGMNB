# Validation results — 2026-11-16

## Actual local run

```powershell
python scripts/validate_senior_datasets.py `
  --archive ".\107-林幸辰" `
  --output ".\outputs\2026-11-16" `
  --dataset all
```

The user ran the command successfully and provided the following PowerShell output:

| Dataset | Samples | Classes | ObservedFeatures | NonzeroValues | RuntimeSeconds |
|---|---:|---:|---:|---:|---:|
| 20news | 18,828 | 20 | **83,415** | 1,505,793 | 1.32 |
| bacteria | 6,386 | 741 | **64,934** | 8,792,221 | 46.85 |
| fungi | 7,730 | 926 | **61,563** | 7,138,523 | 12.45 |

Local outputs (not committed, because `outputs/` is ignored):
```text
outputs/2026-11-16/
  20news_validation.json
  bacteria_validation.json
  fungi_validation.json
  senior_validation_summary.json
```

These are **source count-matrix loading and statistics**, not raw-text tokenizer execution, raw-DNA 8-mer re-extraction, or classification benchmark results. The full JSON contents (e.g., class distribution, duplicate and memory statistics) were not supplied for independent review.

## Code testing

- Earlier local baseline (before modular refactor): **11/11 tests passed** on Python 3.12.7.
- Latest pushed branch commit `e545f086` passed the GitHub Actions **Python 3.10 extraction tests** workflow on 2026-11-16. CI tests are separate from the local three-dataset count audit.
- No final cross-validation, classification accuracy, ensemble, JMH, or PSO result is claimed.

## Known validation boundaries

1. Source data are already-calculated occurrence counts. Original source texts, nucleotide sequences, exact historic tokenizer/stoplist/stemmer and raw sample IDs are unavailable in this extracted archive.
2. DNA class codes have taxon names but no complete kingdom-to-species lineage or confirmed Family/Genus ranks.
3. Some classes have too few records for a fivefold three-part stratified split; explicit experimental screening rules must be defined before Chapter 4 performance comparison.
4. Full-matrix observed-feature counts are descriptive. Training-fold vocabulary must be defined using `Dfit` only; fold-specific `d` remains to be measured.

See [dataset_status.md](dataset_status.md) and [professor_summary_zh-TW.md](professor_summary_zh-TW.md).
