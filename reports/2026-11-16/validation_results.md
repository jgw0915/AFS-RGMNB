# Validation results — initial checkpoint

## Completed

```powershell
git status --short --branch
git check-ignore -v -- '107-林幸辰/Program' '107-林幸辰/Dataset'
python -m unittest discover -s tests -v
```

Before refactoring, **11/11 existing tests passed** in 3.594 seconds (Python 3.12.7). These are synthetic format/counting/CLI tests, not full real-data extraction or classification results. Archive ignore checks passed. Both thesis and archive are readable.

## Pending at this checkpoint

Full multiclass count ingestion for 20Newsgroup, Bacteria and Fungi; measured dimensions, valid counts, duplicates, sparsity, runtime and memory; post-refactor regression tests; explicit all-sample mode; configurable document preprocessing.

No completed cross-validation performance experiment, Family/Genus task validation, or raw-sequence re-extraction is claimed. Actual commands and output paths will be appended as they finish. Heavy outputs are reserved for ignored `outputs/2026-11-16/`.
