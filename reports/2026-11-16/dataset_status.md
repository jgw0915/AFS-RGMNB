# Dataset status — 2026-11-16 verified multiclass count audit

**Status: three original multiclass precomputed count matrices were successfully read and measured locally.** These are not newly extracted from raw files.

| Dataset | Source columns / possible 8-mers | **Measured samples** | **Measured classes** | **Observed full-matrix features** | **Nonzero entries** | **Audit time (s)** |
|---|---:|---:|---:|---:|---:|---:|
| 20Newsgroup | 83,635 | 18,828 | 20 | **83,415** | 1,505,793 | 1.32 |
| Bacteria | 65,536 | 6,386 | 741 | **64,934** | 8,792,221 | 46.85 |
| Fungi | 65,536 | 7,730 | 926 | **61,563** | 7,138,523 | 12.45 |
| MDR88 (optional, not run) | 12,511 (senior thesis) | 14,687 (senior thesis) | 4 (senior thesis) | Not measured | Not measured | — |

The bold measured values are the **user-provided PowerShell results** after running `scripts/validate_senior_datasets.py --dataset all` on the locally extracted archive on 2026-11-16. The JSON output files remain local and ignored by Git. MDR88 figures are from 蔡幸辰 (2019), §4.1, **not Python measurements**.

All three audited datasets meet the preliminary requirements **n >= 500** and **observed full-matrix d >= 5,000**, with original multiclass labels preserved and no binary one-vs-rest conversion. Their observed-feature counts mean feature columns that occur **at least once across the full matrix**, not feature counts in any given training fold (`Dfit`). For DNA, `4^8 = 65,536` is the theoretical feature universe.

**Sources present in ignored local archive:** `Program/00_前置處理/text/20Newsgroup/20newsOutput.txt` (20News count matrix), `Program/00_前置處理/GenePreprocess/GenePreprocess/bin/Debug/bac_instance/` and `fun_instance/` (per-sample DNA count matrices).

**Unresolved:** Raw text and DNA sequences were not recovered from this archive, so historic preprocessing was not reproduced. Bacteria/Fungi numeric labels and names do not prove Family/Genus ranks; very small classes prevent an unrestricted fivefold class-preserving `Dfit/Dsearch/Dtest` protocol. No accuracy, JMH, random MNB or PSO result is claimed. See [validation_results.md](validation_results.md).
