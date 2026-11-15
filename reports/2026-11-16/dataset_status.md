# Dataset status — initial verified checkpoint

Published figures below are supplied from the senior thesis §4.1/Table 4.1 and are **not Python validation results**. Full measured results will replace pending fields after validation.

| Dataset | Senior samples | Senior features | Senior classes | Local availability | Measured eligibility |
|---|---:|---:|---:|---|---|
| 20Newsgroup | 18,828 | 83,635 | 20 | Multiclass precomputed sparse counts, `Program/00_前置處理/text/20Newsgroup/20newsOutput.txt` | Validation pending |
| Bacteria | 6,386 | 65,536 | 741 | Per-sample sparse counts in `Program/00_前置處理/GenePreprocess/GenePreprocess/bin/Debug/bac_instance/` | Validation pending; biological rank unverified |
| Fungi | 7,730 | 65,536 | 926 | Per-sample sparse counts in corresponding `fun_instance/` | Validation pending; biological rank unverified |
| MDR88 | 14,687 | 12,511 | 4 | Not yet verified locally | Optional; not selected for the main validation |

All paths are relative to ignored `107-林幸辰/`. Folder spelling is not proof of thesis authorship. The linked Drive file metadata identifies `蔡幸辰_論文.pdf`.

Use original multiclass labels. The senior's binary experimental variants are excluded from the main workflow. Theoretical 65,536 DNA 8-mers must not be reported as observed training-fold features. Class-frequency constraints must be checked before fivefold stratification; no classes are silently removed.
