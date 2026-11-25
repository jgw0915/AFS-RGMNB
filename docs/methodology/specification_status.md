# Reviewed thesis specification

Reviewed on 2026-11-26 (Asia/Taipei). The read-only sibling is `../Master-Thesis`
at commit `059f00b643fd4e94ad0c8eb49b05ac2d9dd6be44`. Its working tree was clean.

```text
C:\Users\kiwi\NCKU_Master\Thesis\Random Multinomial Model on High Dimension Data\Master-Thesis
```

| Source relative to thesis root | Scope checked |
|---|---|
| `main.tex` | Includes the front matter, Chapters 1, 2 and 3, and the bibliography. The assembly ends before Chapter 4. |
| `Chapter/First Chapter.tex`, `Chapter/Second Chapter.tex` | Motivation and supporting literature unchanged since the prior review |
| `Chapter/Third Chapter.tex` | §§3.1–3.5, 20 numbered equations, final subspaces from the stopping round |
| `Chapter/chapter3-notation-table.tex` | Active notation in §3.1 includes T, S_b, historical bests and a single H |
| `figures/chapter3-method.drawio`, `figures/chapter3-method.pdf` | Figure 3.1, inspected as a rendered figure and as graph edges |
| `build/main.pdf` | November 25 compiled thesis. Printed pages 15–27 contain Chapter 3. Figure 3.1 is on printed page 17. Its stage order, section titles and equation numbering agree with the active source. |

The separately named `main(20261125-163149).pdf` could not be located in the
workspace or local Windows user profiles during review. Equivalence to that
particular export remains unverified. The available compiled PDF has SHA-256
`2964127ba4afec1d34a9d62fac4bc0f303f2cc9d044ef2c678b02d19478e0e72`.

The previous scientific baseline was `669245c9be4073e682ae76d6b0bdf80cd724d89b`.
The checker identified four changed files before implementation. The source and
figure review, affected modules and remaining questions are in the
[migration](../../reports/2026-11-25/chapter3_workflow_migration.md) and
[equation mapping](chapter3_implementation_mapping.md). Q01, Q07 and Q08
remain explicit in [open questions](open_questions.md).

Run `python tools/check_thesis_spec.py` from the implementation root. The
manifest records hashes for all prior review inputs plus the compiled PDF.
On drift, review the active inclusion graph and rerun the affected mathematical
and orchestration tests before approving new hashes.
