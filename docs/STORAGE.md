# Storage and redistribution policy

**Recommended for the current thesis:** Keep the original three datasets on the lab server (e.g. `/lab/datasets/tsai107/`) outside Git, and keep a **separate backed-up copy** on controlled institutional storage or a permission-appropriate Drive folder. The `AFS-RGMNB` GitHub repository is for Python code, small metadata manifests, experiment configuration, reproduction commands and test summaries—not raw archives or generated matrices.

## Why not GitHub for the data?

- GitHub warns on individual regular Git files over 50 MiB and blocks files over 100 MiB. It is inefficient to keep a changing large data collection in source-control history.
- Git LFS may handle large objects, but introduces storage/bandwidth quota and history management, without helping thesis reproducibility if a single server is the only compute location.
- The provided ZIP alone is about **1.09 GiB**; no redistribution permission has been verified for any of the three contained datasets.

## When to use Hugging Face Datasets

Use it **only if** the data sources permit redistribution, you want to release reusable research datasets, you can document provenance, labels, processing and licenses in a dataset card, and you are comfortable with access control (public/private). Current free HF private storage is documented as 100 GB, while free public storage is best-effort and intended for community-reusable data.

As the thesis will run on a single lab server, putting these datasets on HF would add work with little experimental benefit. Revisit for an open-source release or multi-machine collaboration; do not publish a senior's or third party's raw data without permission.

## Recommended reproducibility layout

```text
AFS-RGMNB/                     # GitHub: scripts, docs, hashes, parameter files
/lab/datasets/tsai107/         # raw, read-only, NOT Git
/lab/datasets/chen2012/        # optional real lab validation inputs
/lab/results/afsrgmnb/         # generated CSR matrices/logs, NOT Git
/institutional-backup/         # versioned/backed-up source dataset copies
```

Record `sha256sum` for each source file and keep a small manifest in Git with URL/DOI, license/access notes, expected counts, and file checksums. Prefer read-only datasets plus out-of-tree experiment outputs. For large cross-machine transfer, use `rsync --partial --progress` (SSH) from a trusted lab machine.

Sources:
- GitHub: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- Hugging Face: https://huggingface.co/docs/hub/storage-limits