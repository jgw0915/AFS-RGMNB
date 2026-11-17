from __future__ import annotations

import gzip
from pathlib import Path


def ensure_output_outside_references(
    output: str | Path, *, repo_root: str | Path, archive_root: str | Path | None = None
) -> Path:
    """Resolve an output path and reject read-only source/reference destinations.

    This check performs no filesystem writes. Resolving both sides protects
    against ``..`` paths and existing filesystem aliases. Git-ignore validation
    is a separate CLI responsibility: an ignored reference is still read-only.
    """
    destination = Path(output).resolve()
    repo = Path(repo_root).resolve()
    protected = [repo / "reference", repo / "data" / "raw", repo.parent / "Master-Thesis"]
    if archive_root is not None:
        protected.append(Path(archive_root).resolve())
    for source in protected:
        source = source.resolve()
        if destination == source or destination.is_relative_to(source):
            raise ValueError(f"Output must be outside read-only source/reference directory: {source}")
    return destination


def open_text(path: Path, encoding: str = "utf-8", errors: str = "strict"):
    if not path.is_file():
        raise FileNotFoundError(path)
    return gzip.open(path, "rt", encoding=encoding, errors=errors) if path.suffix == ".gz" \
        else path.open("r", encoding=encoding, errors=errors)
