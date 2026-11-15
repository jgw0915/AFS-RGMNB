from __future__ import annotations

import gzip
from pathlib import Path

def open_text(path: Path, encoding: str = "utf-8", errors: str = "strict"):
    if not path.is_file():
        raise FileNotFoundError(path)
    return gzip.open(path, "rt", encoding=encoding, errors=errors) if path.suffix == ".gz" \
        else path.open("r", encoding=encoding, errors=errors)
