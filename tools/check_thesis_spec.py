"""Read-only comparison with the thesis sources reviewed by this repository.

No source text is copied and neither the thesis nor the baseline is modified.
Exit 0: reviewed files match; 1: source drift/missing files; 2: invalid invocation.
The thesis Git revision is reported separately: an unrelated thesis commit does
not invalidate unchanged reviewed source bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "docs/methodology/thesis_source_fingerprints.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def compare(thesis_root: Path, manifest: dict) -> dict:
    root = thesis_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Thesis root must be a directory")
    checks = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Manifest path escapes thesis root: {relative}")
        actual = sha256_file(path) if path.is_file() else None
        checks.append({
            "path": relative.as_posix(),
            "status": "unchanged" if actual == entry["sha256"] else (
                "missing" if actual is None else "changed"
            ),
            "expected_sha256": entry["sha256"],
            "actual_sha256": actual,
            "review_modules": entry.get("review_modules", []),
        })
    current_revision = git_value(root, "rev-parse", "HEAD")
    return {
        "thesis_root": str(root),
        "review_date": manifest["review_date"],
        "reviewed_revision": manifest.get("thesis_git_revision"),
        "current_revision": current_revision,
        "revision_changed": current_revision != manifest.get("thesis_git_revision"),
        "git_status": git_value(root, "status", "--short"),
        "sources_match": all(item["status"] == "unchanged" for item in checks),
        "files": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        root = args.thesis_root or (REPO_ROOT / manifest["thesis_relative_path"])
        report = compare(root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Thesis specification check failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["sources_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
