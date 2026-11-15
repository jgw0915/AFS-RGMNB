#!/usr/bin/env python3
"""Non-destructive local inventory of an EXTRACTED dataset bundle.

Does not parse an opaque ZIP or upload anything. It reports schemas and sizes;
never writes raw contents to GitHub. Run after unzipping on the lab server.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter

DNA_EXT = {'.fa', '.fasta', '.fna', '.fastq', '.fq'}
TEXT_EXT = {'.csv', '.tsv', '.jsonl', '.txt'}
SOURCE_EXT = {'.py', '.java', '.cpp', '.c', '.h', '.m', '.r'}


def open_maybe_gzip(path):
    return gzip.open(path, 'rt', encoding='utf-8', errors='replace') if path.suffix == '.gz' else path.open('r', encoding='utf-8', errors='replace')


def inspect(path, root, checksum=False):
    rel = str(path.relative_to(root))
    suffix = Path(path.stem).suffix.lower() if path.suffix == '.gz' else path.suffix.lower()
    result = {'path': rel, 'bytes': path.stat().st_size, 'extension': suffix}
    if suffix in DNA_EXT | TEXT_EXT:
        try:
            with open_maybe_gzip(path) as f:
                first = next((x.rstrip('\r\n') for x in f if x.strip()), '')
            if first.startswith('>'):
                result['detected'] = 'FASTA-like'
                result['first_header'] = first[:250]
            elif suffix in {'.csv', '.tsv'}:
                delimiter = '\t' if suffix == '.tsv' else ','
                result['detected'] = 'delimited-table'
                result['columns'] = next(csv.reader([first], delimiter=delimiter), [])[:40]
            elif suffix == '.jsonl':
                result['detected'] = 'JSON Lines'
                result['keys'] = list(json.loads(first))[:40]
            else:
                result['detected'] = 'text/unknown'
        except (OSError, UnicodeError, ValueError, StopIteration) as e:
            result['read_warning'] = str(e)[:200]
    if suffix in SOURCE_EXT:
        result['detected'] = 'program source'
    if checksum:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                h.update(chunk)
        result['sha256'] = h.hexdigest()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, default=Path('preflight_report.json'))
    p.add_argument('--sha256', action='store_true', help='Compute full hashes (slow for large files)')
    args = p.parse_args()
    if not args.directory.is_dir():
        p.error('Provide the EXTRACTED directory, not a ZIP file')
    items = [inspect(f, args.directory, args.sha256) for f in sorted(args.directory.rglob('*')) if f.is_file()]
    summary = {'root': str(args.directory.resolve()), 'files': len(items),
               'total_bytes': sum(x['bytes'] for x in items),
               'extensions': dict(Counter(x['extension'] for x in items)),
               'items': items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Scanned {summary['files']} files ({summary['total_bytes']} bytes). Report: {args.output}")

if __name__ == '__main__':
    main()