"""Run the development pre-PSO milestone on the fingerprinted 20News count source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afs_rgmnb.data.senior_archive import discover_archive_sources, read_document_counts
from afs_rgmnb.experiments.pre_pso import PrePSOConfig, run_pre_pso, save_pre_pso


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/experiments/20news_pre_pso_smoke.json')
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    if output == archive or archive in output.parents:
        parser.error('Output must be outside the read-only reference archive')
    repo = Path(__file__).resolve().parents[1]
    thesis = repo.parent / 'Master-Thesis'
    if output == thesis or thesis in output.parents:
        parser.error('The thesis repository is read-only')
    if output.is_relative_to(repo):
        ignored = subprocess.run(['git', 'check-ignore', '-q', str(output / 'run.json')], cwd=repo)
        if ignored.returncode != 0:
            parser.error('Research output inside this repository must be in a Git-ignored directory (e.g. outputs/)')
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    config = PrePSOConfig(**spec['parameters'])
    source = discover_archive_sources(archive)['20news']
    start = time.perf_counter()
    dataset = read_document_counts(source, n_features=spec['declared_feature_dimensions'])
    source_seconds = time.perf_counter() - start
    if dataset.source_fingerprint != spec['source_fingerprint_sha256']:
        parser.error('Source fingerprint mismatch; no modeling performed. Review dataset provenance first.')
    result = run_pre_pso(dataset.X, dataset.y, config=config, sample_ids=dataset.identifiers,
                         source_feature_ids=list(range(1, dataset.X.shape[1] + 1)))
    report = result[-1]
    report['dataset_identifier'] = spec['dataset_identifier']
    report['source_fingerprint_sha256'] = dataset.source_fingerprint
    report['source_relative_path'] = source.relative_to(archive).as_posix()
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report['timings']['source_loading_seconds'] = source_seconds
    save_pre_pso(output, result, sample_ids=dataset.identifiers, sample_labels=dataset.y)
    print(json.dumps({'output': str(output), 'fit_observed_features': report['fit_observed_features'],
                      'candidate_accuracies': [[c['accuracy'] for c in g['candidates']] for g in report['groups']],
                      'selected_indices': [g['selected_index'] for g in report['groups']],
                      'test_evaluated': report['test_evaluated']}))


if __name__ == '__main__':
    main()
