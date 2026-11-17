"""Small fingerprinted 20News random-MNB/PSO development run (test unscored)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from afs_rgmnb.data.senior_archive import discover_archive_sources, read_document_counts
from afs_rgmnb.experiments.pre_pso import PrePSOConfig
from afs_rgmnb.experiments.pso_search import run_pso_pipeline, save_pso_pipeline
from afs_rgmnb.methods.pso import PSOConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/experiments/20news_pso_smoke.json')
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    thesis = repo.parent / 'Master-Thesis'
    if output == archive or archive in output.parents or output == thesis or thesis in output.parents:
        parser.error('Reference archive and thesis are read-only; select a different output directory')
    if output.is_relative_to(repo):
        if subprocess.run(['git', 'check-ignore', '-q', str(output / 'run.json')], cwd=repo).returncode:
            parser.error('Research output must be in an ignored directory such as outputs/')
    raw_config = args.config.read_bytes()
    spec = json.loads(raw_config.decode('utf-8'))
    pre_config = PrePSOConfig(**spec['pre_pso_parameters'])
    pso_config = PSOConfig(**spec['pso_parameters'])
    source = discover_archive_sources(archive)['20news']
    start = time.perf_counter()
    dataset = read_document_counts(source, n_features=spec['declared_feature_dimensions'])
    load_seconds = time.perf_counter() - start
    if dataset.source_fingerprint != spec['source_fingerprint_sha256']:
        parser.error('Source fingerprint mismatch; modeling not performed')
    result = run_pso_pipeline(dataset.X, dataset.y, pre_config=pre_config,
                              pso_config=pso_config, pso_seed=spec['pso_seed'],
                              sample_ids=dataset.identifiers,
                              source_feature_ids=list(range(1, dataset.X.shape[1] + 1)))
    report = result[-1]
    report.update(dataset_identifier=spec['dataset_identifier'],
                  source_fingerprint_sha256=dataset.source_fingerprint,
                  source_relative_path=source.relative_to(archive).as_posix(),
                  config_sha256=hashlib.sha256(raw_config).hexdigest())
    report['timings']['source_loading_seconds'] = load_seconds
    save_pso_pipeline(output, result, sample_ids=dataset.identifiers, sample_labels=dataset.y)
    print(json.dumps({'output': str(output), 'test_evaluated': report['test_evaluated'],
                      'groups': [{k: group[k] for k in ['group_index', 'initial_best_search_accuracy',
                                  'best_search_accuracy', 'stop_reason', 'fitness_evaluation_count',
                                  'particle_reset_count']} for group in report['pso']['groups']]}))


if __name__ == '__main__':
    main()
