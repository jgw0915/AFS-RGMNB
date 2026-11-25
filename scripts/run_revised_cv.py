"""Evaluate the revised Chapter 3 workflow on fingerprinted preserved 20News counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

from afs_rgmnb.data.senior_archive import discover_archive_sources, read_document_counts
from afs_rgmnb.experiments.runner import run_cv
from afs_rgmnb.methods.adaptive_search import AdaptiveConfig
from afs_rgmnb.methods.pso import PSOConfig
from afs_rgmnb.utils.io import ensure_output_outside_references


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo / "configs/experiments/20news_revised_dev.json")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate every planned outer fold")
    args = parser.parse_args()
    archive = args.archive.resolve()
    output = ensure_output_outside_references(args.output, repo_root=repo, archive_root=archive)
    if output.is_relative_to(repo) and not output.is_relative_to(repo / "outputs"):
        parser.error("Use outputs/ for local experiment artifacts")
    if output.is_relative_to(repo) and subprocess.run(
            ["git", "check-ignore", "-q", str(output / "run.json")], cwd=repo).returncode:
        parser.error("Experiment artifacts must be ignored")
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory for this run")
    start = perf_counter()
    raw = args.config.read_bytes()
    settings = json.loads(raw)
    adaptive, pso = AdaptiveConfig(**settings["adaptive"]), PSOConfig(**settings["pso"])
    source = discover_archive_sources(archive)["20news"]
    dataset = read_document_counts(source, n_features=settings["declared_features"])
    if dataset.source_fingerprint != settings["source_fingerprint_sha256"]:
        parser.error("Preserved count source fingerprint mismatch")
    load_seconds = perf_counter() - start
    cv = settings["cv"].copy()
    if args.all_folds:
        cv["folds"] = None
    result = run_cv(dataset.X, dataset.y, adaptive=adaptive, pso=pso, **cv,
                    sample_ids=dataset.identifiers, source_feature_ids=range(1, dataset.X.shape[1] + 1),
                    output=output)
    result.update(dataset=settings["dataset"], source_fingerprint_sha256=dataset.source_fingerprint,
                  config_sha256=hashlib.sha256(raw).hexdigest(), cv_settings=cv,
                  source_loading_seconds=load_seconds, total_runtime_seconds=perf_counter() - start)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for fold in result["folds"]:
        print(f"Fold {fold['fold']}: {fold['test_correct']}/{fold['test_samples']} correct, "
              f"{fold['candidate_evaluations']} candidates, {fold['independent_searches']} PSO searches")


if __name__ == "__main__":
    main()
