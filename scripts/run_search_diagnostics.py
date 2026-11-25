"""Replay one archived 20News split, then measure Dsearch-only controlled experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from afs_rgmnb.data.senior_archive import discover_archive_sources, read_document_counts
from afs_rgmnb.experiments.diagnostics import (
    SearchData,
    compare_models,
    digest,
    feature_diagnostics,
    reference_controls,
    select_features,
)
from afs_rgmnb.methods.adaptive_search import AdaptiveConfig
from afs_rgmnb.methods.ensemble import VotingEnsemble
from afs_rgmnb.methods.pso import PSOConfig
from afs_rgmnb.utils.reproducibility import environment_record


def load_training_roles(dataset, fold_file, hashes):
    with np.load(fold_file, allow_pickle=False) as fold:
        for role, value in hashes.items():
            if digest(fold[role], "<i8") != value:
                raise ValueError(f"Archived {role} IDs differ from the verified split")
        fit, search = fold["fit"], fold["search"]
        X_fit = dataset.X[fit].tocsr()
        X_fit.eliminate_zeros()
        columns = np.unique(X_fit.indices)
        np.testing.assert_array_equal(columns, fold["fit_columns"])
        np.testing.assert_array_equal(columns + 1, fold["source_feature_ids"])
    labels = np.asarray(dataset.y)
    return SearchData(
        X_fit[:, columns], labels[fit], dataset.X[search][:, columns], labels[search]
    ), columns


def verify_replay(selected, report, ensemble, reference, columns):
    history = json.loads((reference / "run.json").read_text())["folds"][0]
    old = VotingEnsemble.load(reference / "fold_00/ensemble.npz")
    assert_equal = np.testing.assert_array_equal
    with np.load(reference / "fold_00/fold.npz") as fold:
        assert_equal(selected.initial_weights, fold["initial_weights"])
        assert_equal(selected.next_weights, fold["next_weights"])
    assert_equal(len(history["weight_history"]), len(selected.history))
    for h, saved in zip(selected.history, history["weight_history"], strict=True):
        assert_equal(h["sampling_weights_sha256"], saved["sampling_weights_sha256"])
        for g, s in zip(h["groups"], saved["groups"], strict=True):
            assert_equal(g["accuracies"], s["accuracies"])
            assert_equal(g["selected_fit_columns"], s["selected_fit_columns"])
    for i, (m, previous) in enumerate(zip(ensemble.members, old.members, strict=True)):
        assert_equal(m.class_prior_, previous.class_prior_)
        assert_equal(m.feature_prob_, previous.feature_prob_)
        assert_equal(m.feature_indices_, previous.feature_indices_)
        assert_equal(
            columns[m.feature_indices_] + 1, history["final_source_feature_ids"][i]
        )
        member = report["members"][i]
        saved = history["pso"][i]
        assert_equal(member["seed"], saved["seed"])
        if json.loads(json.dumps(member["history"])) != saved["history"]:
            raise ValueError("Swarm histories differ from archived development run")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fold_zero_stage_seed(root, stage):
    """Reproduce the two child streams of evaluate_fold(seed=root, fold_index=0)."""
    return int(np.random.SeedSequence([root, 0]).spawn(2)[stage].generate_state(1)[0])


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, default=repo / "outputs/2026-11-25/run-a"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo / "configs/experiments/20news_diagnostics_v1.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(repo / "outputs") or (
        output.exists() and any(output.iterdir())
    ):
        parser.error("Choose a fresh directory under outputs/")
    output.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads(
        (args.config.parent / schedule["base"]).read_text(encoding="utf-8")
    )
    start = perf_counter()
    source = discover_archive_sources(args.archive)["20news"]
    dataset = read_document_counts(source, n_features=base["declared_features"])
    if dataset.source_fingerprint != base["source_fingerprint_sha256"]:
        raise ValueError("Unexpected source fingerprint")
    data, columns = load_training_roles(
        dataset, args.reference / "fold_00/fold.npz", schedule["indices"]
    )
    del dataset
    adaptive, pso = AdaptiveConfig(**base["adaptive"]), PSOConfig(**base["pso"])
    ledger = {
        "source": base["source_fingerprint_sha256"],
        "base": base,
        "schedule": schedule,
        "environment": environment_record(),
        "reference_controls": reference_controls(data, adaptive.alpha),
        "fit_rows": len(data.y_fit),
        "search_rows": len(data.y_search),
        "column_hash": digest(columns, "<i8"),
        "attempts": [],
    }
    cache = {}
    for trial in schedule["trials"]:
        name = trial["name"]
        a = replace(adaptive, **trial.get("adaptive", {}))
        p = replace(pso, **trial.get("pso", {}))
        for seed in trial.get("seeds", [42]):
            feature_root = trial.get("feature_seed", seed)

            search_seed = fold_zero_stage_seed(feature_root, 0)
            pso_seed = fold_zero_stage_seed(seed, 1)
            key = (a, search_seed)
            run_id = f"{name}-{seed}"
            item = {
                "id": run_id,
                "adaptive": asdict(a),
                "pso": asdict(p),
                "seed": seed,
                "feature_seed": feature_root,
                "sampling_seed": search_seed,
                "pso_seed": pso_seed,
            }
            ledger["attempts"].append(item)
            item["status"] = "running"
            write_json(output / "inventory.json", ledger)
            try:
                if perf_counter() - start > schedule["budget_seconds"]:
                    item["status"] = "budget_skipped"
                    continue
                if key not in cache:
                    selected, elapsed = select_features(data, a, search_seed)
                    path = f"features-{run_id}.json"
                    write_json(output / path, feature_diagnostics(data, selected, a.c0))
                    cache[key] = selected, elapsed, path
                selected, elapsed, path = cache[key]
                report, model = compare_models(
                    data,
                    selected,
                    a,
                    p,
                    pso_seed,
                    elapsed,
                    random_control=trial.get("random_control", False),
                    prefix_sizes=trial.get("prefix_sizes", ()),
                )
                if name == "reference" and seed == 42:
                    verify_replay(selected, report, model, args.reference, columns)
                    item["archived_replay"] = "matched"
                write_json(output / f"{run_id}.json", report)
                item.update(status="ok", features=path, record=f"{run_id}.json")
                print(run_id, "correct:", report["final_vote"]["correct"], flush=True)
            except Exception as exc:
                item.update(status="failed", error=repr(exc))
                if name == "reference":
                    raise
            finally:
                ledger["wall_seconds"] = perf_counter() - start
                write_json(output / "inventory.json", ledger)
    if any(item["status"] != "ok" for item in ledger["attempts"]):
        raise SystemExit("Incomplete schedule; inspect inventory.json")


if __name__ == "__main__":
    main()
