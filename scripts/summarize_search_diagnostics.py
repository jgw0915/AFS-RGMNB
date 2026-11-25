"""Build aggregate Markdown tables and a figure from completed diagnostics (matplotlib required)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"

    return "\n".join(
        [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]
    )


def describe(r):
    m = r["members"]
    steps = sum(x["particle_updates"] for x in m)
    coords = sum(x["particle_updates"] * x["dimension"] for x in m)

    def ratio(key, denom):
        return 100 * sum(x[key] for x in m) / denom if denom else 0

    return {
        "reset": ratio("resets", steps),
        "coords": ratio("redrawn_coords", coords),
        "velocity": ratio("repaired_velocities", steps),
        "gain": 100 * np.mean([x["gain"] for x in m]),
        "pbest": ratio("pbest_improvements", steps),
        "gbest": ratio("gbest_improvements", sum(len(x["history"]) - 1 for x in m)),
        "evaluations": sum(x["fitness_evaluations"] for x in m),
    }


def build_tables(runs, ref_dir):
    inventory, repairs, controls = [], [], []
    for phase, item, r in runs:
        if r is None:
            inventory.append([phase, item["id"], item["status"]] + ["—"] * 8)
            continue
        values = describe(r)
        a, p = item["adaptive"], item["pso"]
        cfg = "/".join(
            str(a[k])
            for k in ("n_groups", "candidates_per_group", "subspace_size", "tmax")
        )
        swarm = f"{p['n_particles']}/{p['max_iterations']}/{p['vmax']:g}"
        accuracies = [
            f"{100 * r[k]['accuracy']:.2f}"
            for k in ("fitted_vote", "initial_vote", "final_vote")
        ]
        times = [f"{r['seconds'][k]:.3f}" for k in ("feature", "pso", "build")]
        inventory.append(
            [phase, item["id"], cfg, swarm, item["feature_seed"], *accuracies, *times]
        )
        repairs.append(
            [
                phase,
                item["id"],
                values["evaluations"],
                f"{values['gain']:.2f}",
                *(
                    f"{values[k]:.1f}"
                    for k in ("reset", "coords", "velocity", "pbest", "gbest")
                ),
            ]
        )
        if "budget_matched_random_vote" in r:
            controls.append(
                [
                    item["id"],
                    r["initial_vote"]["correct"],
                    r["budget_matched_random_vote"]["correct"],
                    r["final_vote"]["correct"],
                    values["evaluations"],
                ]
            )
    rounds = read(ref_dir / "features-reference-42.json")
    round_rows, density = [], []
    for h in rounds:
        pct = lambda x: f"{100 * x:.2f}"
        round_rows.append(
            [
                h["iteration"],
                h["evaluations"],
                *(pct(h[k]) for k in ("min", "mean", "max")),
                ", ".join(map(pct, h["selected_accuracies"])),
                h["union"],
                f"{np.mean(h['pair_jaccard']):.3f}",
                f"{h['delta']:.4f}",
                f"{h['after']['effective_features']:.1f}",
                pct(h["after"]["top_1pct_mass"]),
            ]
        )
        for g, group in enumerate(h["groups"]):
            c = h["search_coverage"][g][group["selected_index"]]
            counts = np.repeat(np.arange(len(c["nnz_histogram"])), c["nnz_histogram"])
            density.append(
                [
                    h["iteration"],
                    g + 1,
                    c["zeros"],
                    pct(c["zero_fraction"]),
                    f"{c['mean_nnz']:.2f}",
                    *np.quantile(counts, [0.25, 0.5, 0.75], method="nearest"),
                ]
            )
    ref = read(ref_dir / "reference-42.json")
    diagnostics = [ref[k] for k in ("fitted_vote", "initial_vote", "final_vote")]
    recalls = [
        [
            c,
            sum(diagnostics[0]["confusion"][i]),
            *(m["predicted_counts"][i] for m in diagnostics),
            *(f"{100 * m['recall'][i]:.1f}" for m in diagnostics),
        ]
        for i, c in enumerate(ref["classes"])
    ]
    return (
        "\n\n".join(
            [
                "# Measured configuration inventory",
                "Every accuracy uses the same 3,020 Dsearch documents. "
                "The suffix is the model root seed. F-root is the feature RNG root. The split seed stays 42. "
                "Stage times are seconds. Build includes feature search, PSO and construction. "
                "On cache hits the original measured feature time is attributed once to the standalone build.",
                table(
                    [
                        "Phase",
                        "Run",
                        "G/L/m/Tmax",
                        "Np/Imax/vmax",
                        "F-root",
                        "Fitted %",
                        "Initial %",
                        "Final %",
                        "Feature s",
                        "PSO s",
                        "Build s",
                    ],
                    inventory,
                ),
                "## Swarm behavior",
                "Gain is the mean increase in per-member historical-best accuracy in percentage points. "
                "Reset and velocity-repair percentages use particle updates. Coordinate redraw uses updated coordinates. "
                "pbest counts strict improvements per particle update and gbest counts strict improvements per swarm update.",
                table(
                    [
                        "Phase",
                        "Run",
                        "Fitness calls",
                        "Gain pp",
                        "Reset %",
                        "Coord redraw %",
                        "Velocity repaired %",
                        "pbest %",
                        "gbest %",
                    ],
                    repairs,
                ),
                "## Equal-budget independent random control",
                "Correct predictions out of 3,020. "
                "Both searches share the initial particles. The control spends the remaining evaluations on independent draws.",
                table(
                    [
                        "Run",
                        "Initial",
                        "Independent random",
                        "PSO",
                        "Fitness calls per method",
                    ],
                    controls,
                ),
                "## Reference feature rounds",
                "Effective dimension is 1/Σq². Top mass is the weight held by the largest 1% of coordinates after updating.",
                table(
                    [
                        "Round",
                        "Fits",
                        "Min %",
                        "Mean %",
                        "Max %",
                        "Group winners %",
                        "Union",
                        "Mean Jaccard",
                        "L1",
                        "Effective d",
                        "Top mass %",
                    ],
                    round_rows,
                ),
                table(
                    [
                        "Round",
                        "Group",
                        "Zero rows",
                        "Zero %",
                        "Mean nnz",
                        "Q1",
                        "Median",
                        "Q3",
                    ],
                    density,
                ),
                "## Reference class outcomes",
                "Original numeric class order. Counts are document counts and recall is a percentage.",
                table(
                    [
                        "Class",
                        "Actual",
                        "Fitted pred.",
                        "Initial pred.",
                        "PSO pred.",
                        "Fitted recall",
                        "Initial recall",
                        "PSO recall",
                    ],
                    recalls,
                ),
                "Final PSO ensemble confusion matrix. Rows are actual classes 1–20 and columns are predicted classes in the same order.",
                "```text\n"
                + "\n".join(
                    " ".join(f"{n:3d}" for n in row)
                    for row in diagnostics[-1]["confusion"]
                )
                + "\n```",
                "All member/ensemble confusion matrices, predictions by class, recall, particle histories and movement totals "
                "are in the ignored per-run JSON. Per-draw Dfit and Dsearch nonzero histograms are in the feature records.",
            ]
        )
        + "\n"
    )


def plot(runs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    velocity = defaultdict(list)
    for phase, item, r in runs:
        if r is None:
            continue
        name = item["id"].rsplit("-", 1)[0]
        if (
            phase == "OFAT" and (name == "reference" or name.startswith("v"))
        ) or name == "tiny":
            velocity[item["pso"]["vmax"]].append(r)
        if item["seed"] == 42:
            color = "#1b7080" if phase == "OFAT" else "#bf5327"
            axes[2].scatter(
                r["seconds"]["build"],
                100 * r["final_vote"]["accuracy"],
                color=color,
                s=28,
            )
    for i, (v, records) in enumerate(sorted(velocity.items())):
        for ax, key in zip(axes[:2], ("accuracy", "reset")):
            y = [
                100 * r["final_vote"]["accuracy"]
                if key == "accuracy"
                else describe(r)["reset"]
                for r in records
            ]
            ax.scatter([i] * len(y), y, color="#1b7080", alpha=0.75)
            ax.plot([i - 0.2, i + 0.2], [np.mean(y)] * 2, color="#162c46", lw=2)
    for ax in axes[:2]:
        ax.set_xticks(range(len(velocity)), [f"{v:.0e}" for v in sorted(velocity)])
        ax.set_xlabel("Velocity bound")
    axes[0].set(
        title="Fixed subspaces · three PSO seeds",
        ylabel="Dsearch ensemble accuracy (%)",
    )
    axes[1].set(
        title="Boundary handling", ylabel="Particle updates reset (%)", ylim=(-3, 105)
    )
    axes[2].set(
        title="Seed 42 · measured builds",
        xlabel="Standalone build time (seconds)",
        ylabel="Dsearch ensemble accuracy (%)",
    )
    from matplotlib.lines import Line2D

    axes[2].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=c, label=l)
            for c, l in [("#1b7080", "One factor"), ("#bf5327", "Follow-up")]
        ],
        fontsize=8,
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_search_cost(runs, path):
    """Compare the same three seeds, with separate component times and swarm outcomes."""
    import matplotlib.pyplot as plt

    cases = [("OFAT", "reference", "Reference"),
             ("Confirmation", "N50confirm", "50 particles"),
             ("Follow-up", "long20", "20 particles, 50 updates"),
             ("Follow-up", "long20wide", "Same, vmax = 0.001"),
             ("Confirmation", "G10matched", "10 groups, 20 × 20 swarm")]
    labels, feature, pso, gain, redraw = [], [], [], [], []
    for phase, name, label in cases:
        rows = [r for batch, item, r in runs if r is not None
                and phase == batch and item["id"].rsplit("-", 1)[0] == name]
        if not rows:
            continue
        labels.append(label)
        feature.append(np.mean([r["seconds"]["feature"] for r in rows]))
        pso.append(np.mean([r["seconds"]["pso"] for r in rows]))
        gain.append(np.mean([describe(r)["gain"] for r in rows]))
        redraw.append(np.mean([describe(r)["coords"] for r in rows]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, layout="constrained")
    ys = np.arange(len(labels))
    axes[0].barh(ys, feature, color="#1b7080", label="Feature search")
    axes[0].barh(ys, pso, left=feature, color="#bf5327", label="PSO")
    axes[0].set_yticks(ys, labels)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_xlabel("Stage time (seconds)")
    axes[1].barh(ys, gain, color="#1b7080")
    axes[1].set_xlabel("Mean best-particle gain (pp)")
    axes[2].barh(ys, redraw, color="#bf5327")
    axes[2].set(xlabel="Velocity coordinates redrawn (%)", xlim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ofat", type=Path, required=True)
    p.add_argument("--followup", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    runs = []
    for phase, directory in [
        ("OFAT", args.ofat),
        ("Follow-up", args.followup),
        ("Confirmation", args.confirmation),
    ]:
        for item in read(directory / "inventory.json")["attempts"]:
            runs.append(
                (
                    phase,
                    item,
                    read(directory / item["record"])
                    if item["status"] == "ok"
                    else None,
                )
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "measurement_tables.md").write_text(
        build_tables(runs, args.ofat), encoding="utf-8"
    )
    plot(runs, args.output / "sensitivity.png")
    plot_search_cost(runs, args.output / "swarm_cost.png")


if __name__ == "__main__":
    main()
