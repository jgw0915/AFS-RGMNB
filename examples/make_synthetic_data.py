#!/usr/bin/env python3
"""Create NON-BIOLOGICAL synthetic data solely for software smoke tests."""
from pathlib import Path
import argparse
import csv
import gzip
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_data"))
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out / "synthetic_dada2.fa.gz", "wt", encoding="utf-8") as fdna, \
         (args.out / "synthetic_documents.csv").open("w", newline="", encoding="utf-8") as ftxt:
        wr = csv.writer(ftxt)
        wr.writerow(["id", "label", "text"])
        for cls in range(args.classes):
            for sample in range(args.samples):
                fdna.write(f">Bacteria;PhylumX;ClassX;OrderX;Family_{cls};Genus_{cls};\n")
                fdna.write("".join(rng.choices("ACGT", k=700)) + "\n")
                terms = [f"common{rng.randrange(100)}" for _ in range(20)]
                terms += [f"class{cls}term{rng.randrange(1000)}" for _ in range(12)]
                terms += [f"specific{cls}_{sample}_{j}" for j in range(20)]
                wr.writerow([f"doc_{cls}_{sample}", f"category_{cls}", " ".join(terms)])
    print(f"Wrote synthetic data to {args.out} (NOT a scientific benchmark)")


if __name__ == "__main__":
    main()