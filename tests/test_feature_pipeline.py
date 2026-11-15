import csv
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("feature_pipeline", ROOT / "feature_pipeline.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class KmerTests(unittest.TestCase):
    def test_overlapping_exact_counts(self):
        counts = module.count_kmers("ACGTACGTAC", 8)
        named = {module.kmer_to_text(k, 8): v for k, v in counts.items()}
        self.assertEqual(named, {"ACGTACGT": 1, "CGTACGTA": 1, "GTACGTAC": 1})

    def test_ambiguous_does_not_bridge(self):
        self.assertEqual(module.count_kmers("AAAAAAAA-N-CCCCCCCC", 8),
                         {module.count_kmers("AAAAAAAA", 8).popitem()[0]: 1,
                          module.count_kmers("CCCCCCCC", 8).popitem()[0]: 1})

    def test_uracil(self):
        self.assertEqual(module.count_kmers("ACGUACGU", 8), module.count_kmers("ACGTACGT", 8))

    def test_rank_schemas_and_canonical_path(self):
        a = module.parse_taxonomy("Bacteria;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;", "six", "Bacteria")
        self.assertTrue(module.label_from_taxonomy(a, "family").endswith("Lactobacillaceae"))
        self.assertTrue(module.label_from_taxonomy(a, "genus").endswith("Lactobacillus"))
        b = module.parse_taxonomy("Bacteria;Kingdom1;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;", "seven", "Bacteria")
        self.assertTrue(module.label_from_taxonomy(b, "family").endswith("Lactobacillaceae"))
        with self.assertRaises(ValueError):
            module.parse_taxonomy("Bacteria;Kingdom1;Firmicutes;Bacilli;Lactobacillales;Lactobacillaceae;Lactobacillus;", "auto", "Bacteria")


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "feature_pipeline.py"), *map(str, args)],
                              check=True, capture_output=True, text=True)

    def assert_outputs(self, out, k_classes=5):
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["selected_class_count"], k_classes)
        self.assertEqual(len(manifest["folds"]), 3)
        for fold in sorted(out.glob("fold_*")):
            matrices = {r: sparse.load_npz(fold / f"X_{r}.npz") for r in ["fit", "search", "test"]}
            self.assertEqual(len({m.shape[1] for m in matrices.values()}), 1)
            with (fold / "features.tsv").open() as f:
                feature_names = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(feature_names), matrices["fit"].shape[1])
            allids = []
            for role, mat in matrices.items():
                self.assertEqual(mat.dtype, np.int32)
                self.assertTrue(np.all(mat.data >= 0))
                label_ids = np.load(fold / f"y_{role}.npy")
                self.assertEqual(mat.shape[0], len(label_ids))
                with (fold / f"ids_{role}.tsv").open() as f:
                    ids = [row["sample_id"] for row in csv.DictReader(f, delimiter="\t")]
                allids.extend(ids)
            self.assertEqual(len(allids), len(set(allids)))

    def test_dada2_family_folds(self):
        fasta = self.root / "ref.fa"
        rng = random.Random(55)
        with fasta.open("w") as f:
            for cls in range(5):
                for j in range(20):
                    f.write(f">Bacteria;Phylum;Class;Order;Family{cls};Genus{cls};\n")
                    seq = "".join(rng.choice("ACGT") for _ in range(90))
                    f.write(seq + "\n")
        out = self.root / "dna_out"
        self.cli("extract", "--kind", "dna", "--input", fasta, "--output", out,
                 "--rank", "family", "--rank-schema", "six", "--min-classes", 5,
                 "--min-class-size", 12, "--samples-per-class", 12,
                 "--min-total-samples", 40, "--min-features", 1,
                 "--folds", 3, "--seed", 7, "--baseline")
        self.assert_outputs(out)

    def test_text_csv_folds(self):
        csvpath = self.root / "docs.csv"
        with csvpath.open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["id", "label", "text"])
            for cls in range(5):
                for j in range(16):
                    wr.writerow([f"id_{cls}_{j}", f"cat_{cls}",
                                 f"Topic{cls} Training Word{j} word{j + 1} pattern{cls} Features"])
        out = self.root / "txt_out"
        self.cli("extract", "--kind", "text", "--input", csvpath, "--output", out,
                 "--min-class-size", 10, "--samples-per-class", 10,
                 "--min-total-samples", 40, "--min-features", 1, "--folds", 3)
        self.assert_outputs(out)


if __name__ == "__main__":
    unittest.main()