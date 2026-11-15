"""Strict schema and preservation tests use small synthetic count fixtures."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afs_rgmnb.data.senior_archive import (read_document_counts, read_gene_instances,
    count_vector_duplicates, fit_observed_columns)


class SeniorArchiveTests(unittest.TestCase):
    def test_document_counts_multiclass_and_training_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.txt"
            path.write_text("1,2,1,3,4,1\n2,5,2,8\n3,1,1,3,4,1\n", encoding="ascii")
            dataset = read_document_counts(path, n_features=5)
            self.assertEqual(dataset.y.tolist(), [1, 2, 3])
            self.assertEqual(dataset.X.toarray().tolist(), [[3, 0, 0, 1, 0], [0, 8, 0, 0, 0], [3, 0, 0, 1, 0]])
            self.assertEqual(dataset.historical_folds.tolist(), [2, 5, 1])
            self.assertEqual(fit_observed_columns(dataset.X[[0]]).tolist(), [0, 3])
            duplicates = count_vector_duplicates(dataset)
            self.assertEqual(duplicates["duplicate_excess_rows"], 1)
            self.assertEqual(duplicates["conflicting_label_groups"], 1)

    def test_gene_zero_based_ids_and_cp950_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ClassName_bac.txt").write_bytes("代碼 名稱\n0 TaxonA\n1 TaxonB\n".encode("cp950"))
            for label in [0, 1]:
                (root / str(label)).mkdir()
                (root / str(label) / "0.txt").write_text(f"0 {label + 2}\n3 1\n", encoding="ascii")
            dataset = read_gene_instances(root, n_features=4)
            self.assertEqual(dataset.X.toarray().tolist(), [[2, 0, 0, 1], [3, 0, 0, 1]])
            self.assertEqual(dataset.identifiers, ["0/0.txt", "1/0.txt"])
            self.assertEqual(dataset.label_names, {0: "TaxonA", 1: "TaxonB"})
            self.assertEqual(dataset.source_fingerprint, read_gene_instances(root, 4).source_fingerprint)

    def test_invalid_document_rows_fail_loudly(self):
        for row in ["1,1,1,0", "1,1,0,2", "1,1,4,2", "1,1,1,2,1,3",
                    "1,1,1,-2", "1,1,1,2.0", "1,1,1", "0,1,1,2", ""]:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.txt"
                path.write_text(row, encoding="ascii")
                with self.assertRaises(ValueError):
                    read_document_counts(path, n_features=3)


if __name__ == "__main__":
    unittest.main()
