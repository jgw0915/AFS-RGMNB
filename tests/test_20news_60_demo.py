"""Offline tests of the 60-document demo exporter (no sklearn download)."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.export_20news_60_demo import export_demo, select_indices


def synthetic_bunch():
    # Synthetic fixtures only. Real advisor CSVs must be obtained via sklearn.
    items = []
    labels = []
    for idx in range(60):
        items.append(
            f"From: person{idx}@example.invalid\nSubject: Example\n\n"
            "The real-time systems systems and more words.\n"
        )
        labels.append(idx % 20)
    return SimpleNamespace(data=items, target=labels,
                           target_names=[f"class_{i}" for i in range(20)])


class Export20NewsDemoTests(unittest.TestCase):
    def test_first60_and_three_per_class(self):
        y = [i % 20 for i in range(80)]
        self.assertEqual(select_indices(y, "first60"), list(range(60)))
        picks = select_indices(y, "three_per_class")
        self.assertEqual(len(picks), 60)
        self.assertEqual(len({y[i] for i in picks}), 20)
        self.assertEqual(len(set(picks)), 60)

    @patch("scripts.export_20news_60_demo.fetch_20newsgroups")
    def test_export_uses_project_vectorizer_and_true_count_csv(self, fetch):
        fetch.return_value = synthetic_bunch()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            result = export_demo(output=output)
            self.assertEqual(result["sample_count"], 60)
            self.assertEqual(len(result["class_distribution"]), 20)
            self.assertEqual(result["observed_vocabulary_features"], 3)
            original = list(csv.DictReader(
                (output / "20news_original_60.csv").open(encoding="utf-8-sig", newline="")))
            observed = list(csv.DictReader(
                (output / "20news_extracted_word_counts_60.csv").open(
                    encoding="utf-8-sig", newline="")))
            self.assertEqual(len(original), 60)
            self.assertIn("From: person0@", original[0]["document_text"])
            records = {(row["sample_id"], row["term"]): int(row["occurrence_count"])
                       for row in observed}
            self.assertEqual(records[("sklearn20ng_00000", "system")], 2)
            self.assertEqual(records[("sklearn20ng_00000", "real-tim")], 1)
            self.assertFalse(any(row["term"] in ("subject", "person0", "the")
                                 for row in observed))
            self.assertEqual({r["sample_id"] for r in original},
                             {r["sample_id"] for r in observed})
            manifest = json.loads((output / "demo_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["project_component"],
                             "afs_rgmnb.features.document.document_vectorizer")
            self.assertTrue(manifest["original_csv_is_unmodified_downloaded_text"])
            self.assertFalse(manifest["historical_senior_dataset_equivalent"])
            self.assertEqual(len(fetch.call_args.kwargs["remove"]), 0)

    @patch("scripts.export_20news_60_demo.fetch_20newsgroups")
    def test_optional_metadata_removal_is_declared(self, fetch):
        fetch.return_value = synthetic_bunch()
        with tempfile.TemporaryDirectory() as tmp:
            result = export_demo(output=Path(tmp), selection="three_per_class",
                                 strip_metadata=True)
            self.assertFalse(result["original_csv_is_unmodified_downloaded_text"])
            self.assertEqual(result["download_remove"], ["headers", "footers", "quotes"])
            self.assertFalse(result["project_analyzer_remove_headers"])


if __name__ == "__main__":
    unittest.main()
