"""Regression checks for class-distribution preservation and explicit exclusions."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from feature_pipeline import argument_parser
from afs_rgmnb.data.readers import Record
from afs_rgmnb.data.preprocessing import (
    choose_labels, label_selection_report, sample_records, sampling_settings,
)
from afs_rgmnb.features.pipeline import run, validated_splits


class SamplingModeTests(unittest.TestCase):
    def arguments(self, *extra):
        return argument_parser().parse_args([
            "extract", "--kind", "text", "--input", "unused.csv", "--output", "unused",
            "--min-classes", "2", "--min-total-samples", "1", "--min-features", "1",
            *extra,
        ])

    def test_all_mode_has_no_legacy_balance_or_class_cap(self):
        args = self.arguments("--sampling-mode", "all")
        counts = Counter({f"class{i}": i + 1 for i in range(12)})
        self.assertEqual(len(choose_labels(counts, args)), 12)
        self.assertEqual(sampling_settings(args)["class_size_threshold"], 1)
        records = [Record(f"{label}-{i}", label, f"{label} unique{i}")
                   for label, size in counts.items() for i in range(size)]
        with patch("afs_rgmnb.data.preprocessing.stream", return_value=iter(records)):
            retained, audit = sample_records(args, list(counts), return_report=True)
        self.assertEqual(Counter(r.label for r in retained), counts)
        self.assertEqual(audit["downsampled_unique_records"], 0)

    def test_max_classes_zero_is_not_all_record_sampling(self):
        args = self.arguments("--max-classes", "0", "--min-class-size", "1",
                              "--samples-per-class", "2")
        records = [Record(f"{label}-{i}", label, f"{label} unique{i}")
                   for label, size in (("a", 4), ("b", 6)) for i in range(size)]
        with patch("afs_rgmnb.data.preprocessing.stream", return_value=iter(records)):
            with self.assertLogs("afs-rgmnb", level="WARNING"):
                retained, audit = sample_records(args, ["a", "b"], return_report=True)
        self.assertEqual(Counter(r.label for r in retained), {"a": 2, "b": 2})
        self.assertEqual(audit["downsampled_unique_records"], 6)

    def test_explicit_class_exclusions_are_reported(self):
        args = self.arguments("--sampling-mode", "all", "--min-class-size", "5",
                              "--exclude-class", "d")
        report = label_selection_report(Counter(a=10, b=20, c=3, d=9), args)
        self.assertEqual(report["selected_labels"], ["b", "a"])
        self.assertEqual({r["label"]: r["reason"] for r in report["excluded_classes"]},
                         {"c": "below_class_size_threshold", "d": "explicit_exclude_list"})
        self.assertEqual(report["excluded_class_records"], 12)

    def test_duplicates_and_conflicting_labels_have_disjoint_counts(self):
        args = self.arguments("--sampling-mode", "all")
        records = [Record("1", "a", "same"), Record("2", "b", "same"),
                   Record("3", "a", "duplicate"), Record("4", "a", "duplicate"),
                   Record("5", "b", "unique"), Record("6", "a", "")]
        with patch("afs_rgmnb.data.preprocessing.stream", return_value=iter(records)):
            retained, audit = sample_records(args, ["a", "b"], return_report=True)
        self.assertEqual(len(retained), 2)
        self.assertEqual(audit["conflicting_label_records_removed"], 2)
        self.assertEqual(audit["duplicate_content_records_removed"], 1)
        self.assertEqual(audit["invalid_content_records"], 1)
        self.assertEqual(sum(audit[key] for key in (
            "conflicting_label_records_removed", "duplicate_content_records_removed",
            "invalid_content_records", "excluded_label_records", "downsampled_unique_records",
            "retained_records")), audit["reader_emitted_records"])

    def test_duplicate_sample_ids_fail_before_splitting(self):
        args = self.arguments("--sampling-mode", "all")
        records = [Record("id", "a", "different content"), Record("id", "b", "other")]
        with patch("afs_rgmnb.data.preprocessing.stream", return_value=iter(records)):
            with self.assertRaisesRegex(ValueError, "duplicate sample identifier"):
                sample_records(args, ["a", "b"])

    def test_fivefold_rejects_rare_classes_without_removing_them(self):
        args = self.arguments("--sampling-mode", "all", "--folds", "5")
        with self.assertRaisesRegex(ValueError, "No classes were automatically removed"):
            validated_splits(np.array([0] * 4 + [1] * 20), ["rare", "common"], args)

    def test_inner_split_must_represent_every_class(self):
        args = self.arguments("--sampling-mode", "all", "--folds", "5",
                              "--search-fraction", "0.01")
        with self.assertRaisesRegex(ValueError, "Dfit/Dsearch"):
            validated_splits(np.array([0] * 10 + [1] * 20), ["a", "b"], args)

    def test_all_mode_end_to_end_retains_unequal_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "records.csv", root / "output"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "label", "text"])
                for label, size in (("a", 10), ("b", 20)):
                    for index in range(size):
                        writer.writerow([f"{label}{index}", label,
                                         f"topic{label} individual{label}{index} repeated repeated"])
            args = self.arguments("--sampling-mode", "all", "--input", str(source),
                                  "--output", str(output), "--folds", "5")
            manifest = run(args)
            self.assertEqual(manifest["selected_class_counts"], {"a": 10, "b": 20})
            self.assertEqual(manifest["n_selected"], 30)
            self.assertEqual(len(manifest["folds"]), 5)
            self.assertEqual(manifest["sampling_audit"]["downsampled_unique_records"], 0)
            self.assertTrue((output / "sampling_audit.json").is_file())
            for fold in manifest["folds"]:
                for role in ("fit", "search", "test"):
                    self.assertEqual(set(fold["class_distribution"][role]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
