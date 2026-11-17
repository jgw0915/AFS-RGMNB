"""Output destinations must never overwrite raw inputs or research references."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from afs_rgmnb.utils.io import ensure_output_outside_references


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace with spaces" / "AFS-RGMNB"
        self.root.mkdir(parents=True)

    def test_readonly_roots_and_descendants_are_rejected(self):
        for root in (self.root / "reference", self.root / "data" / "raw",
                     self.root.parent / "Master-Thesis"):
            for output in (root, root / "another senior" / "results"):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "read-only"):
                    ensure_output_outside_references(output, repo_root=self.root)
                self.assertFalse(output.exists())

    def test_dotdot_resolution_cannot_bypass_readonly_roots(self):
        output = self.root / "outputs" / ".." / "reference" / "other archive" / "result"
        with self.assertRaisesRegex(ValueError, "read-only"):
            ensure_output_outside_references(output, repo_root=self.root / "data" / "..")

    def test_custom_archive_protects_external_and_legacy_sources(self):
        archive = self.root.parent / "external archive"
        for output in (archive, archive / "results"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "read-only"):
                ensure_output_outside_references(output, repo_root=self.root, archive_root=archive)

    def test_extracted_outputs_and_external_results_are_allowed_without_writes(self):
        for output in (self.root / "data" / "extracted" / "dataset" / "version",
                       self.root / "outputs" / "run", self.root.parent / "external results",
                       self.root / "reference_results", self.root / "data" / "raw_counts"):
            with self.subTest(output=output):
                resolved = ensure_output_outside_references(output, repo_root=self.root)
                self.assertEqual(resolved, output.resolve())
                self.assertFalse(resolved.exists())

    def test_all_audit_runners_reject_another_reference_before_reading_or_writing(self):
        repo = Path(__file__).resolve().parents[1]
        # Different archives under reference/: checking only --archive missed
        # this destination even though all reference materials are read-only.
        output = repo / "reference" / "output_guard_probe_do_not_create"
        self.assertFalse(output.exists())
        for script in ("run_pre_pso.py", "run_pso.py", "validate_senior_datasets.py"):
            with self.subTest(script=script):
                process = subprocess.run(
                    [sys.executable, str(repo / "scripts" / script),
                     "--archive", str(repo / "reference" / "different_archive_probe"),
                     "--output", str(output)], cwd=repo, capture_output=True,
                    text=True, encoding="utf-8")
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertIn("read-only", process.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
