import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from feature_pipeline import text_records, parse_taxonomy, label_from_taxonomy, count_kmers


class ReaderRegressionTests(unittest.TestCase):
    def test_latin1_is_explicit_and_invalid_utf8_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "docs.csv"
            path.write_bytes("id,label,text\n1,a,café\n".encode("latin-1"))
            args = Namespace(input=path, text_format="csv", label_col="label", text_col="text",
                             id_col="id", delimiter=",", encoding="utf-8", encoding_errors="strict")
            with self.assertRaises(UnicodeDecodeError):
                list(text_records(args))
            args.encoding = "latin-1"
            self.assertEqual(list(text_records(args))[0].content, "café")

    def test_json_null_labels_and_bodies_are_not_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "docs.jsonl"
            path.write_text(json.dumps({"label": None, "text": None}) + "\n", encoding="utf-8")
            args = Namespace(input=path, text_format="jsonl", label_col="label", text_col="text", id_col="id")
            self.assertEqual(list(text_records(args)), [])

    def test_missing_internal_taxonomy_rank_does_not_shift_labels(self):
        tax = parse_taxonomy("Bacteria;P;;Order;Family;Genus;", "six", "Bacteria")
        self.assertEqual(tax["family"], "Family")
        self.assertIsNone(label_from_taxonomy(tax, "family"))

    def test_invalid_k_is_rejected_in_public_api(self):
        for k in [0, -1, 16, True, 2.5]:
            with self.assertRaises(ValueError):
                count_kmers("ACGT", k)
