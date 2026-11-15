import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afs_rgmnb.features.document import (
    DocumentAnalyzer, DocumentPreprocessing, document_vectorizer,
    preprocessing_from_args, strip_rfc822_headers,
)


class DocumentPreprocessingTests(unittest.TestCase):
    def test_hyphen_stopwords_porter_and_repeated_counts(self):
        config = DocumentPreprocessing(stopwords=("the", "and"), stopword_source="test")
        vectorizer = document_vectorizer(config)
        X = vectorizer.fit_transform(["The Real-time systems, systems! AND connected."])
        actual = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
        self.assertEqual(actual, {"real-tim": 1, "system": 2, "connect": 1})
        self.assertEqual(X.dtype, np.int32)

    def test_hyphen_preserved_before_stemming_and_single_char_tokens(self):
        analyzer = DocumentAnalyzer(DocumentPreprocessing(porter_mode="none"))
        self.assertEqual(analyzer("Real-time; A x 7 --left-- a_b"),
                         ["real-time", "a", "x", "7", "left", "a", "b"])

    def test_holdout_only_word_does_not_enter_vocabulary(self):
        vectorizer = document_vectorizer(DocumentPreprocessing(porter_mode="none"))
        vectorizer.fit_transform(["training training word"])
        X = vectorizer.transform(["unseen training"])
        self.assertNotIn("unseen", vectorizer.vocabulary_)
        self.assertEqual(int(X.sum()), 1)

    def test_headers_are_explicit_and_prose_is_preserved(self):
        raw = "Subject: secret\nFrom: person\n\nBody words"
        self.assertEqual(strip_rfc822_headers(raw), "Body words")
        self.assertEqual(strip_rfc822_headers("Prose first\n\nMore prose"),
                         "Prose first\n\nMore prose")
        self.assertIn("subject", DocumentAnalyzer(DocumentPreprocessing(porter_mode="none"))(raw))

    def test_stoplist_not_invented_and_matrices_rejected(self):
        with self.assertRaisesRegex(ValueError, "historical stoplist unknown"):
            preprocessing_from_args(Namespace(stopwords="none"))
        with self.assertRaises(TypeError):
            DocumentAnalyzer(DocumentPreprocessing(porter_mode="none"))(np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
