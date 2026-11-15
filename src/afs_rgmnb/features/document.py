"""Explicit raw-text preprocessing; never apply this to existing count matrices.

Tsai (2019), section 3.1, specifies hyphen-preserving tokenization, lowercase,
stopword removal and Porter stemming. The historical full stoplist and exact
stemmer implementation are unavailable, so callers must name their choices.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Letters/digits, including one-character tokens; ASCII hyphen inside a word.
# Repeated/edge hyphens, underscores and other punctuation delimit tokens.
TOKEN_PATTERN = r"[^\W_]+(?:-[^\W_]+)*"


@dataclass(frozen=True)
class DocumentPreprocessing:
    lowercase: bool = True
    stopwords: tuple[str, ...] = ()
    stopword_source: str = "explicit-empty-list"
    porter_mode: str = "original"
    remove_headers: bool = False

    def __post_init__(self):
        if self.porter_mode not in {"none", "original", "martin", "nltk"}:
            raise ValueError("porter_mode must be none, original, martin or nltk")


def strip_rfc822_headers(text: str) -> str:
    """Optional raw-email policy; not asserted to reproduce the senior code."""
    parts = re.split(r"\r?\n\r?\n", text, maxsplit=1)
    if len(parts) == 1:
        return text
    lines = parts[0].splitlines()
    # Only strip a syntactically plausible initial header block, not arbitrary prose.
    if lines and re.match(r"^[A-Za-z][A-Za-z0-9-]*:", lines[0]) and all(
        line.startswith((" ", "\t")) or re.match(r"^[A-Za-z][A-Za-z0-9-]*:", line)
        for line in lines
    ):
        return parts[1]
    return text


class DocumentAnalyzer:
    def __init__(self, config: DocumentPreprocessing):
        self.config = config
        self.stopwords = frozenset(
            word.lower() if config.lowercase else word for word in config.stopwords
        )
        self.stemmer = None
        if config.porter_mode != "none":
            try:
                from nltk.stem import PorterStemmer
            except ImportError as exc:
                raise ImportError("Porter stemming requires nltk; install project requirements") from exc
            modes = {"original": PorterStemmer.ORIGINAL_ALGORITHM,
                     "martin": PorterStemmer.MARTIN_EXTENSIONS,
                     "nltk": PorterStemmer.NLTK_EXTENSIONS}
            self.stemmer = PorterStemmer(mode=modes[config.porter_mode])

    def __call__(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("DocumentAnalyzer accepts raw strings, not count vectors")
        if self.config.remove_headers:
            text = strip_rfc822_headers(text)
        if self.config.lowercase:
            text = text.lower()
        tokens = re.findall(TOKEN_PATTERN, text)
        tokens = [word for word in tokens if word not in self.stopwords]
        if self.stemmer is not None:
            tokens = [self.stemmer.stem(word, to_lowercase=False) for word in tokens]
        return tokens


def document_vectorizer(config: DocumentPreprocessing, *, min_df: int = 1,
                        max_features: int | None = None) -> CountVectorizer:
    return CountVectorizer(analyzer=DocumentAnalyzer(config), min_df=min_df,
                           max_features=max_features, dtype=np.int32)


def preprocessing_from_args(args) -> DocumentPreprocessing:
    path = getattr(args, "stopword_file", None)
    source = getattr(args, "stopwords", "none")
    if path is not None:
        path = Path(path)
        raw = path.read_bytes()
        words = tuple(line.strip() for line in raw.decode("utf-8-sig").splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
        provenance = f"file-sha256:{hashlib.sha256(raw).hexdigest()}"
    elif source == "english":
        words = tuple(sorted(ENGLISH_STOP_WORDS))
        provenance = "scikit-learn-english (alternative, not verified historical stoplist)"
    else:
        raise ValueError("senior document preprocessing requires --stopword-file or an "
                         "explicit --stopwords english alternative; historical stoplist unknown")
    return DocumentPreprocessing(stopwords=words, stopword_source=provenance,
                                 porter_mode=getattr(args, "porter_mode", "original"),
                                 remove_headers=getattr(args, "remove_headers", False))


def vectorizer_from_args(args) -> CountVectorizer:
    if getattr(args, "document_preprocessing", "legacy") == "senior":
        return document_vectorizer(preprocessing_from_args(args), min_df=args.min_df,
                                   max_features=args.max_features or None)
    return CountVectorizer(lowercase=True, strip_accents="unicode",
                           stop_words="english" if args.stopwords == "english" else None,
                           min_df=args.min_df, max_features=args.max_features or None,
                           token_pattern=args.token_pattern, dtype=np.int32)
