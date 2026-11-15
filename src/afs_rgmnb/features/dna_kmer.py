from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from scipy import sparse
DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

def kmer_to_text(index: int, k: int) -> str:
    arr = ["A"] * k
    for i in range(k - 1, -1, -1):
        arr[i] = "ACGT"[index & 3]
        index >>= 2
    return "".join(arr)


def count_kmers(seq: str, k: int) -> Dict[int, int]:
    """Rolling two-bit index, overlapping windows, no ambiguous-base bridging."""
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 15:
        raise ValueError("k must be an integer between 1 and 15")
    mask, code, length = (1 << (2 * k)) - 1, 0, 0
    counts: Dict[int, int] = {}
    for ch in seq.upper():
        if ch == "U":  # RNA origin
            ch = "T"
        digit = DNA_MAP.get(ch)
        if digit is None:
            code, length = 0, 0
            continue
        code = ((code << 2) | digit) & mask
        length += 1
        if length >= k:
            counts[code] = counts.get(code, 0) + 1
    return counts


def sparse_dna_matrix(texts: Sequence[str], k: int, vocab: Optional[np.ndarray]) -> sparse.csr_matrix:
    if vocab is None:
        ncols = 4 ** k
        lookup = None
    else:
        ncols = len(vocab)
        lookup = {int(feature): i for i, feature in enumerate(vocab)}
    row_ind, col_ind, values = [], [], []
    for row, seq in enumerate(texts):
        for col, count in count_kmers(seq, k).items():
            mapped = col if lookup is None else lookup.get(col)
            if mapped is not None:
                row_ind.append(row)
                col_ind.append(mapped)
                values.append(count)
    return sparse.csr_matrix((np.asarray(values, dtype=np.int32),
                              (row_ind, col_ind)), shape=(len(texts), ncols), dtype=np.int32)
