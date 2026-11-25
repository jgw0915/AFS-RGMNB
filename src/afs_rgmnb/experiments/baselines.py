"""Building blocks shared by proposed Q08 protocols. No selection policy is implicit."""

from __future__ import annotations

import numpy as np

from ..methods.mnb import _counts, class_statistics


def document_frequencies(X_fit, y_fit):
    """Return fit document-presence counts by class, with class sizes and order.

    Matrix entry [c,j] is the A cell of the term-present vs class-c 2×2 table.
    The other cells follow from term, class and sample totals. Repetitions in
    a document count once here; the caller's occurrence matrix is unchanged.
    """
    presence = _counts(X_fit)
    presence.data.fill(1)
    return class_statistics(presence, y_fit)


def bootstrap_indices(n_rows, *, sample_size, n_members, rng):
    """Yield independent ordinary bootstrap row indices, retaining multiplicities.

    The caller specifies draw size and handles absent classes explicitly.
    """
    for value in (n_rows, sample_size, n_members):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError("Row, draw and member counts must be positive integers")
    for _ in range(n_members):
        yield rng.integers(0, n_rows, size=sample_size)
