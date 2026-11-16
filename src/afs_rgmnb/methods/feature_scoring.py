"""Chapter 3 equations (3.1)-(3.3): count-event JMH, fitted on Dfit only.

Input counts remain sparse. Class aggregates are sparse; only individual
length-d probability/score vectors are dense. Bernoulli here splits one count
event into feature j versus the other features, not sample-level presence.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import rel_entr


def jmh_scores(X_fit, y_fit, *, alpha: float) -> np.ndarray:
    """Sum unweighted class KLs against empirical-prior-weighted complements.

    Natural logarithms, float64. Reject undefined or unrepresentable regimes
    rather than silently clipping probabilities or substituting an entropy score.
    This API accepts only fit data; no search/test input participates in scoring.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    X = sparse.csr_matrix(X_fit)
    if X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("JMH requires nonempty fit data with at least two features")
    if not np.isrealobj(X.data):
        raise ValueError("X_fit counts must be real")
    if (not np.all(np.isfinite(X.data)) or np.any(X.data < 0)
            or np.any(X.data != np.floor(X.data))):
        raise ValueError("X_fit must contain finite nonnegative integer counts")
    y = np.asarray(y_fit)
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y_fit must be one-dimensional and aligned with X_fit")
    if any(label is None for label in y):
        raise ValueError("Missing class labels are not allowed")
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError("Class labels must be finite")
    classes, encoded, sizes = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("JMH requires at least two represented classes")
    n, d = X.shape
    assignment = sparse.csr_matrix((np.ones(n), (encoded, np.arange(n))),
                                    shape=(len(classes), n))
    counts = (assignment @ X.astype(np.float64)).tocsr()
    denominators = np.asarray(counts.sum(axis=1)).ravel() + d * alpha
    priors = sizes / float(n)

    def feature_prob(class_index):
        values = np.full(d, alpha, dtype=np.float64)
        row = counts.getrow(class_index)
        values[row.indices] += row.data
        values /= denominators[class_index]
        if not np.all(np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
            raise ValueError("Smoothed probabilities cannot be represented safely in float64 (0,1)")
        return values

    mixture = np.zeros(d, dtype=np.float64)
    reference = feature_prob(0)
    identical_columns = np.ones(d, dtype=bool)
    for class_index, prior in enumerate(priors):
        p = feature_prob(class_index)
        mixture += prior * p
        identical_columns &= p == reference
    scores = np.zeros(d, dtype=np.float64)
    for class_index, prior in enumerate(priors):
        p = feature_prob(class_index)
        # Exactly equation (3.2); class-frequency weights enter only this mixture.
        complement = (mixture - prior * p) / (1 - prior)
        if np.any((complement <= 0) | (complement >= 1)):
            raise ValueError("Complement probabilities outside representable (0,1)")
        kl = rel_entr(p, complement) + rel_entr(1 - p, 1 - complement)
        if not np.all(np.isfinite(kl)) or np.any(kl < -1e-12):
            raise ValueError("Nonfinite or materially negative JMH divergence")
        # Remove only negative cancellation within 1e-12, not positive scores.
        scores += np.maximum(kl, 0)
    # If all class probabilities are exactly equal, KL is exactly zero by the
    # definition. Avoid spurious positive scores from complement-mixture rounding;
    # no tolerance is used to erase small genuinely discriminative differences.
    scores[identical_columns] = 0
    return scores
