"""Subspace MNB from active Chapter 3 equations (3.7) and (3.8).

Only Dfit estimates parameters. Conditional probabilities are normalized over
the selected columns, never over the original feature universe. The scoring
function also accepts explicitly supplied valid parameters for later model
search integration; no random generation or PSO is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class ClassStatistics:
    classes: np.ndarray
    class_counts: np.ndarray
    feature_counts: sparse.csr_matrix
    n_samples: int
    n_features: int


@dataclass(frozen=True)
class MNBParameters:
    classes: np.ndarray
    feature_indices: np.ndarray
    class_prior: np.ndarray
    feature_prob: np.ndarray
    n_features_in: int


def _counts(X, *, n_features: int | None = None, allow_empty: bool = False) -> sparse.csr_matrix:
    try:
        if not sparse.issparse(X) and np.asarray(X).ndim != 2:
            raise ValueError("Count input must be a two-dimensional matrix")
        matrix = sparse.csr_matrix(X)
    except (TypeError, ValueError) as exc:
        raise ValueError("Count input must be a two-dimensional numeric matrix") from exc
    if matrix.shape[1] < 2 or (not allow_empty and matrix.shape[0] == 0):
        raise ValueError("Count input needs at least two features and nonempty fit rows")
    if n_features is not None and matrix.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} input columns, received {matrix.shape[1]}")
    try:
        valid = (np.isrealobj(matrix.data) and np.all(np.isfinite(matrix.data))
                 and np.all(matrix.data >= 0)
                 and np.all(matrix.data == np.floor(matrix.data)))
    except TypeError as exc:
        raise ValueError("Counts must be finite nonnegative integers") from exc
    if not valid:
        raise ValueError("Counts must be finite nonnegative integers")
    matrix = matrix.astype(np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Count totals cannot be represented safely in float64")
    return matrix


def _class_labels(values, *, name: str) -> np.ndarray:
    # Validate before NumPy can silently coerce mixed numeric/string labels.
    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional label array")
    kinds = set()
    for label in labels:
        if label is None or isinstance(label, (list, dict, set, tuple)):
            raise ValueError(f"{name} contains an invalid or missing class label")
        if isinstance(label, (str, np.str_)):
            kinds.add("string")
            if not label.strip():
                raise ValueError(f"{name} contains an empty class label")
        elif isinstance(label, (int, float, bool, np.integer, np.floating, np.bool_)):
            kinds.add("numeric")
            if not np.isfinite(label):
                raise ValueError(f"{name} contains a nonfinite class label")
        else:
            raise ValueError(f"{name} labels must be finite numbers or nonempty strings")
    if len(kinds) != 1:
        raise ValueError(f"{name} cannot mix numeric and string labels")
    return np.asarray(values)


def _classes_and_encoding(y, n_samples: int, class_order=None):
    labels = _class_labels(y, name="y_fit")
    if len(labels) != n_samples:
        raise ValueError("y_fit must align with fit rows")
    try:
        observed = np.unique(labels)
    except TypeError as exc:
        raise ValueError("Class labels must have a consistent sortable type") from exc
    if observed.size < 2:
        raise ValueError("At least two represented classes are required")
    if class_order is None:
        classes = observed
    else:
        classes = _class_labels(class_order, name="class_order")
        try:
            if (np.unique(classes).size != classes.size
                    or set(classes.tolist()) != set(observed.tolist())):
                raise ValueError("class_order must contain every observed class exactly once")
        except TypeError as exc:
            raise ValueError("class_order must have a consistent scalar type") from exc
    lookup = {label: i for i, label in enumerate(classes.tolist())}
    encoded = np.asarray([lookup[label] for label in labels.tolist()], dtype=np.int64)
    return classes.copy(), encoded


def _indices(feature_indices, n_features: int) -> np.ndarray:
    indices = np.asarray(feature_indices)
    if (indices.ndim != 1 or indices.size < 2
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or np.any(indices < 0) or np.any(indices >= n_features)
            or np.unique(indices).size != indices.size):
        raise ValueError("feature_indices must contain at least two unique valid integer columns")
    return indices.astype(np.int64, copy=True)


def _alpha(alpha: float) -> float:
    try:
        if isinstance(alpha, (bool, np.bool_)) or np.ndim(alpha) != 0:
            raise ValueError("alpha must be a finite positive scalar")
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a finite positive scalar") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("alpha must be a finite positive scalar")
    return value


def class_statistics(X_fit, y_fit, *, class_order=None) -> ClassStatistics:
    """Aggregate occurrence totals N_yj and sample totals n_y using fit data only."""
    X = _counts(X_fit)
    classes, encoded = _classes_and_encoding(y_fit, X.shape[0], class_order)
    n_classes, n_samples = len(classes), X.shape[0]
    assignment = sparse.csr_matrix((np.ones(n_samples), (encoded, np.arange(n_samples))),
                                  shape=(n_classes, n_samples))
    totals = (assignment @ X).tocsr()
    totals.sum_duplicates()
    if not np.all(np.isfinite(totals.data)):
        raise ValueError("Class occurrence totals cannot be represented safely in float64")
    sample_counts = np.bincount(encoded, minlength=n_classes).astype(np.float64)
    return ClassStatistics(classes, sample_counts, totals, n_samples, X.shape[1])


def estimate_parameters(statistics: ClassStatistics, feature_indices, *, alpha: float) -> MNBParameters:
    """Equation (3.7), with empirical fit priors and subspace-specific smoothing."""
    smoothing = _alpha(alpha)
    if not isinstance(statistics, ClassStatistics):
        raise ValueError("statistics must be a ClassStatistics result")
    classes = _class_labels(statistics.classes, name="statistics.classes")
    if not np.isrealobj(statistics.class_counts):
        raise ValueError("Class sample counts must be real")
    sample_counts = np.asarray(statistics.class_counts, dtype=np.float64)
    if (len(classes) < 2 or np.unique(classes).size != len(classes)
            or sample_counts.shape != (len(classes),)
            or not np.all(np.isfinite(sample_counts)) or np.any(sample_counts <= 0)
            or np.any(sample_counts != np.floor(sample_counts))
            or not isinstance(statistics.n_samples, (int, np.integer))
            or statistics.n_samples != sample_counts.sum()
            or not isinstance(statistics.n_features, (int, np.integer))):
        raise ValueError("Invalid class statistics or class sample counts")
    counts = statistics.feature_counts
    if not sparse.issparse(counts) or counts.shape != (len(classes), statistics.n_features):
        raise ValueError("Invalid sparse class feature-count shape")
    # Validate only class aggregates here; do not reaggregate fit data per candidate.
    counts = _counts(counts)
    indices = _indices(feature_indices, statistics.n_features)
    selected = counts[:, indices].toarray()
    smoothed = selected + smoothing
    denominator = smoothed.sum(axis=1, keepdims=True)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        conditional = smoothed / denominator
    prior = sample_counts / statistics.n_samples
    parameters = MNBParameters(classes.copy(), indices, prior, conditional, statistics.n_features)
    return _validated_parameters(parameters)


def _validated_parameters(parameters: MNBParameters) -> MNBParameters:
    if not isinstance(parameters, MNBParameters):
        raise ValueError("parameters must be MNBParameters")
    classes = _class_labels(parameters.classes, name="parameters.classes")
    if len(classes) < 2 or np.unique(classes).size != len(classes):
        raise ValueError("Parameters need at least two unique classes")
    if (isinstance(parameters.n_features_in, (bool, np.bool_))
            or not isinstance(parameters.n_features_in, (int, np.integer))
            or parameters.n_features_in < 2):
        raise ValueError("n_features_in must be an integer at least two")
    indices = _indices(parameters.feature_indices, parameters.n_features_in)
    if not np.isrealobj(parameters.class_prior) or not np.isrealobj(parameters.feature_prob):
        raise ValueError("Model probabilities must be real")
    prior = np.asarray(parameters.class_prior, dtype=np.float64)
    conditional = np.asarray(parameters.feature_prob, dtype=np.float64)
    if (prior.shape != (len(classes),) or conditional.shape != (len(classes), len(indices))
            or not np.all(np.isfinite(prior)) or not np.all(np.isfinite(conditional))
            or np.any(prior <= 0) or np.any(conditional <= 0)
            or not np.isclose(prior.sum(), 1, rtol=0, atol=1e-12)
            or not np.allclose(conditional.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Probabilities must be finite, positive, correctly shaped and normalized")
    return MNBParameters(classes.copy(), indices, prior.copy(), conditional.copy(), parameters.n_features_in)


def class_log_scores(X, parameters: MNBParameters) -> np.ndarray:
    """Equation (3.8) scores in the fixed parameter class order.

    These are log scores up to the class-independent multinomial coefficient,
    not normalized posterior probabilities. X uses the original input columns.
    """
    parameters = _validated_parameters(parameters)
    X = _counts(X, n_features=parameters.n_features_in, allow_empty=True)
    scores = (X[:, parameters.feature_indices] @ np.log(parameters.feature_prob).T
              + np.log(parameters.class_prior))
    if not np.all(np.isfinite(scores)):
        raise ValueError("Log scores cannot be represented safely in float64")
    return np.asarray(scores, dtype=np.float64)


class SubspaceMNB:
    """Fit-only MNB, sorted classes by default; score ties select the first class."""

    def __init__(self, *, alpha: float, class_order=None):
        self.alpha = _alpha(alpha)
        self.class_order = class_order

    def fit(self, X_fit, y_fit, feature_indices):
        statistics = class_statistics(X_fit, y_fit, class_order=self.class_order)
        return self._set_parameters(estimate_parameters(statistics, feature_indices, alpha=self.alpha))

    @classmethod
    def from_statistics(cls, statistics: ClassStatistics, feature_indices, *, alpha: float):
        """Fit a candidate from reused Dfit sufficient statistics, without Dsearch."""
        parameters = estimate_parameters(statistics, feature_indices, alpha=alpha)
        model = cls(alpha=alpha, class_order=statistics.classes)
        return model._set_parameters(parameters)

    @classmethod
    def from_parameters(cls, parameters: MNBParameters):
        """Build from validated supplied probabilities; no estimation or generation."""
        parameters = _validated_parameters(parameters)
        model = cls.__new__(cls)
        model.alpha = None  # Estimation was not performed by this constructor.
        model.class_order = parameters.classes
        return model._set_parameters(parameters)

    def _set_parameters(self, parameters: MNBParameters):
        self.parameters_ = parameters
        self.classes_ = parameters.classes
        self.feature_indices_ = parameters.feature_indices
        self.class_prior_ = parameters.class_prior
        self.feature_prob_ = parameters.feature_prob
        self.n_features_in_ = parameters.n_features_in
        return self

    def predict_log_scores(self, X) -> np.ndarray:
        if not hasattr(self, "parameters_"):
            raise ValueError("SubspaceMNB must be fitted before prediction")
        return class_log_scores(X, self.parameters_)

    def predict(self, X) -> np.ndarray:
        scores = self.predict_log_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]
