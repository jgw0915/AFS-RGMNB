"""Final equal-weight vote in the fixed fit-class order, equation (3.20)."""
from __future__ import annotations

import numpy as np

from .mnb import MNBParameters, SubspaceMNB


class VotingEnsemble:
    """Members share the Dfit column space. Each member indexes its own subspace."""

    def __init__(self, members):
        self.members = tuple(members)
        if not self.members:
            raise ValueError("An ensemble requires at least one member")
        self.classes = self.members[0].classes_.copy()
        self.n_features_in = self.members[0].n_features_in_
        for member in self.members:
            if (member.n_features_in_ != self.n_features_in
                    or not np.array_equal(member.classes_, self.classes)):
                raise ValueError("Members must share input columns and class order")

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)), dtype=np.int64)
        rows = np.arange(X.shape[0])
        for member in self.members:
            # argmax of member scores is its hard label's index in the fixed order.
            winner = np.argmax(member.predict_log_scores(X), axis=1)
            votes[rows, winner] += 1
        return self.classes[np.argmax(votes, axis=1)]

    def save(self, path):
        np.savez_compressed(path, classes=self.classes, n_features_in=self.n_features_in,
                            features=np.stack([m.feature_indices_ for m in self.members]),
                            priors=np.stack([m.class_prior_ for m in self.members]),
                            conditionals=np.stack([m.feature_prob_ for m in self.members]))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            features, priors, conditionals = (data[k] for k in ("features", "priors", "conditionals"))
            members = [SubspaceMNB.from_parameters(MNBParameters(data["classes"], indices, prior,
                       conditional, int(data["n_features_in"])))
                       for indices, prior, conditional in zip(features, priors, conditionals, strict=True)]
        return cls(members)
