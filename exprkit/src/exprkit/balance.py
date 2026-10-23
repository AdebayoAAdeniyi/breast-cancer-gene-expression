"""Class balance without copying samples.

Copying minority samples (random oversampling) does not add information. If
done before splitting, it also places copies of the same sample on both sides
of a split, which inflates every score. Two alternatives are offered:

* class weights (the default for logistic regression and random forests):
  every class contributes equally to the loss, no rows are added or removed.
* ``UndersampleEnsemble``: each of ``n_bags`` models is trained on all
  minority samples and an equal sized random subset of each larger class,
  drawn from the training rows only; predicted probabilities are averaged.
  Every majority sample is used by some bag, so little information is lost
  (the idea of EasyEnsemble; Liu, Wu and Zhou 2009, IEEE TSMC B 39:539).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class UndersampleEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_bags: int = 10, ratio: float = 1.0, seed: int = 0):
        self.estimator = estimator
        self.n_bags = n_bags
        self.ratio = ratio
        self.seed = seed

    def fit(self, X, y, groups=None):
        from .models import fit_estimator
        y = np.asarray(y).astype(str)
        self.classes_ = np.array(sorted(set(y)))
        counts = {c: int((y == c).sum()) for c in self.classes_}
        m = min(counts.values())
        take = {c: min(n, max(m, int(round(self.ratio * m)))) for c, n in counts.items()}
        rng = np.random.default_rng(self.seed)
        self.estimators_ = []
        for _ in range(self.n_bags):
            idx = np.concatenate([rng.choice(np.flatnonzero(y == c), take[c], replace=False) for c in self.classes_])
            idx.sort()
            est = clone(self.estimator)
            Xi = X.iloc[idx] if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]
            fit_estimator(est, Xi, y[idx], None if groups is None else np.asarray(groups)[idx])
            self.estimators_.append(est)
        self.bag_sizes_ = take
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        for est in self.estimators_:
            p = est.predict_proba(X)
            cls = [str(c) for c in est.classes_]
            for j, c in enumerate(cls):
                out[:, list(self.classes_).index(c)] += p[:, j]
        return out / len(self.estimators_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
