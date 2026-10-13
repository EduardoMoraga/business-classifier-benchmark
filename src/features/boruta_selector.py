from __future__ import annotations

import numpy as np
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Boruta all-relevant feature selection as a pipeline step.

    Fitting inside the pipeline keeps selection within each CV training fold, so the
    held-out fold never influences which features survive.
    """

    def __init__(self, max_depth: int = 5, perc: int = 100, max_iter: int = 60, random_state: int = 42):
        self.max_depth = max_depth
        self.perc = perc
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x, y):
        forest = RandomForestClassifier(
            n_jobs=1,
            class_weight="balanced",
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.boruta_ = BorutaPy(
            forest,
            n_estimators="auto",
            perc=self.perc,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.boruta_.fit(np.asarray(x), np.asarray(y))
        self.support_ = self.boruta_.support_
        if not self.support_.any():
            # Keep the strongest feature so downstream models always receive input.
            self.support_ = self.boruta_.ranking_ == self.boruta_.ranking_.min()
        return self

    def transform(self, x):
        return np.asarray(x)[:, self.support_]
