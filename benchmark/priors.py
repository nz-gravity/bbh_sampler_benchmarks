"""Compatibility for Bilby 2.8.0 constrained draws with NumPy 2.4."""

from bilby.gw.prior import BBHPriorDict
import numpy as np


class BBHPriors(BBHPriorDict):
    def sample_subset_constrained(self, keys=(), size=None):
        # Bilby casts a length-one constraint array to int for size=1, which
        # NumPy 2.4 rejects. Use its scalar rejection sampler and restore shape.
        if size == 1:
            samples = super().sample_subset_constrained(keys=keys, size=None)
            return {key: np.atleast_1d(value) for key, value in samples.items()}
        return super().sample_subset_constrained(keys=keys, size=size)
