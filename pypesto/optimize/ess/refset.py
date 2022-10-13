"""ReferenceSet functionality for scatter search."""

import numpy as np

from .function_evaluator import FunctionEvaluator


class RefSet:
    """Scatter search reference set.

    Attributes
    ----------
    dim:
        Reference set size
    evaluator:
        Function evaluator
    """

    def __init__(self, dim: int, evaluator: FunctionEvaluator):
        """Construct.

        Parameters
        ----------
        dim:
            Reference set size
        evaluator:
            Function evaluator
        x:
            Parameters in the reference set
        fx:
            Function values at the parameters in the reference set
        n_stuck:
            Counts the number of times a refset member did not lead to an
            improvement in the objective (length: ``dim``).
        """
        self.dim = dim
        self.evaluator = evaluator
        # \epsilon in [PenasGon2017]_
        self.proximity_threshold = 1e-3

        self.fx = np.full(shape=(dim,), fill_value=np.inf)
        self.x = np.full(
            shape=(dim, self.evaluator.problem.dim), fill_value=np.nan
        )
        self.n_stuck = np.zeros(shape=[dim])

    def sort(self):
        """Sort RefSet by quality."""
        order = np.argsort(self.fx)
        self.fx = self.fx[order]
        self.x = self.x[order]
        self.n_stuck = self.n_stuck[order]

    def initialize(self, n_diverse: int):
        """Create initial reference set with random parameters.

        Sample ``n_diverse`` random points, populate half of the RefSet using
        the best solutions and fill the rest with random points.
        """
        # sample n_diverse points
        x_diverse, fx_diverse = self.evaluator.multiple_random(n_diverse)

        # create initial refset with 50% best values
        num_best = int(self.dim / 2)
        order = np.argsort(fx_diverse)
        self.x[:num_best] = x_diverse[order[:num_best]]
        self.fx[:num_best] = fx_diverse[order[:num_best]]

        # ... and 50% random points
        random_idxs = np.random.choice(
            order[num_best:], size=self.dim - num_best, replace=False
        )
        self.x[num_best:] = x_diverse[random_idxs]
        self.fx[num_best:] = fx_diverse[random_idxs]

    def prune_too_close(self):
        """Prune too similar RefSet members.

        Replace a parameter vector if its maximum relative difference to a
        better parameter vector is below the given threshold.

        Assumes RefSet is sorted.
        """
        x = self.x
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # check proximity
                while (
                    np.max(np.abs((x[i] - x[j]) / x[j]))
                    <= self.proximity_threshold
                ):
                    # too close. replace x_j.
                    x[j], self.fx[j] = self.evaluator.single_random()
                    self.sort()

    def update(self, i: int, x: np.array, fx: float):
        """Update a RefSet entry."""
        self.x[i] = x
        self.fx[i] = fx
        self.n_stuck[i] = 0

    def replace_by_random(self, i: int):
        """Replace the RefSet member with the given index by a random point."""
        self.x[i], self.fx[i] = self.evaluator.single_random()
        self.n_stuck[i] = 0
